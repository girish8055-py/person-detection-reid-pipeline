"""
pipeline/video_processor.py

Same processing logic as your main.py, refactored into a callable function
so it can be triggered from the web app (Flask) instead of only running as
a standalone script. No cv2.imshow / waitKey here -- this runs headless on
a server, so no GUI window.

Model loading (YOLO detector + OSNet ReID extractor) happens ONCE per
VideoProcessor instance and is reused across videos, since loading these
models is the expensive part. person_manager and screenshot_manager are
created fresh per video, since their state (appearances, durations) belongs
to a single video/job, not the whole app.
"""

import logging
import os
import time

import cv2

from pipeline.detector import PersonDetector
from pipeline.person_manager import PersonManager
from pipeline.json_writer import JsonWriter
from pipeline.screenshot_manager import ScreenshotManager
from pipeline.identity_manager import IdentityManager
from pipeline.reid.utils import crop_person
from pipeline.reid.config import prefix_for_class
from pipeline.telemetry import TelemetryReader
from pipeline.utils import format_time, get_severity_from_label, resolve_severity, ensure_detailed_suggested_action, limit_text_words
from pipeline.vlm_verifier import verify_object_class

logger = logging.getLogger(__name__)


def _rename_identity(persons, old_id, new_id, p_data, output_folder):
    """
    Re-key a person entry so its ID matches its (corrected) class, and move
    the on-disk evidence folder (screenshots / behavior clip / keyframes)
    to the new name.
    """
    if old_id == new_id or old_id not in persons:
        return old_id

    # Rebuild dict preserving insertion order with the new key
    rebuilt = {}
    for k, v in persons.items():
        if k == old_id:
            v["person_id"] = new_id
            rebuilt[new_id] = v
        else:
            rebuilt[k] = v
    persons.clear()
    persons.update(rebuilt)

    # Move evidence folder (screenshot_manager and behavior clips share it)
    old_folder = os.path.join(output_folder, str(old_id).lower())
    new_folder = os.path.join(output_folder, str(new_id).lower())
    if os.path.isdir(old_folder) and old_folder != new_folder:
        try:
            os.rename(old_folder, new_folder)
        except OSError as exc:
            logger.warning("Could not rename %s -> %s: %s", old_folder, new_folder, exc)
            return new_id

    def _fix_path(path):
        if not path:
            return path
        path = str(path)
        if os.path.basename(os.path.dirname(path)) == str(old_id).lower():
            return os.path.join(new_folder, os.path.basename(path))
        return path

    p_data["behavior_clip"] = _fix_path(p_data.get("behavior_clip"))
    p_data["keyframes"] = [_fix_path(kp) for kp in p_data.get("keyframes", [])]
    return new_id


class VideoProcessor:

    def __init__(self, model_path: str, reid_weights_path: str, confidence: float = 0.5, target_fps: int = 10):
        """
        Loads YOLO + OSNet once. Reuse this instance across multiple videos
        so you don't pay model-loading cost per job.
        """
        self.detector = PersonDetector(model_path=model_path, confidence=confidence) if model_path else None
        self._current_model_path = model_path
        # NOTE: identity_manager is intentionally re-created per video in
        # process_video() below -- identities (Person_001, Person_002...)
        # should not carry over between unrelated videos. Only the OSNet
        # weights path is fixed here; the gallery/database itself is fresh
        # per job.
        self.reid_weights_path = reid_weights_path
        self.target_fps = target_fps

    def process_video(self, video_path: str, output_folder: str, progress_callback=None, teledata_csv_path: str = None, telemetry_data: list = None, video_start_epoch=None, model_path: str = None, model_name: str = None):
        """
        Runs the full pipeline on a single video file.

        Args:
            video_path: local path to the downloaded video.
            output_folder: where json_writer / screenshot_manager write results
                for THIS job (keep jobs in separate folders so results don't
                overwrite each other).
            progress_callback: optional callable(percent: float, message: str)
                called periodically so the web app can report progress.
            teledata_csv_path: optional path to the drone's paired
                offline_teledata_<flight_id>.csv. If given, every detection
                gets tagged with lat/lon/alt/detection_time. If None, GPS
                fields are simply omitted (backward compatible).
            telemetry_data: optional in-memory telemetry data list.
            video_start_epoch: unix timestamp marking when the VIDEO started
                recording. Passed to TelemetryReader.

        Returns:
            dict: {person_id: person_data} -- same shape as person_manager.get_persons()
        """
        if model_path is not None:
            if self._current_model_path != model_path:
                logger.info("Reloading detector model from: %s", model_path)
                confidence = self.detector.confidence if self.detector else 0.5
                self.detector = PersonDetector(model_path=model_path, confidence=confidence)
                self._current_model_path = model_path

        if self.detector is None:
            raise RuntimeError("No object detection model loaded. Please specify a valid model path.")

        # Print the model name being used for detection
        file_name = os.path.basename(self._current_model_path) if self._current_model_path else "default"
        display_name = f"{model_name} ({file_name})" if model_name else file_name
        print(f"\n[DETECTION ENGINE] Active model being used for detection: {display_name}\n")
        logger.info("[DETECTION ENGINE] Active model being used for detection: %s", display_name)

        identity_manager = IdentityManager(weights_path=self.reid_weights_path)
        person_manager = PersonManager()
        json_writer = JsonWriter(output_folder=output_folder)
        screenshot_manager = ScreenshotManager(output_folder=output_folder)

        telemetry_reader = None
        if telemetry_data is not None or teledata_csv_path:
            try:
                telemetry_reader = TelemetryReader(
                    teledata_csv_path=teledata_csv_path,
                    telemetry_data=telemetry_data,
                    video_start_epoch=video_start_epoch
                )
                if telemetry_data is not None:
                    logger.info("Telemetry loaded: %d GPS samples from in-memory data", len(telemetry_reader.samples))
                else:
                    logger.info("Telemetry loaded: %d GPS samples from %s", len(telemetry_reader.samples), teledata_csv_path)
            except Exception:
                logger.exception("Failed to load telemetry -- continuing without GPS tagging")
                telemetry_reader = None

        if not os.path.exists(video_path):
            raise RuntimeError(f"Video file does not exist at path: {video_path}")

        file_size = os.path.getsize(video_path)
        if file_size == 0:
            raise RuntimeError(f"Video file is empty (0 bytes): {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file ({file_size} bytes, potentially corrupted or unsupported format): {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        skip_interval = max(1, int(video_fps / self.target_fps))

        # Prepare VideoWriter to save detection video with bounding boxes
        annotated_video_path = os.path.join(output_folder, "detection_annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(annotated_video_path, fourcc, video_fps, (width, height))

        frame_count = 0
        detections = []

        def report(pct, msg):
            if progress_callback:
                progress_callback(pct, msg)
            logger.info("[%.1f%%] %s", pct, msg)

        report(0.0, "Starting video processing")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

            if frame_count % skip_interval == 0:
                detections = self.detector.detect(frame)

                if len(detections) > 0:
                    track_ids = [det["track_id"] for det in detections]
                    bboxes = [det["bbox"] for det in detections]
                    confidences = [det["confidence"] for det in detections]

                    crops = [crop_person(frame, bbox, pad_ratio=0.05) for bbox in bboxes]

                    person_ids = identity_manager.resolve_batch(
                        track_ids=track_ids,
                        crops_bgr=crops,
                        bboxes=bboxes,
                        confidences=confidences,
                        timestamp=current_time,
                        classes=[det.get("class", "person") for det in detections],
                    )

                    for det, person_id in zip(detections, person_ids):
                        det["person_id"] = person_id
                        det["frame_num"] = frame_count

                        if telemetry_reader is not None:
                            position = telemetry_reader.get_position(
                                current_time,
                                bbox=det.get("bbox"),
                                frame_size=(width, height),
                                obj_class=det.get("class", "person"),
                            )
                            if position:
                                det["lat"] = position["lat"]
                                det["lon"] = position["lon"]
                                det["alt"] = position["alt"]
                                det["distance"] = position["distance"]
                                det["heading"] = position["yaw"]
                                det["detection_time"] = position["iso_time"]

                    detections = person_manager.update(detections, current_time)

                    det_info = [
                        f"Track #{det['track_id']} ({det.get('person_id', 'N/A')}): class='{det.get('class', 'person')}' (conf={det['confidence']:.2f})"
                        for det in detections
                    ]
                    print(f"[DETECTION] Frame {frame_count} @ {current_time:.2f}s | {len(detections)} detection(s): {', '.join(det_info)}")

                    for det in detections:
                        screenshot_manager.save_best_detection(frame, det["person_id"], det)

            # Draw bounding boxes and IDs on the annotated video frame
            if detections and video_writer.isOpened():
                annotated_frame = frame.copy()
                for det in detections:
                    bbox = det.get("bbox")
                    if bbox and len(bbox) >= 4:
                        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
                        person_id = det.get("person_id", "N/A")
                        conf = det.get("confidence", 0.0)
                        pid_str = str(person_id).strip()
                        lower = pid_str.lower()
                        if lower.startswith(("person", "vehicle")):
                            display_id = pid_str
                        elif det.get("class"):
                            display_id = f"{det['class']}_{pid_str}"
                        else:
                            display_id = pid_str
                        label = f"{display_id} ({conf:.2f})"
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, label, (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                video_writer.write(annotated_frame)
            elif video_writer.isOpened():
                video_writer.write(frame)

            if total_frames > 0 and frame_count % 30 == 0:
                pct = min(99.0, 100.0 * frame_count / total_frames)
                report(pct, f"Processed frame {frame_count}/{int(total_frames)}")

        cap.release()
        if video_writer.isOpened():
            video_writer.release()
            logger.info("Saved annotated verification video to: %s", annotated_video_path)

        final_time = frame_count / video_fps if video_fps > 0 else time.time()
        person_manager.finalize(final_time)

        persons = person_manager.get_persons()

        # Extract entry and exit keyframes (and behavior clip) for each detected person
        for p_id, p_data in persons.items():
            appearances = p_data.get("appearances", [])
            best_app = max(appearances, key=lambda x: x.get("duration", 0), default=None)
            if best_app:
                overall_entry = appearances[0]["entry_time"] if appearances else best_app["entry_time"]
                overall_exit = appearances[-1]["exit_time"] if appearances else best_app["exit_time"]
                best_time = (best_app["entry_time"] + best_app["exit_time"]) / 2

                clip_path, kf_paths = extract_behavior_clip_and_keyframes(
                    video_path=video_path,
                    best_time=best_time,
                    fps=video_fps,
                    output_folder=output_folder,
                    person_id=p_id,
                    entry_time=overall_entry,
                    exit_time=overall_exit
                )
                p_data["behavior_clip"] = clip_path
                p_data["keyframes"] = kf_paths

        # ------------------------------------------------------------------
        # VLM class verification: let the VLM independently decide whether
        # each tracked object is a person or vehicle, correcting any YOLO
        # misclassifications before the final result is built.
        # ------------------------------------------------------------------
        print("\n" + "=" * 70)
        print("[VLM CLASS VERIFICATION] Verifying detected object classes via VLM...")
        print("=" * 70)

        for p_id, p_data in persons.items():
            keyframes = p_data.get("keyframes", [])
            detected_class = p_data.get("class", "person")
            if keyframes:
                verified_class = verify_object_class(
                    keyframe_paths=keyframes,
                    target_id=str(p_id),
                    fallback_class=detected_class,
                )
                if verified_class != detected_class:
                    p_data["class"] = verified_class
                    print(
                        f"[VLM VERIFY] Target {p_id}: "
                        f"class corrected '{detected_class}' -> '{verified_class}'"
                    )
                else:
                    print(
                        f"[VLM VERIFY] Target {p_id}: "
                        f"class confirmed '{verified_class}'"
                    )
            else:
                print(
                    f"[VLM VERIFY] Target {p_id}: "
                    f"no keyframes, keeping '{detected_class}'"
                )

        print("=" * 70 + "\n")

        # ------------------------------------------------------------------
        # VLM behavior analysis: description, suggested action and SEVERITY
        # are decided by the VLM from the keyframes. Severity is NOT derived
        # from the ReID count anymore -- the VLM is the single source of truth.
        # ------------------------------------------------------------------
        print("\n" + "=" * 70)
        print("[VLM THREAT ANALYSIS] Assessing description/action/severity via VLM...")
        print("=" * 70)

        from worker.worker import get_vlm_behavior_description_and_action

        for p_id, p_data in persons.items():
            kf_paths = p_data.get("keyframes", [])
            appearances = p_data.get("appearances", [])
            pseudo_result = {
                "id_number": str(p_id),
                "class": p_data.get("class", "person"),
                "total_duration": p_data.get("total_duration", 0),
                "entry_time": format_time(
                    appearances[0]["entry_time"] if appearances else p_data.get("current_entry", 0)
                ),
                "exit_time": format_time(
                    appearances[-1]["exit_time"] if appearances else p_data.get("last_seen", 0)
                ),
                "reidentification_count": p_data.get("reidentification_count", 0),
                "appearances": appearances,
            }
            description, suggest_action, vlm_class, vlm_severity = (
                get_vlm_behavior_description_and_action(pseudo_result, kf_paths)
            )
            p_data["description"] = limit_text_words(description, 30)
            if vlm_class and vlm_class != p_data.get("class"):
                p_data["class"] = vlm_class

            reid_cnt = p_data.get("reidentification_count", 0)
            sev_level, sev_label = resolve_severity(vlm_severity, reid_cnt)
            p_data["severity_level"], p_data["severity"] = sev_level, sev_label
            p_data["suggest_action"] = ensure_detailed_suggested_action(
                suggest_action,
                sev_label,
                str(p_id),
                p_data.get("total_duration", 0),
                reid_cnt
            )
            print(
                f"[VLM ANALYSIS] Target {p_id}: "
                f"Severity = Level {sev_level} ({sev_label}) | Action = {p_data['suggest_action']}"
            )

        print("=" * 70 + "\n")

        json_writer.write(persons)

        print("\n" + "=" * 70)
        print("[DETECTION OUTPUT SUMMARY]")
        print(f"Total Unique Persons Detected: {len(persons)}")
        for p_id, p_data in persons.items():
            reid_count = p_data.get("reidentification_count", 0)
            print(
                f"  -> {p_id}: Total Duration = {p_data.get('total_duration', 0):.2f}s | "
                f"Appearances = {len(p_data.get('appearances', []))} | "
                f"ReID Count = {reid_count} | "
                f"Severity = {p_data.get('severity') or 'N/A (no VLM decision)'}"
            )
        print("=" * 70 + "\n")

        report(100.0, f"Done. {len(persons)} unique person(s) detected.")
        return persons


def extract_behavior_clip_and_keyframes(video_path: str, best_time: float, fps: float, output_folder: str, person_id: str, entry_time: float = None, exit_time: float = None):
    """
    Extracts keyframes at entry_time, middle, and exit_time, as well as a behavior clip.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Could not open video to extract clip for person %s", person_id)
        return None, []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_sec = total_frames / fps if fps > 0 else 0

    # Calculate frame range
    if entry_time is None or exit_time is None:
        start_time = max(0.0, best_time - 2.0)
        end_time = min(total_sec, start_time + 5.0)
    else:
        start_time = max(0.0, entry_time)
        end_time = min(total_sec, max(exit_time, entry_time + 1.0))

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    pid_str = str(person_id).lower()
    # Keep the ID's own class prefix (person_001 / vehicle_001); only
    # fall back to person_ for legacy numeric-only ids.
    folder_name = pid_str if pid_str.startswith(("person_", "vehicle_")) else f"person_{pid_str}"
    person_folder = os.path.join(output_folder, folder_name)
    os.makedirs(person_folder, exist_ok=True)

    clip_path = os.path.join(person_folder, "behavior_clip.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    
    current_frame = start_frame
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frames.append(frame.copy())
        current_frame += 1

    cap.release()
    out.release()

    # Extract keyframes (entry frame, middle frame, exit frame)
    keyframe_paths = []
    if frames:
        entry_kf_path = os.path.join(person_folder, "entry_frame.jpg")
        cv2.imwrite(entry_kf_path, frames[0])
        keyframe_paths.append(entry_kf_path)

        if len(frames) > 2:
            mid_kf_path = os.path.join(person_folder, "middle_frame.jpg")
            cv2.imwrite(mid_kf_path, frames[len(frames) // 2])
            keyframe_paths.append(mid_kf_path)

        exit_kf_path = os.path.join(person_folder, "exit_frame.jpg")
        cv2.imwrite(exit_kf_path, frames[-1])
        keyframe_paths.append(exit_kf_path)

    return clip_path, keyframe_paths