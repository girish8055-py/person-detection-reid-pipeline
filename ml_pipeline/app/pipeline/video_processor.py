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
from pipeline.telemetry import TelemetryReader

logger = logging.getLogger(__name__)


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

    def process_video(self, video_path: str, output_folder: str, progress_callback=None, teledata_csv_path: str = None, video_start_epoch=None, model_path: str = None, model_name: str = None):
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
        if teledata_csv_path:
            try:
                telemetry_reader = TelemetryReader(teledata_csv_path, video_start_epoch=video_start_epoch)
                logger.info("Telemetry loaded: %d GPS samples from %s", len(telemetry_reader.samples), teledata_csv_path)
            except Exception:
                logger.exception("Failed to load telemetry CSV %s -- continuing without GPS tagging", teledata_csv_path)
                telemetry_reader = None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        skip_interval = max(1, int(video_fps / self.target_fps))

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
                    )

                    for det, person_id in zip(detections, person_ids):
                        det["person_id"] = person_id

                        if telemetry_reader is not None:
                            position = telemetry_reader.get_position(current_time)
                            if position:
                                det["lat"] = position["lat"]
                                det["lon"] = position["lon"]
                                det["alt"] = position["alt"]
                                det["distance"] = position["distance"]
                                det["heading"] = position["yaw"]
                                det["detection_time"] = position["iso_time"]

                    detections = person_manager.update(detections, current_time)

                    for det in detections:
                        screenshot_manager.save_best_detection(frame, det["person_id"], det)

            if total_frames > 0 and frame_count % 30 == 0:
                pct = min(99.0, 100.0 * frame_count / total_frames)
                report(pct, f"Processed frame {frame_count}/{int(total_frames)}")

        cap.release()

        final_time = frame_count / video_fps if video_fps > 0 else time.time()
        person_manager.finalize(final_time)

        persons = person_manager.get_persons()
        json_writer.write(persons)

        report(100.0, f"Done. {len(persons)} unique person(s) detected.")
        return persons