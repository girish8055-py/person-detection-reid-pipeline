"""
detection_pipeline.py

Orchestrates the full flow:

    frame -> PersonDetector (YOLO + ByteTrack)
          -> crop each person
          -> IdentityManager.resolve_batch() (OSNet embeddings + ReID matching)
          -> stamp det["person_id"] on every detection
          -> PersonManager.update() (appearance / duration bookkeeping)
          -> JsonWriter.write() (final export, at the end)

Works for a video file, a webcam index, or an RTSP url -- cv2.VideoCapture
accepts all three. For single images use process_image().
"""

import time
import cv2
import os
import glob

from pipeline.detector import PersonDetector
from pipeline.identity_manager import IdentityManager
from pipeline.person_manager import PersonManager
from pipeline.json_writer import JsonWriter
from pipeline.reid.utils import crop_person


def get_latest_model_path():
    """Return path to models/best.pt."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "best.pt")


class DetectionPipeline:

    def __init__(
        self,
        model_path: str,
        reid_weights_path: str = "models/osnet_ain_x1_0_imagenet.pth",
        output_folder: str = "outputs",
        confidence: float = 0.25,
    ):
        self.detector = PersonDetector(model_path=model_path, confidence=confidence)
        self.identity_manager = IdentityManager(weights_path=reid_weights_path)
        self.person_manager = PersonManager()
        self.json_writer = JsonWriter(output_folder=output_folder)

    # ---------------------------------------------------------------- #
    # Per-frame processing
    # ---------------------------------------------------------------- #
    def process_frame(self, frame, current_time: float):
        detections = self.detector.detect(frame)

        if not detections:
            return self.person_manager.update([], current_time)

        track_ids = [d["track_id"] for d in detections]
        bboxes = [d["bbox"] for d in detections]
        confidences = [d["confidence"] for d in detections]

        # Crop every tracked person out of this frame
        crops = [crop_person(frame, bbox, pad_ratio=0.05) for bbox in bboxes]

        # One batched OSNet forward pass for the whole frame -> stable ids
        person_ids = self.identity_manager.resolve_batch(
            track_ids=track_ids,
            crops_bgr=crops,
            bboxes=bboxes,
            confidences=confidences,
            timestamp=current_time,
            classes=[d.get("class", "person") for d in detections],
        )

        for det, person_id in zip(detections, person_ids):
            det["person_id"] = person_id

        return self.person_manager.update(detections, current_time)

    # ---------------------------------------------------------------- #
    # Video / webcam / RTSP
    # ---------------------------------------------------------------- #
    def run(self, source):
        """
        source: path to a video file, an int webcam index, or an RTSP url.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_idx = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                current_time = frame_idx / fps
                self.process_frame(frame, current_time)
                frame_idx += 1
        finally:
            cap.release()

        final_time = frame_idx / fps if fps > 0 else time.time()
        self.person_manager.finalize(final_time=final_time)
        self.json_writer.write(self.person_manager.get_persons())

    # ---------------------------------------------------------------- #
    # Single image
    # ---------------------------------------------------------------- #
    def process_image(self, image_path: str):
        frame = cv2.imread(image_path)
        if frame is None:
            raise RuntimeError(f"Could not read image: {image_path}")

        current_time = 0.0
        self.process_frame(frame, current_time)

        self.person_manager.finalize(final_time=current_time)
        self.json_writer.write(self.person_manager.get_persons())


if __name__ == "__main__":
    pipeline = DetectionPipeline(
        model_path=get_latest_model_path(),
        reid_weights_path="models/osnet_ain_x1_0_imagenet.pth",
        output_folder="outputs",
    )
    pipeline.run("video.mp4")