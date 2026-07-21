"""
detector.py

Responsibilities:
1. Load the YOLO model.
2. Run person detection.
3. Run ByteTrack tracking.
4. Return tracked detections.
"""

from ultralytics import YOLO


class PersonDetector:
    def __init__(
        self,
        model_path: str,
        confidence: float = 0.25,
        tracker: str = "bytetrack.yaml"
    ):
        """
        Initialize YOLO model.

        Args:
            model_path: Path to YOLO model.
            confidence: Detection confidence threshold.
            tracker: ByteTrack configuration file.
        """

        self.model = YOLO(model_path)
        self.confidence = confidence
        self.tracker = tracker

    def detect(self, frame):
        """
        Detect and track people in a frame.

        Returns:
            List of tracked detections.
        """

        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker,
            conf=self.confidence,
            verbose=True
        )

        detections = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                # Skip detections without a tracking ID
                if box.id is None:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append(
                    {
                        "track_id": int(box.id.item()),
                        "bbox": [
                            int(x1),
                            int(y1),
                            int(x2),
                            int(y2)
                        ],
                        "confidence": float(box.conf.item()),
                        "class_id": int(box.cls.item())
                    }
                )

        return detections