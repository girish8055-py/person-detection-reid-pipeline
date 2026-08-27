"""
detector.py

Responsibilities:
1. Load the YOLO model.
2. Run person detection.
3. Run ByteTrack tracking.
4. Return tracked detections.
"""

from ultralytics import YOLO


def normalize_class(name: str) -> str:
    if not name:
        return "person"
    name_lower = str(name).lower().strip()
    if name_lower in ["person", "human", "pedestrian", "people", "camouflage_soldier", "soldier", "military", "man", "woman", "child"]:
        return "person"
    return "vehicle"


class PersonDetector:
    def __init__(
        self,
        model_path: str,
        confidence: float = 0.25,
        tracker: str = "bytetrack.yaml",
        min_box_width: int = 12,
        min_box_height: int = 15,
        min_box_area: int = 180,
    ):
        """
        Initialize YOLO model.

        Args:
            model_path: Path to YOLO model.
            confidence: Detection confidence threshold.
            tracker: ByteTrack configuration file.
            min_box_width: Minimum allowed bounding box width in pixels.
            min_box_height: Minimum allowed bounding box height in pixels.
            min_box_area: Minimum allowed bounding box area in sq pixels.
        """

        self.model = YOLO(model_path)
        self.confidence = confidence
        self.tracker = tracker
        self.min_box_width = min_box_width
        self.min_box_height = min_box_height
        self.min_box_area = min_box_area

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

            for idx, box in enumerate(result.boxes):

                # Use ByteTrack ID if present; fallback to box index if tracker didn't assign ID
                track_id = int(box.id.item()) if box.id is not None else (idx + 1)

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w = int(x2) - int(x1)
                h = int(y2) - int(y1)

                # Filter out tiny noise bounding boxes
                if w < self.min_box_width or h < self.min_box_height or (w * h) < self.min_box_area:
                    continue

                raw_class = self.model.names.get(int(box.cls.item()), "person")

                detections.append(
                    {
                        "track_id": track_id,
                        "bbox": [
                            int(x1),
                            int(y1),
                            int(x2),
                            int(y2)
                        ],
                        "confidence": float(box.conf.item()),
                        "class_id": int(box.cls.item()),
                        "class": normalize_class(raw_class)
                    }
                )

        return detections