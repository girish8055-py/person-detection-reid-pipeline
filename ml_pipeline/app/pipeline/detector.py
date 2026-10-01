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
    return str(name).lower().strip()


class PersonDetector:
    def __init__(
        self,
        model_path: str,
        confidence: float = 0.25,
        tracker: str = "bytetrack.yaml",
        min_box_width: int = 25,
        min_box_height: int = 35,
        min_box_area: int = 800,
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

    def get_model_classes(self):
        """Returns the list of normalized class names supported by the loaded YOLO model."""
        names_dict = getattr(self.model, "names", {})
        if isinstance(names_dict, dict):
            return [normalize_class(v) for v in names_dict.values()]
        elif isinstance(names_dict, (list, tuple)):
            return [normalize_class(v) for v in names_dict]
        return []

    def detect(self, frame):
        """
        Detect and track people in a frame.

        Returns:
            List of tracked detections.
        """

        import torch
        device = 0 if torch.cuda.is_available() else "cpu"
        
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker,
            conf=self.confidence,
            device=device,
            verbose=False
        )

        raw_detections = []

        for result in results:
            if result.boxes is None:
                continue

            for idx, box in enumerate(result.boxes):
                track_id = int(box.id.item()) if box.id is not None else (idx + 1)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w = int(x2) - int(x1)
                h = int(y2) - int(y1)

                if w < self.min_box_width or h < self.min_box_height or (w * h) < self.min_box_area:
                    continue

                class_id = int(box.cls.item())
                names_dict = getattr(self.model, "names", getattr(result, "names", {}))
                raw_class = names_dict.get(class_id, f"class_{class_id}") if isinstance(names_dict, dict) else f"class_{class_id}"

                raw_detections.append(
                    {
                        "track_id": track_id,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": float(box.conf.item()),
                        "class_id": class_id,
                        "class": normalize_class(raw_class)
                    }
                )

        # Apply IoU Non-Maximum Suppression (NMS) to eliminate duplicate overlapping boxes
        detections = []
        # Sort by confidence descending
        raw_detections.sort(key=lambda d: d["confidence"], reverse=True)

        for det in raw_detections:
            keep = True
            x1_a, y1_a, x2_a, y2_a = det["bbox"]
            area_a = (x2_a - x1_a) * (y2_a - y1_a)

            for kept in detections:
                x1_b, y1_b, x2_b, y2_b = kept["bbox"]
                # Compute intersection
                inter_x1 = max(x1_a, x1_b)
                inter_y1 = max(y1_a, y1_b)
                inter_x2 = min(x2_a, x2_b)
                inter_y2 = min(y2_a, y2_b)

                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    area_b = (x2_b - x1_b) * (y2_b - y1_b)
                    iou = inter_area / float(area_a + area_b - inter_area)

                    # If IoU > 25% or smaller box has >40% overlap with higher-confidence box, drop duplicate
                    overlap_ratio = inter_area / float(min(area_a, area_b))
                    if iou > 0.25 or overlap_ratio > 0.40:
                        keep = False
                        break

            if keep:
                detections.append(det)

        return detections


# Generic class-agnostic alias for multi-class target detection
ObjectDetector = PersonDetector