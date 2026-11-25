"""
detector.py

Responsibilities:
1. Load the YOLO model.
2. Run person detection.
3. Run ByteTrack tracking.
4. Return tracked detections.
"""

import os
from ultralytics import YOLO

DEFAULT_TRACKER_CONFIG = os.path.join(os.path.dirname(__file__), "bytetrack.yaml")

# Class-specific confidence and bounding box geometry filters
# Eliminates background clutter and road markings while preserving small/distant drones
DEFAULT_CLASS_THRESHOLDS = {
    "drone":   {"conf": 0.25, "min_w": 8,  "min_h": 8,  "min_area": 50},
    "person":  {"conf": 0.50, "min_w": 18, "min_h": 30, "min_area": 450},
    "vehicle": {"conf": 0.55, "min_w": 28, "min_h": 22, "min_area": 650},
}


def normalize_class(name: str) -> str:
    return str(name).lower().strip()


class PersonDetector:
    def __init__(
        self,
        model_path: str,
        confidence: float = 0.25,
        tracker: str = None,
        min_box_width: int = 0,
        min_box_height: int = 0,
        min_box_area: int = 0,
        class_thresholds: dict = None,
    ):
        """
        Initialize YOLO model.

        Args:
            model_path: Path to YOLO model.
            confidence: Base detection confidence threshold (lowest threshold for model.track).
            tracker: ByteTrack configuration file.
            min_box_width: Global fallback minimum bounding box width in pixels.
            min_box_height: Global fallback minimum bounding box height in pixels.
            min_box_area: Global fallback minimum bounding box area in sq pixels.
            class_thresholds: Dict mapping class names to {conf, min_w, min_h, min_area}.
        """

        self.model = YOLO(model_path)
        self.class_thresholds = class_thresholds or DEFAULT_CLASS_THRESHOLDS
        # Model tracker runs with the lowest threshold across classes so no candidates are missed
        min_conf = min(cfg.get("conf", confidence) for cfg in self.class_thresholds.values()) if self.class_thresholds else confidence
        self.confidence = min_conf
        self.tracker = tracker or (DEFAULT_TRACKER_CONFIG if os.path.exists(DEFAULT_TRACKER_CONFIG) else "bytetrack.yaml")
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
        # Check if CUDA has sufficient free memory (>1GB) to allocate cuBLAS workspace
        device = "cpu"
        if torch.cuda.is_available():
            try:
                free_bytes, _ = torch.cuda.mem_get_info()
                if free_bytes > 1024 * 1024 * 1024:
                    device = 0
            except Exception:
                pass
        
        try:
            results = self.model.track(
                source=frame,
                persist=True,
                tracker=self.tracker,
                conf=self.confidence,
                iou=0.50,
                device=device,
                verbose=False
            )
        except RuntimeError as oom_cuda:
            if "cuda" in str(oom_cuda).lower() and device != "cpu":
                logger.warning("CUDA memory constrained; running tracker on CPU fallback: %s", oom_cuda)
                results = self.model.track(
                    source=frame,
                    persist=True,
                    tracker=self.tracker,
                    conf=self.confidence,
                    iou=0.50,
                    device="cpu",
                    verbose=False
                )
            else:
                raise oom_cuda

        raw_detections = []

        for result in results:
            if result.boxes is None:
                continue

            for idx, box in enumerate(result.boxes):
                track_id = int(box.id.item()) if box.id is not None else (idx + 1)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w = int(x2) - int(x1)
                h = int(y2) - int(y1)

                if w <= 0 or h <= 0:
                    continue

                class_id = int(box.cls.item())
                names_dict = getattr(self.model, "names", getattr(result, "names", {}))
                raw_class = names_dict.get(class_id, f"class_{class_id}") if isinstance(names_dict, dict) else f"class_{class_id}"
                norm_class = normalize_class(raw_class)
                conf = float(box.conf.item())

                # Apply Per-Class Confidence and Geometry Filtering
                class_rule = self.class_thresholds.get(norm_class, {})
                req_conf = class_rule.get("conf", self.confidence)
                req_min_w = class_rule.get("min_w", self.min_box_width)
                req_min_h = class_rule.get("min_h", self.min_box_height)
                req_min_area = class_rule.get("min_area", self.min_box_area)

                if conf < req_conf or w < req_min_w or h < req_min_h or (w * h) < req_min_area:
                    continue

                raw_detections.append(
                    {
                        "track_id": track_id,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": conf,
                        "class_id": class_id,
                        "class": norm_class
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

                    # If IoU > 50% or smaller box has >50% overlap with higher-confidence box, drop duplicate
                    overlap_ratio = inter_area / float(min(area_a, area_b))
                    if iou > 0.50 or overlap_ratio > 0.50:
                        keep = False
                        break

            if keep:
                detections.append(det)

        return detections


# Generic class-agnostic alias for multi-class target detection
ObjectDetector = PersonDetector