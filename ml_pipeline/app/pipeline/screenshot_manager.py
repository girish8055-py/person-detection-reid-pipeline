import os
import cv2

# Evidence sent to the admin must be high quality: never smaller than
# Full-HD on its shortest side, saved with near-maximum JPEG quality.
MIN_EVIDENCE_SIDE = 1080
EVIDENCE_JPEG_QUALITY = 97


class ScreenshotManager:

    def __init__(self, output_folder="outputs"):
        self.output_folder = output_folder
        self.best_area = {}
        self.seen_targets = set()

    def _format_and_save_image(self, frame, detection, target_id, filename):
        """Helper to draw bounding box, class label, scale image to 1080p, and write JPEG evidence."""
        x1, y1, x2, y2 = detection["bbox"]
        tid_str = str(target_id).strip()
        lower = tid_str.lower()
        det_class = str(detection.get("class", "object")).strip().lower() if isinstance(detection, dict) else "object"

        # Keep the ID's own class prefix (e.g. Person_001, Vehicle_001, Drone_001)
        folder_name = lower if "_" in lower else f"{det_class}_{lower}"
        target_folder = os.path.join(self.output_folder, folder_name)
        os.makedirs(target_folder, exist_ok=True)

        image = frame.copy()

        # Draw green bounding box around the target
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if "_" in lower:
            label = tid_str
        elif isinstance(detection, dict) and detection.get("class"):
            label = f"{det_class}_{tid_str}"
        else:
            label = tid_str

        cv2.putText(
            image,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Upscale evidence below Full-HD so it reaches the admin in 1080p quality or higher
        h, w = image.shape[:2]
        small_side = min(h, w)
        if small_side < MIN_EVIDENCE_SIDE:
            scale = MIN_EVIDENCE_SIDE / small_side
            image = cv2.resize(
                image,
                (int(round(w * scale)), int(round(h * scale))),
                interpolation=cv2.INTER_LANCZOS4,
            )

        save_path = os.path.join(target_folder, filename)
        cv2.imwrite(save_path, image, [cv2.IMWRITE_JPEG_QUALITY, EVIDENCE_JPEG_QUALITY])

    def save_detection_evidence(self, frame, target_id, detection):
        """
        Saves 3 evidence images for ANY detected target class (person, vehicle, drone, etc.):
        - first_detection.jpg  (First frame seen)
        - middle_detection.jpg (Largest/clearest box)
        - last_detection.jpg   (Updated on every frame -> last frame seen)
        - best_detection.jpg   (Backward compatibility)
        """
        x1, y1, x2, y2 = detection["bbox"]
        area = (x2 - x1) * (y2 - y1)

        # 1. Save FIRST detection (entry frame)
        if target_id not in self.seen_targets:
            self.seen_targets.add(target_id)
            self._format_and_save_image(frame, detection, target_id, "first_detection.jpg")

        # 2. Save MIDDLE / BEST detection (largest box area)
        if target_id not in self.best_area or area > self.best_area[target_id]:
            self.best_area[target_id] = area
            self._format_and_save_image(frame, detection, target_id, "middle_detection.jpg")
            self._format_and_save_image(frame, detection, target_id, "best_detection.jpg")

        # 3. Save LAST detection (updated every frame until exit frame)
        self._format_and_save_image(frame, detection, target_id, "last_detection.jpg")

    def save_best_detection(self, frame, person_id, detection):
        """Backward-compatible call delegating to save_detection_evidence."""
        self.save_detection_evidence(frame, person_id, detection)