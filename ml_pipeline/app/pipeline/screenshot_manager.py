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

    def save_best_detection(self, frame, person_id, detection):

        x1, y1, x2, y2 = detection["bbox"]

        area = (x2 - x1) * (y2 - y1)

        if person_id in self.best_area:

            if area <= self.best_area[person_id]:
                return

        self.best_area[person_id] = area

        pid_str = str(person_id).strip()
        lower = pid_str.lower()
        det_class = str(detection.get("class", "person")).strip().lower() if isinstance(detection, dict) else "person"

        # Keep the ID's own class prefix (person_001 / vehicle_001); only
        # fall back to class prefix for numeric-only ids.
        folder_name = lower if lower.startswith(("person_", "vehicle_")) else f"{det_class}_{lower}"

        person_folder = os.path.join(
            self.output_folder,
            folder_name
        )

        os.makedirs(person_folder, exist_ok=True)

        image = frame.copy()

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        if lower.startswith(("person", "vehicle")):
            label = pid_str
        elif isinstance(detection, dict) and detection.get("class"):
            label = f"{det_class}_{pid_str}"
        else:
            label = pid_str

        cv2.putText(
            image,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Upscale evidence below Full-HD so it reaches the admin in
        # 1080p-class quality or higher. Frames already >= 1080p
        # (e.g. 4K drone footage) are kept at their native resolution.
        h, w = image.shape[:2]
        small_side = min(h, w)
        if small_side < MIN_EVIDENCE_SIDE:
            scale = MIN_EVIDENCE_SIDE / small_side
            image = cv2.resize(
                image,
                (int(round(w * scale)), int(round(h * scale))),
                interpolation=cv2.INTER_LANCZOS4,
            )

        save_path = os.path.join(
            person_folder,
            "best_detection.jpg"
        )

        cv2.imwrite(
            save_path,
            image,
            [cv2.IMWRITE_JPEG_QUALITY, EVIDENCE_JPEG_QUALITY],
        )