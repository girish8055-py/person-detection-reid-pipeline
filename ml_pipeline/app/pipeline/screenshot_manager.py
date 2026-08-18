import os
import cv2


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

        pid_str = str(person_id)
        folder_name = pid_str.lower() if pid_str.lower().startswith("person_") else f"person_{pid_str}"

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

        cv2.putText(
            image,
            f"Person ID : {person_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        save_path = os.path.join(
            person_folder,
            "best_detection.jpg"
        )

        cv2.imwrite(save_path, image)