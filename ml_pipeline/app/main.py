import cv2
import logging
import os
logging.basicConfig(level=logging.DEBUG)
from pipeline.detector import PersonDetector
from pipeline.person_manager import PersonManager
from pipeline.json_writer import JsonWriter
from pipeline.screenshot_manager import ScreenshotManager
from pipeline.identity_manager import IdentityManager
from pipeline.reid.utils import crop_person
from pipeline.telemetry import TelemetryReader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
output_folder = os.path.join(BASE_DIR, "outputs")

# once, near your other manager initializations:
telemetry_reader = TelemetryReader(
    teledata_csv_path=os.path.join(BASE_DIR, "MK15 12-11-25", "offline_teledata_67d53de3-7e4b-4d0f-b913-f0776058dca8.csv")
)
# -------------------------------------------------------
# Initialize Managers
# -------------------------------------------------------
person_manager = PersonManager()
identity_manager = IdentityManager(weights_path=os.path.join(BASE_DIR, "models", "dinov2_vitb14_pretrain.pth"))
json_writer = JsonWriter()
screenshot_manager = ScreenshotManager(output_folder=output_folder)
#screenshot_manager = ScreenshotManager()
# -------------------------------------------------------
# Paths
# -------------------------------------------------------
DEFAULT_MODEL = os.path.join(BASE_DIR, "downloaded_model", "best.pt")
if os.path.exists(DEFAULT_MODEL):
    MODEL_PATH = DEFAULT_MODEL
else:
    print(f"Model URL is not present and downloaded model is missing at: {DEFAULT_MODEL}")
    MODEL_PATH = DEFAULT_MODEL
VIDEO_PATH = os.path.join(BASE_DIR, "1.mp4")

# -------------------------------------------------------
# Initialize Detector
# -------------------------------------------------------
detector = PersonDetector(
    model_path=MODEL_PATH,
    confidence=0.25
)

# -------------------------------------------------------
# Open Video
# -------------------------------------------------------
cap = cv2.VideoCapture(VIDEO_PATH)

video_fps = cap.get(cv2.CAP_PROP_FPS)
target_fps = 10
skip_interval = max(1, int(video_fps / target_fps))

frame_count = 0
detections = []

# -------------------------------------------------------
# Process Video
# -------------------------------------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

    # -------------------------------------------------------
    # Run detection every Nth frame
    # -------------------------------------------------------
    if frame_count % skip_interval == 0:

        detections = detector.detect(frame)

        if len(detections) > 0:

            # -----------------------------------------------
            # Identity Manager
            # -----------------------------------------------
            # -----------------------------------------------
            # Identity Manager
            # -----------------------------------------------
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
            # -----------------------------------------------
            # Person Manager
            # -----------------------------------------------
            detections = person_manager.update(
                detections,
                current_time
            )
            for det, person_id in zip(detections, person_ids):
                det["person_id"] = person_id

                position = telemetry_reader.get_position(current_time)
                if position:
                    det["lat"] = position["lat"]
                    det["lon"] = position["lon"]
                    det["alt"] = position["alt"]
                    det["detection_time"] = position["iso_time"]
            # -----------------------------------------------
            # Save Best Screenshot
            # -----------------------------------------------
            for det in detections:

                target_id = det.get("target_id") or det["person_id"]
                screenshot_manager.save_detection_evidence(
                    frame,
                    target_id,
                    det
                )

    # -------------------------------------------------------
    # Draw Bounding Boxes
    # -------------------------------------------------------
    for det in detections:

        x1, y1, x2, y2 = det["bbox"]

        person_id = det["person_id"]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        pid_str = str(person_id).strip()
        lower = pid_str.lower()
        if lower.startswith(("person", "vehicle")):
            label = pid_str
        elif det.get("class"):
            label = f"{det['class']}_{pid_str}"
        else:
            label = pid_str

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow("Person Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -------------------------------------------------------
# Finalize Persons
# -------------------------------------------------------
final_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

person_manager.finalize(final_time)

persons = person_manager.get_persons()

json_writer.write(persons)

# -------------------------------------------------------
# Print Results
# -------------------------------------------------------
print("\nDetected Persons:\n")

for person in persons.values():
    print(person)

cap.release()
cv2.destroyAllWindows()