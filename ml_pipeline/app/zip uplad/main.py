import cv2
import logging
logging.basicConfig(level=logging.DEBUG)
from pipeline.detector import PersonDetector
from pipeline.person_manager import PersonManager
from pipeline.json_writer import JsonWriter
from pipeline.screenshot_manager import ScreenshotManager
from pipeline.identity_manager import IdentityManager
from pipeline.reid.utils import crop_person
# -------------------------------------------------------
# Initialize Managers
# -------------------------------------------------------
person_manager = PersonManager()
identity_manager = IdentityManager(weights_path=r"D:\deployment_model\ml_pipeline\app\models\osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth")
json_writer = JsonWriter()
screenshot_manager = ScreenshotManager(output_folder=output_folder)
#screenshot_manager = ScreenshotManager()
# -------------------------------------------------------
# Paths
# -------------------------------------------------------
MODEL_PATH = r"D:\deployment_model\ml_pipeline\app\models\best.pt"
VIDEO_PATH = r"D:\deployment_model\9.mp4"

# -------------------------------------------------------
# Initialize Detector
# -------------------------------------------------------
detector = PersonDetector(
    model_path=MODEL_PATH,
    confidence=0.5
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

            # -----------------------------------------------
            # Save Best Screenshot
            # -----------------------------------------------
            for det in detections:

                screenshot_manager.save_best_detection(
                    frame,
                    det["person_id"],
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

        cv2.putText(
            frame,
            f"Person ID : {person_id}",
            (x1, y1 - 10),
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