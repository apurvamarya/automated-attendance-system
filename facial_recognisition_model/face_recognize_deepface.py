import cv2
import pandas as pd
from datetime import datetime
import os
import logging
import threading
import requests
from deepface import DeepFace

# ---------------- LOGGING SETUP ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("attendance_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------- CONFIG ----------------
SERVER_URL            = "https://attendance-system-9ptl.onrender.com/upload-attendance"
API_KEY               = "attendance_upload_key"
DATASET_PATH          = "Project/automated-attendance-system/facial_recognisition_model/dataset"
ATTENDANCE_FILE       = "Project/automated-attendance-system/dashboard/attendance/attendance.csv"

CAMERA_INDEX          = 1
FRAME_WIDTH           = 640
FRAME_HEIGHT          = 480
FRAME_SKIP            = 10
CONFIDENCE_THRESHOLD  = 60.0
MIN_BRIGHTNESS        = 30 
MIN_BLUR_SCORE        = 50

# ---------------- CAMERA SETUP ----------------
cam = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cam.isOpened():
    logger.error("Camera not accessible. Check CAMERA_INDEX.")
    exit()

# ---------------- ATTENDANCE FILE INIT ----------------
os.makedirs(os.path.dirname(ATTENDANCE_FILE), exist_ok=True)
if not os.path.exists(ATTENDANCE_FILE):
    pd.DataFrame(columns=["Name", "Date", "Time", "Status"]).to_csv(ATTENDANCE_FILE, index=False)
    logger.info("Attendance file created.")

# ---------------- STATE ----------------
marked_today   = set()
frame_count    = 0
name           = "Unknown"
confidence_pct = 0.0

# ---------------- CLOUD UPLOAD (non-blocking) ----------------
def upload_attendance(payload: dict) -> None:
    """Upload a single attendance record to the server in a background thread."""
    try:
        response = requests.post(
            SERVER_URL,
            json=payload,
            headers={"X-API-KEY": API_KEY},
            timeout=5
        )
        response.raise_for_status()
        logger.info(f"Uploaded attendance for {payload['Name']} — HTTP {response.status_code}")
    except requests.exceptions.HTTPError as e:
        logger.warning(f"Server error during upload: {e}")
    except requests.exceptions.RequestException as e:
        logger.warning(f"Upload failed (network): {e}")

# ---------------- MARK ATTENDANCE ----------------
def mark_attendance(person_name: str) -> None:
    """Write attendance to CSV and trigger a background cloud upload."""
    now      = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    key      = (person_name, date_str)

    if key in marked_today:
        return

    marked_today.add(key)

    df = pd.read_csv(ATTENDANCE_FILE)
    df.loc[len(df)] = [person_name, date_str, time_str, "Present"]
    df.to_csv(ATTENDANCE_FILE, index=False)
    logger.info(f"Attendance marked — {person_name} at {time_str}")

    payload = {"Name": person_name, "Date": date_str, "Time": time_str, "Status": "Present"}
    threading.Thread(target=upload_attendance, args=(payload,), daemon=True).start()

# ---------------- FRAME QUALITY CHECK ----------------
def is_frame_valid(frame) -> tuple[bool, str]:
    """
    Returns (True, '') if the frame is usable.
    Returns (False, reason) if the frame is too dark or blocked.
    """
    gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    if brightness < MIN_BRIGHTNESS:
        return False, f"Too dark (brightness={brightness:.1f})"
    if blur_score < MIN_BLUR_SCORE:
        return False, f"Blocked/blurry (blur={blur_score:.1f})"
    return True, ""

# ---------------- MAIN LOOP ----------------
logger.info("Starting DeepFace Attendance System. Press 'q' to quit.")

while True:
    ret, frame = cam.read()
    if not ret:
        logger.error("Failed to read frame. Exiting.")
        break

    frame_count += 1

    # ---------- Run DeepFace every N frames ----------
    if frame_count % FRAME_SKIP == 0:

        valid, reject_reason = is_frame_valid(frame)

        if not valid:
            # Frame is dark or blocked — reset and warn
            name           = "Unknown"
            confidence_pct = 0.0
            logger.debug(f"Frame rejected — {reject_reason}")

        else:
            try:
                result = DeepFace.find(
                    img_path=frame,
                    db_path=DATASET_PATH,
                    enforce_detection=False,
                    model_name="Facenet",
                    distance_metric="cosine",
                    silent=True
                )

                if result and len(result[0]) > 0:
                    top_match      = result[0].iloc[0]
                    identity_path  = top_match["identity"]
                    distance       = top_match["distance"]
                    confidence     = 1.0 - distance
                    confidence_pct = round(max(0.0, min(100.0, confidence * 100)), 1)

                    if confidence_pct >= CONFIDENCE_THRESHOLD:
                        name = os.path.basename(os.path.dirname(identity_path))
                        mark_attendance(name)
                    else:
                        name = "Unknown"
                else:
                    name           = "Unknown"
                    confidence_pct = 0.0

            except Exception as e:
                logger.debug(f"Recognition error: {e}")
                name           = "Unknown"
                confidence_pct = 0.0

    # ---------- Draw overlay ----------
    label_color  = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
    display_text = f"{name} ({confidence_pct:.1f}%)"

    cv2.putText(
        frame, display_text,
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0, label_color, 2,
        cv2.LINE_AA
    )

    # Status bar at bottom
    status = "PRESENT" if name != "Unknown" else "SCANNING..."
    cv2.putText(
        frame, status,
        (20, FRAME_HEIGHT - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, label_color, 1,
        cv2.LINE_AA
    )

    cv2.imshow("DeepFace Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        logger.info("Quit signal received.")
        break

# ---------------- CLEANUP ----------------
cam.release()
cv2.destroyAllWindows()
logger.info("System shut down cleanly.")