import cv2
import os
import time

# ---------------- CONFIG ----------------
CAMERA_INDEX   = 1 
FRAME_WIDTH    = 640
FRAME_HEIGHT   = 480
MAX_IMAGES     = 60
MIN_FACE_SIZE  = 80 
CAPTURE_DELAY  = 0.3
DATASET_ROOT   = "Project/automated-attendance-system/facial_recognisition_model/dataset"

# ---------------- INPUT ----------------
student_name = input("Enter Student Name: ").strip()
if not student_name:
    print("Error: Student name cannot be empty.")
    exit()

dataset_path = os.path.join(DATASET_ROOT, student_name)
os.makedirs(dataset_path, exist_ok=True)
print(f"Saving images to: {dataset_path}")
print("Look at the camera. Press 'q' to quit early.")

# ---------------- CAMERA SETUP ----------------
cam = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cam.isOpened():
    print("Error: Camera not accessible. Check CAMERA_INDEX.")
    exit()

# ---------------- FACE DETECTOR ----------------
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- STATE ----------------
count          = 0
last_save_time = 0.0

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Failed to read frame.")
        break

    # Use grayscale only for detection; save color crops for DeepFace
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray  = cv2.equalizeHist(gray)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)   # Filter out tiny detections
    )

    now = time.time()

    for (x, y, w, h) in faces:
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)

        # Save one face per frame, respecting cooldown
        if count < MAX_IMAGES and (now - last_save_time) >= CAPTURE_DELAY:
            count += 1
            last_save_time = now

            # Save BGR crop (color) — required for DeepFace accuracy
            face_crop = frame[y:y + h, x:x + w]
            save_path = os.path.join(dataset_path, f"{count}.jpg")
            cv2.imwrite(save_path, face_crop)

        # Only process first (largest) detected face per frame
        break

    # ---------------- HUD ----------------
    progress_text = f"Captured: {count}/{MAX_IMAGES}"
    bar_width     = int((count / MAX_IMAGES) * (FRAME_WIDTH - 40))

    # Progress bar background
    cv2.rectangle(frame, (20, FRAME_HEIGHT - 30), (FRAME_WIDTH - 20, FRAME_HEIGHT - 12), (60, 60, 60), -1)
    # Progress bar fill
    if bar_width > 0:
        cv2.rectangle(frame, (20, FRAME_HEIGHT - 30), (20 + bar_width, FRAME_HEIGHT - 12), (0, 200, 0), -1)

    cv2.putText(frame, progress_text, (20, FRAME_HEIGHT - 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, f"Student: {student_name}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2, cv2.LINE_AA)

    cv2.imshow("Face Capture", frame)

    # Exit on 'q' or when capture is complete
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Capture stopped by user.")
        break
    if count >= MAX_IMAGES:
        print(f"Done! {MAX_IMAGES} images saved for '{student_name}'.")
        break

# ---------------- CLEANUP ----------------
cam.release()
cv2.destroyAllWindows()