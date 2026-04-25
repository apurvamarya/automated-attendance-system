import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
import requests
import threading  # ← async upload fix

SERVER_URL = "https://attendance-system-9ptl.onrender.com/upload-attendance"
API_KEY = "attendance_upload_key"

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Project/automated-attendance-system/facial_recognisition_model/trainer/trainer.yml")

label_map = {}
with open("Project/automated-attendance-system/facial_recognisition_model/trainer/labels.txt", "r") as f:
    for line in f:
        key, value = line.strip().split(":")
        label_map[int(key)] = value

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cam.set(3, 640)
cam.set(4, 480)

attendance_file = "Project/automated-attendance-system/dashboard/attendance/attendance.csv"
if not os.path.exists(attendance_file):
    with open(attendance_file, "w") as f:
        f.write("Name,Date,Time,Status\n")

marked_today = set()

def upload_async(payload):
    """Non-blocking upload — won't freeze the recognition loop"""
    try:
        requests.post(
            SERVER_URL,
            json=payload,
            headers={"X-API-KEY": API_KEY},
            timeout=5
        )
    except Exception as e:
        print("Upload failed:", e)

while True:
    ret, frame = cam.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # detectMultiScale already returns ALL faces — no change needed here
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        face_roi = cv2.resize(gray[y:y+h, x:x+w], (100, 100))  # ← normalize size
        id_, confidence = recognizer.predict(face_roi)
        confidence_pct = max(0, min(100, 100 - confidence))

        if confidence < 70 and id_ in label_map:
            name = label_map[id_]
            color = (0, 255, 0)
        else:
            name = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{name} ({confidence_pct:.1f}%)",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        if name != "Unknown":
            now = datetime.now()
            date, time_str = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")
            key = (name, date)

            if key not in marked_today:
                marked_today.add(key)
                df = pd.read_csv(attendance_file)
                df.loc[len(df)] = [name, date, time_str, "Present"]
                df.to_csv(attendance_file, index=False)
                print(f"✓ Marked: {name}")

                payload = {"Name": name, "Date": date, "Time": time_str, "Status": "Present"}
                threading.Thread(target=upload_async, args=(payload,), daemon=True).start()

    cv2.imshow("Face Recognition Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()