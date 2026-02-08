import cv2
import numpy as np
import pandas as pd 
from datetime import datetime
import os

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Project/automated-attendance-system/facial_recognisition_model/trainer/trainer.yml")

# Load labels
label_map = {}
with open("Project/automated-attendance-system/facial_recognisition_model/trainer/labels.txt", "r") as f:
    for line in f:
        key, value = line.strip().split(":")
        label_map[int(key)] = value

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cam = cv2.VideoCapture(0)

attendance_file = "Project/automated-attendance-system/dashboard/attendance/attendance.csv"
if not os.path.exists(attendance_file):
    with open(attendance_file, "w") as f:
        f.write("Name,Date,Time,Status\n")

marked_today = set()

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if confidence < 70:
            name = label_map[id_]
        else:
            name = "Unknown"

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(
            frame,
            f"{name}",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255,255,255),
            2
        )

        if name != "Unknown":
            now = datetime.now()
            date = now.strftime("%Y-%m-%d")
            time = now.strftime("%H:%M:%S")

            key = (name, date)
            if key not in marked_today:
                marked_today.add(key)

                df = pd.read_csv(attendance_file)
                df.loc[len(df)] = [name, date, time, "Present"]
                df.to_csv(attendance_file, index=False)

                print(f"Attendance marked for {name}")

    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()