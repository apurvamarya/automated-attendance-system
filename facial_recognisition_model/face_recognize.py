import cv2
import numpy as np
import pandas as pd 
from datetime import datetime
import os
import requests

SERVER_URL = "https://attendance-system-9ptl.onrender.com/upload-attendance"
API_KEY = "attendance_upload_key"
 
#* Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Project/automated-attendance-system/facial_recognisition_model/trainer/trainer.yml")

#* Load labels
label_map = {}
with open("Project/automated-attendance-system/facial_recognisition_model/trainer/labels.txt", "r") as f:
    for line in f:
        key, value = line.strip().split(":")
        label_map[int(key)] = value

#* Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

#! cam = cv2.VideoCapture(1)
cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cam.set(3, 640)   # width
cam.set(4, 480)   # height



attendance_file = "Project/automated-attendance-system/dashboard/attendance/attendance.csv"
if not os.path.exists(attendance_file):
    with open(attendance_file, "w") as f:
        f.write("Name,Date,Time,Status\n")

marked_today = set()

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        confidence_percentage = max(0, min(100, 100 - confidence))
        if confidence < 70:
            name = label_map[id_]
        else:
            name = "Unknown"

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        display_text = f"{name} ({confidence_percentage:.1f}%)"
        cv2.putText(
            frame,
            display_text,
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

                #* Send attendance data to server

                payload = {
                    "Name": name,
                    "Date": date,
                    "Time": time,
                    "Status": "Present"
                }

                try:
                    requests.post(
                        SERVER_URL,
                        json=payload,
                        headers={"X-API-KEY": API_KEY},
                        timeout=5
                    )
                except Exception as e:
                    print("Upload failed:", e)


    cv2.imshow("Face Recognition Attendance", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    

cam.release()
cv2.destroyAllWindows()


