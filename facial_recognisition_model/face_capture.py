import cv2 
import os

cam = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier( 
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

student_id = input("Enter Student ID: ")
student_name = input("Enter Student Name: ")

dataset_path = f"Project/automated-attendance-system/facial_recognisition_model/dataset/{student_name}"
os.makedirs(dataset_path, exist_ok=True)

count = 0

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        cv2.imwrite(
            f"{dataset_path}/{count}.jpg",
            gray[y:y+h, x:x+w]
        )
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

    cv2.imshow("Face Capture", img)

    if cv2.waitKey(50) & 0xFF == ord('q'):
        break
    elif count >= 100:
        break

cam.release()
cv2.destroyAllWindows()
