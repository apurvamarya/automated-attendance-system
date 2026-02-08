import cv2
import numpy as np
import os 
from PIL import Image

dataset_path = "Project/automated-attendance-system/facial_recognisition_model/dataset"
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def get_images_and_labels(path):
    image_paths = []
    labels = []
    label_id = 0
    label_map = {}

    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if not os.path.isdir(folder_path):
            continue

        label_map[label_id] = folder

        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            image_paths.append(image_path)
            labels.append(label_id)

        label_id += 1

    faces = []
    face_labels = []

    for i, image_path in enumerate(image_paths):
        img = Image.open(image_path).convert('L')
        img_np = np.array(img, 'uint8')

        detected_faces = face_detector.detectMultiScale(img_np)
        for (x, y, w, h) in detected_faces:
            faces.append(img_np[y:y+h, x:x+w])
            face_labels.append(labels[i])

    return faces, face_labels, label_map


faces, face_labels, label_map = get_images_and_labels(dataset_path)

print("Training model...")
recognizer.train(faces, np.array(face_labels))
recognizer.save("Project/automated-attendance-system/facial_recognisition_model/trainer/trainer.yml")

# Save label mapping
with open("Project/automated-attendance-system/facial_recognisition_model/trainer/labels.txt", "w") as f:
    for k, v in label_map.items():
        f.write(f"{k}:{v}\n")

print("Training completed successfully!")
