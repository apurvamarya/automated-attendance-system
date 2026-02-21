import cv2
import numpy as np
import os
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

dataset_path = "Project/automated-attendance-system/facial_recognisition_model/dataset"


def process_image(data):
    image_path, label = data

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    img = Image.open(image_path).convert('L')
    img_np = np.array(img, 'uint8')
    img_np = cv2.equalizeHist(img_np)

    detected_faces = face_detector.detectMultiScale(img_np)

    results = []
    for (x, y, w, h) in detected_faces:
        face = img_np[y:y+h, x:x+w]
        results.append((face, label))

    return results


def main():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # recognizer = cv2.face.LBPHFaceRecognizer_create(
    #     radius=2,
    #     neighbors=16,
    #     grid_x=8,
    #     grid_y=8
    # )

    image_data = []
    label_map = {}
    label_id = 0

    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            continue

        label_map[label_id] = folder

        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            image_data.append((image_path, label_id))

        label_id += 1

    print("Using", multiprocessing.cpu_count(), "CPU cores")

    faces = []
    face_labels = []

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        results = executor.map(process_image, image_data)

    for result in results:
        for face, label in result:
            faces.append(face)
            face_labels.append(label)

    print("Training model...")
    recognizer.train(faces, np.array(face_labels))

    recognizer.save(
        "Project/automated-attendance-system/facial_recognisition_model/trainer/trainer.yml"
    )

    with open(
        "Project/automated-attendance-system/facial_recognisition_model/trainer/labels.txt",
        "w"
    ) as f:
        for k, v in label_map.items():
            f.write(f"{k}:{v}\n")

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
