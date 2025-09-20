# facetrainer.py
import cv2
import numpy as np
from PIL import Image
import os
import json

class FaceTrainer:
    def __init__(self,
                 dataset_path="dataset",
                 model_path="trainer/trainer.yml",
                 face_cascade_path="haarcascade_frontalface_default.xml",
                 face_size=(100, 100),
                 labels_file="labels.json"):
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.face_cascade_path = face_cascade_path
        self.face_size = face_size
        self.labels_file = labels_file

        if not hasattr(cv2, "face"):
            raise RuntimeError("cv2.face not found. Install opencv-contrib-python: pip install opencv-contrib-python")

        self.lbph_recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=8, grid_x=8, grid_y=8)
        self.detector = cv2.CascadeClassifier(self.face_cascade_path)
        self._create_model_directory()

    def _create_model_directory(self):
        os.makedirs(os.path.dirname(self.model_path) or "trainer", exist_ok=True)

    def get_images_and_labels(self):
        files = [f for f in os.listdir(self.dataset_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        processed = [f for f in files if ".processed." in f]
        use_list = processed if processed else files

        face_samples, ids = [], []
        print(f"Found {len(files)} images, using {len(use_list)} for training.")

        for fname in use_list:
            image_path = os.path.join(self.dataset_path, fname)
            try:
                parts = fname.split(".")
                if len(parts) < 3:
                    continue
                id = int(parts[1])
            except Exception as e:
                print(f"Skipping {fname}: {e}")
                continue

            pil_img = Image.open(image_path).convert("L")
            img_numpy = np.array(pil_img, dtype="uint8")

            faces = self.detector.detectMultiScale(img_numpy, scaleFactor=1.1, minNeighbors=5)
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    face = img_numpy[y:y+h, x:x+w]
                    face = cv2.resize(face, self.face_size)
                    face_samples.append(face)
                    ids.append(id)
            else:
                face = cv2.resize(img_numpy, self.face_size)
                face_samples.append(face)
                ids.append(id)

        return face_samples, ids

    def train_faces(self):
        print("\n[INFO] Training faces...")
        faces, ids = self.get_images_and_labels()
        if len(faces) == 0:
            print("No faces found in dataset.")
            return

        np_ids = np.array(ids, dtype=np.int32)
        self.lbph_recognizer.train(faces, np_ids)
        self.lbph_recognizer.write(self.model_path)
        print(f"[INFO] {len(ids)} faces trained across {len(np.unique(ids))} people. Model saved to {self.model_path}")

        # Update labels.json with IDs and names
        self.update_labels(np.unique(ids))

    def update_labels(self, unique_ids):
        # Load existing labels
        if os.path.exists(self.labels_file):
            with open(self.labels_file, "r") as f:
                labels = json.load(f)
        else:
            labels = {}

        updated = False
        for uid in unique_ids:
            if str(uid) not in labels:
                name = input(f"Enter name for ID {uid}: ").strip()
                labels[str(uid)] = name
                updated = True

        if updated:
            with open(self.labels_file, "w") as f:
                json.dump(labels, f, indent=4)
            print(f"[INFO] labels.json updated: {labels}")
        else:
            print("[INFO] No new IDs, labels.json unchanged.")

if __name__ == "__main__":
    trainer = FaceTrainer()
    trainer.train_faces()
