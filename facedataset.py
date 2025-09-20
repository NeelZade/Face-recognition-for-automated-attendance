# facedataset.py
import cv2
import os
import json

class FaceRegistration:
    def __init__(self,
                 dataset_path="dataset",
                 face_cascade_path="haarcascade_frontalface_default.xml",
                 target_count=100,
                 face_size=(100, 100),
                 labels_file="labels.json"):
        self.dataset_path = dataset_path
        self.face_cascade_path = face_cascade_path
        self.target_count = target_count
        self.face_size = face_size
        self.labels_file = labels_file

        self.cam = cv2.VideoCapture(0)
        self.face_detector = cv2.CascadeClassifier(self.face_cascade_path)
        if not self.face_detector.load(self.face_cascade_path):
            pass
        self._create_dataset_directory()

    @staticmethod
    def _preprocess_face(face_img):
        equalized = cv2.equalizeHist(face_img)
        blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
        norm_img = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
        return norm_img

    def _create_dataset_directory(self):
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)

    def update_labels(self, face_id, name):
        # Load existing labels
        if os.path.exists(self.labels_file):
            with open(self.labels_file, "r") as f:
                labels = json.load(f)
        else:
            labels = {}

        # Add/update entry
        labels[str(face_id)] = name

        with open(self.labels_file, "w") as f:
            json.dump(labels, f, indent=4)

        print(f"[INFO] labels.json updated: {labels}")

    def capture_faces(self, face_id, name):
        print("\n[INFO] Initializing face capture. Look at the camera ...")
        count = 0

        while True:
            ret, img = self.cam.read()
            if not ret:
                continue
            img = cv2.flip(img, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            cv2.putText(img, "Give different angle face", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            for (x, y, w, h) in faces:
                face_gray = gray[y:y+h, x:x+w]
                processed_face = self._preprocess_face(face_gray)
                processed_face = cv2.resize(processed_face, self.face_size)

                filename = os.path.join(self.dataset_path, f"User.{face_id}.{count}.processed.jpg")
                cv2.imwrite(filename, processed_face)
                count += 1

                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(img, f"Captures: {count}/{self.target_count}",
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            cv2.imshow("Face Registration", img)
            k = cv2.waitKey(100) & 0xFF
            if k == 27 or count >= self.target_count:
                break

        print(f"\n[INFO] {count} face images captured. Exiting Program")
        self.cleanup()
        # Update labels file after capture
        self.update_labels(face_id, name)

    def cleanup(self):
        self.cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    face_id = input("\nEnter numeric user ID: ").strip()
    try:
        face_id = int(face_id)
    except:
        print("Please enter a numeric ID (e.g., 1).")
        exit(1)

    name = input("Enter name for this user: ").strip()
    if not name:
        print("Name cannot be empty.")
        exit(1)

    face_reg = FaceRegistration()
    face_reg.capture_faces(face_id, name)
