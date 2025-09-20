# facerecong.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import json
from collections import defaultdict

class FaceRecognizer:
    def __init__(self, model_path="trainer/trainer.yml",
                 cascade_path="haarcascade_frontalface_default.xml",
                 face_size=(100, 100),
                 unknown_threshold=None,
                 labels_file="labels.json",
                 config_file="config.json"):

        if not hasattr(cv2, "face"):
            raise RuntimeError("cv2.face not found. Install opencv-contrib-python: pip install opencv-contrib-python")

        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=2, neighbors=8, grid_x=8, grid_y=8
        )
        self.face_size = face_size
        self.config_file = config_file

        # Load saved threshold if available
        self.unknown_threshold = self.load_threshold(unknown_threshold)

        self.frame_times, self.num_faces_detected, self.confidences = [], [], []
        self.start_time = time.time()
        self.load_model(model_path)
        self.labels = self.load_labels(labels_file)

        self.id_confidences = defaultdict(list)

    def load_threshold(self, default_value):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    cfg = json.load(f)
                if "threshold" in cfg:
                    print(f"[INFO] Loaded threshold={cfg['threshold']} from {self.config_file}")
                    return cfg["threshold"]
            except:
                pass
        return default_value  # fallback

    def save_threshold(self, value):
        with open(self.config_file, "w") as f:
            json.dump({"threshold": int(value)}, f, indent=4)
        print(f"[INFO] Saved threshold={int(value)} into {self.config_file}")

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            raise SystemExit(1)
        try:
            self.recognizer.read(model_path)
            print("[INFO] Model loaded from", model_path)
        except cv2.error as e:
            print("Error: Could not load trained model. Exception:", e)
            raise SystemExit(1)

    def load_labels(self, labels_file):
        if os.path.exists(labels_file):
            with open(labels_file, "r") as f:
                data = json.load(f)
            print("[INFO] Loaded labels:", data)
            return {int(k): v for k, v in data.items()}
        else:
            print(f"[WARN] Labels file not found: {labels_file}. Using IDs only.")
            return {}

    @staticmethod
    def _preprocess(face_img, size=(100, 100)):
        face_img = cv2.equalizeHist(face_img)
        face_img = cv2.GaussianBlur(face_img, (5, 5), 0)
        face_img = cv2.resize(face_img, size)
        return face_img

    def recognize_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=7, minSize=(50, 50)
        )
        detected_confidences = []

        for (x, y, w, h) in faces:
            pad_w, pad_h = int(w * 0.1), int(h * 0.1)
            x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
            x2, y2 = min(gray.shape[1], x + w + pad_w), min(gray.shape[0], y + h + pad_h)

            roi = gray[y1:y2, x1:x2]
            roi = self._preprocess(roi, size=self.face_size)

            try:
                label, confidence = self.recognizer.predict(roi)
                print(f"[DEBUG] Predicted ID={label}, confidence={confidence:.2f}")
                self.id_confidences[label].append(confidence)
            except cv2.error as e:
                print("Predict failed:", e)
                continue

            detected_confidences.append(confidence)

            if self.unknown_threshold is None:
                # calibration mode: show raw ID
                name = self.labels.get(label, f"ID-{label}")
            else:
                if confidence <= self.unknown_threshold:
                    name = self.labels.get(label, f"ID-{label}")
                else:
                    name = "Unknown"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({confidence:.1f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        self.frame_times.append(time.time() - self.start_time)
        self.num_faces_detected.append(len(faces))
        self.confidences.append(np.mean(detected_confidences) if detected_confidences else None)

        return frame

    def run(self):
        frame_count = 0
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frame = self.recognize_faces(frame)

            os.makedirs("processed_frames", exist_ok=True)
            cv2.imwrite(f"processed_frames/frame_{frame_count}.jpg", frame)
            frame_count += 1
            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.report_calibration()
        self.plot_results()

    def report_calibration(self):
        if not self.id_confidences:
            print("[INFO] No predictions recorded.")
            return

        print("\n[CALIBRATION REPORT]")
        all_values = []
        for label, confs in self.id_confidences.items():
            avg_conf = np.mean(confs)
            name = self.labels.get(label, f"ID-{label}")
            print(f"  {name}: avg confidence={avg_conf:.2f} (n={len(confs)})")
            all_values.extend(confs)

        if all_values:
            avg = np.mean(all_values)
            suggested = avg + 20
            print(f"\nSuggested threshold ≈ {suggested:.1f}")
            self.save_threshold(suggested)

    def plot_results(self):
        if len(self.frame_times) == 0:
            return
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.frame_times, self.num_faces_detected, marker="o", linestyle="-")
        plt.xlabel("Time (s)"); plt.ylabel("Faces Detected"); plt.title("Faces Detected Over Time"); plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(self.frame_times, self.confidences, marker="o", linestyle="-")
        plt.xlabel("Time (s)"); plt.ylabel("Confidence"); plt.title("Confidence Over Time"); plt.grid(True)

        plt.savefig("confidence_over_time.png")
        print("[INFO] Saved confidence_over_time.png")

if __name__ == "__main__":
    recognizer = FaceRecognizer(
        model_path="trainer/trainer.yml",
        cascade_path="haarcascade_frontalface_default.xml",
        unknown_threshold=None,  # auto-calibrate if no config.json
        labels_file="labels.json",
        config_file="config.json"
    )
    recognizer.run()
