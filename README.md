
# Face Recognition for Automated Attendance

A Python-based system for registering faces, training a recognition model, and automatically marking attendance using live video. Uses OpenCV’s face detection & recognition.

---

## 📂 Repository Contents

| File / Directory                      | Description                                                     |
| ------------------------------------- | --------------------------------------------------------------- |
| `facedataset.py`                      | Captures face images for each user; used to build a dataset.    |
| `facetrainer.py`                      | Trains a face recognizer model using the dataset.               |
| `facerecong.py`                       | Performs live recognition and marks attendance.                 |
| `haarcascade_frontalface_default.xml` | Haar Cascade model for detecting faces.                         |
| `labels.json`                         | Maps numeric IDs to user names.                                 |
| `config.json`                         | Configuration (e.g. threshold settings for recognition).        |
| `trainer/`                            | Stores the trained model (e.g. `.yml` file).                    |
| `confidence_over_time.png`            | Graph of confidence over time (used for threshold calibration). |
| `New Text Document.txt`               | (Probably placeholder or notes) — may remove or update.         |

---

## ⚙️ Dependencies

You will need:

* Python 3.x
* OpenCV (`opencv-python` and possibly `opencv-contrib-python`)
* Other Python libraries such as `numpy`, `os`, `json` etc.

---

## 🚀 Getting Started

Here’s how to use the project from scratch:

1. **Dataset creation**
   Run:

   ```bash
   python facedataset.py
   ```

   Follow prompts to enter user ID and name. The script will capture images and save them; `labels.json` will be updated.

2. **Training**
   After you have enough face images in your dataset:

   ```bash
   python facetrainer.py
   ```

   This will train the recognizer and save it under `trainer/`.

3. **Recognition & Attendance**
   Run:

   ```bash
   python facerecong.py
   ```

   The system uses the webcam to recognize faces in real time. If recognition confidence is good (below a threshold), presence is marked; otherwise the face is marked “Unknown”.

---

## ⚠️ Configuration & Thresholds

* The file `config.json` stores settings such as the confidence threshold (the maximum allowed “distance” or error for recognizing a face).
* Use the `confidence_over_time.png` graph to see how confidence scores vary. This helps in choosing a good threshold value.
* Adjust threshold in `config.json` as per your environment (camera quality, lighting etc.).

---

## 📋 Notes & Suggestions

* Ensure good lighting and face orientation when capturing dataset images — more diverse poses improves recognition accuracy.
* Make sure Haar Cascade XML file (`haarcascade_frontalface_default.xml`) is present and accessible.
* The dataset folder structure, names, and the mapping between IDs & person names in `labels.json` must be consistent.

---

## 🔧 Potential Improvements

* Add GUI for face registration and attendance marking.
* Use more advanced face detection / recognition (e.g. DNN-based or deep learning models) to improve accuracy.
* Store attendance logs (e.g. timestamp + user) to file or database.
* Improve error handling & user feedback.
* If many users, consider optimizing dataset/training or using incremental learning.

---
