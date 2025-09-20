# Copilot Instructions for Face_recog_v2

## Project Overview
This is a face recognition system using OpenCV and Python. The main workflow includes dataset creation, face training, and recognition. The project is organized for local experimentation and prototyping.

## Key Components
- `facedataset.py`: Script for collecting face images and building the dataset.
- `facetrainer.py`: Trains the face recognizer and saves the model to `trainer/trainer.yml`.
- `facerecong.py`: Runs real-time face recognition using the trained model.
- `haarcascade_frontalface_default.xml`: Haar Cascade classifier for face detection.
- `labels.json`: Stores label mappings for recognized faces.
- `config.json`: Configuration file for project settings.
- `dataset/`: Stores raw face images.
- `processed_frames/`: May store processed images (usage depends on scripts).
- `trainer/`: Contains the trained model file.

## Developer Workflows
- **Dataset Creation:** Run `facedataset.py` to collect images. Images are saved in `dataset/`.
- **Training:** Run `facetrainer.py` to train the model. Output is `trainer/trainer.yml`.
- **Recognition:** Run `facerecong.py` for live recognition. Uses webcam and displays results.
- **Configuration:** Adjust parameters in `config.json` as needed.

## Patterns & Conventions
- All scripts expect local file paths and use OpenCV for image processing.
- Label mappings are managed via `labels.json`.
- Model files are stored in `trainer/`.
- Face detection uses Haar Cascade (`haarcascade_frontalface_default.xml`).
- Scripts are intended to be run directly (no CLI framework).

## Integration Points
- OpenCV (`cv2`) is required for all scripts.
- No external API calls; all processing is local.
- Model and label files are shared between scripts for interoperability.

## Example Usage
```powershell
# Collect dataset
python facedataset.py

# Train model
python facetrainer.py

# Run recognition
python facerecong.py
```

## Tips for AI Agents
- When adding new features, follow the pattern of storing models in `trainer/` and datasets in `dataset/`.
- Update `labels.json` if new classes/labels are introduced.
- Reference `config.json` for adjustable parameters.
- Use OpenCV for all image operations to maintain consistency.

---
If any section is unclear or missing details, please provide feedback for further refinement.