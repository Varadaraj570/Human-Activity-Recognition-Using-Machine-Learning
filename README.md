# Human Activity Recognition Using Machine Learning

## Overview

This project presents a real-time Human Activity Recognition (HAR) system
using Machine Learning, OpenCV, and MediaPipe Pose estimation. The system
captures live webcam video, detects human body landmarks, extracts pose
features, and predicts human activities such as:

- Standing
- Sitting
- Walking

The project demonstrates the integration of Computer Vision and Machine
Learning techniques for real-time activity prediction without using wearable
sensors.

---

## Features

- Real-time webcam activity detection
- MediaPipe Pose landmark extraction
- Random Forest Machine Learning classifier
- Pose skeleton visualization
- Real-time prediction display
- Output video recording support
- Low-cost and sensor-free activity recognition

---

## Technologies Used

- Python
- OpenCV
- MediaPipe
- Scikit-learn
- NumPy
- Pandas
- Joblib

---

## System Workflow

1. Capture live webcam video
2. Detect human pose using MediaPipe
3. Extract body landmark coordinates
4. Generate feature vectors
5. Train Random Forest classifier
6. Predict human activity
7. Display real-time output

---

## Dataset

The dataset is generated using live webcam input and MediaPipe Pose
estimation. The extracted x and y coordinates of 33 body landmarks are stored
in CSV format along with activity labels.

### Activities Used

- Standing
- Sitting
- Walking

---

## Project Structure

```text
Human_Activity_Recognition_Project/
│
├── collect_data.py
├── train_model.py
├── predict.py
├── dataset/
├── models/
├── results/
├── figure/
├── report.tex
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/Varadaraj570/Human-Activity-Recognition-Using-Machine-Learning.git
```

### Install Required Libraries

```bash
pip install opencv-python mediapipe scikit-learn pandas numpy joblib
```

---

## How to Run

### Step 1 — Collect Dataset

```bash
python collect_data.py
```

### Step 2 — Train Model

```bash
python train_model.py
```

### Step 3 — Run Real-Time Prediction

```bash
python predict.py
```

---

## Machine Learning Model

The project uses:

- Random Forest Classifier

for activity classification based on pose landmark coordinates.

---

## Applications

- Healthcare Monitoring
- Fitness Tracking
- Smart Surveillance
- Human Computer Interaction
- Smart Environment Systems

---

## Future Enhancements

- Add more human activities
- Improve prediction accuracy
- Deep Learning integration
- Mobile deployment
- Multi-person activity detection

---

## Authors

- Varadaraj — 4AL23CS177
- Shivakumara D B — 4AL23CS149

Department of Computer Science and Engineering  
Alva's Institute of Engineering and Technology

---

## License

This project is developed for academic and educational purposes.
