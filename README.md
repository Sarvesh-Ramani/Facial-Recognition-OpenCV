<div align="center">

# 👁 Facial Recognition — OpenCV

### Real-time face detection using Haar Cascade classifiers — streaming from webcam, drawing bounding boxes, zero ML training overhead.

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org)
[![Haar Cascade](https://img.shields.io/badge/Classifier-Haar_Cascade-orange?style=flat-square)]()
[![Webcam](https://img.shields.io/badge/Input-Live_Webcam-green?style=flat-square)]()

</div>

---

## ⚡ Hook

> Real-time face detection in under 25 lines of Python — using OpenCV's Haar Cascade classifier to draw bounding boxes over every face it finds in a live webcam stream, frame by frame.

---

## 🔭 The Problem

Face detection is the foundational layer beneath virtually every biometric security system, video analytics pipeline, and AR application. Before deep learning approaches became mainstream, Haar Cascade classifiers — introduced by Viola and Jones (2001) — were the state-of-the-art method: fast, lightweight, and deployable with zero GPU requirement.

This project demonstrates:
- How classical computer vision approaches work at the algorithmic level
- The full pipeline from raw webcam input → greyscale conversion → cascade detection → bounding box output
- Why scale factor and minimum neighbours tuning matters for precision vs. recall trade-offs in detection

---

## 🏗 How It Works

### Detection pipeline

```
Webcam frame (BGR)
    ↓  cv2.cvtColor → GRAY
Greyscale frame
    ↓  CascadeClassifier.detectMultiScale(scaleFactor=1.1, minNeighbours=2)
Bounding box coordinates [(x, y, w, h), ...]
    ↓  cv2.rectangle → overlay on original BGR frame
Live display with detection boxes
```

### Key components

- **`haarcascade_frontalface_default.xml`** — OpenCV's pre-trained frontal face Haar Cascade model. Contains 6,000+ features learned from thousands of positive/negative face image pairs
- **`detectMultiScale`** — slides a detection window across the image at multiple scales; `scaleFactor=1.1` controls pyramid step size, `minNeighbours=2` controls false positive suppression
- **Greyscale conversion** — Haar features operate on intensity, not colour. Converting BGR → GRAY reduces computation by 2/3 with no accuracy loss for this classifier
- **`cv2.VideoCapture(0)`** — reads frames from the default system webcam in real time
- **ESC key exit** — clean release of camera resource via `cap.release()`

---

## 🛠 Tech Stack

| Tool | Role |
|---|---|
| **Python** | Runtime and glue language |
| **OpenCV (`cv2`)** | Camera capture, image processing, cascade classification, display |
| **Haar Cascade XML** | Pre-trained frontal face detection model (bundled with OpenCV) |

### Why Haar Cascade over a CNN-based detector?

Haar Cascades are the right choice for understanding the foundations of face detection:
- **No training required** — the pre-trained XML model is included with OpenCV
- **No GPU required** — runs in real time on any CPU
- **Interpretable** — the cascade of classifiers is algorithmically transparent, unlike a black-box CNN
- **Fast** — the Viola-Jones algorithm rejects non-face regions early in the cascade, keeping per-frame latency low

For production face detection, a MTCNN or RetinaFace model would be more accurate — but this is about understanding the classical baseline first.

---

## 🏆 The Engineering Note — Tuning for Precision vs. Recall

The two parameters in `detectMultiScale` are not arbitrary:

- **`scaleFactor=1.1`** — how much the detection window shrinks per image pyramid level. Lower = more scales checked = more detections but slower. `1.1` is a standard starting point.
- **`minNeighbours=2`** — how many overlapping detections must agree before a region is classified as a face. Higher = fewer false positives but may miss faces. `2` keeps sensitivity high for a single-person webcam demo.

Increasing `minNeighbours` to 4–5 significantly reduces false positive boxes at the cost of missing smaller or partially occluded faces. This is the core precision/recall trade-off in classical detection.

---

## 🚀 Getting Started

**Step 1 — Clone and install**
```bash
git clone https://github.com/Sarvesh-Ramani/Facial-Recognition-OpenCV.git
cd Facial-Recognition-OpenCV
pip install opencv-python
```

**Step 2 — Run**
```bash
python detect_face.py
# A window opens showing your webcam feed with green bounding boxes over detected faces
# Press ESC to exit cleanly
```

**Step 3 — Test on a static image (optional)**
```python
# In detect_face.py, comment out VideoCapture and uncomment:
# img = cv2.imread('testimg.jpg')
# Then run: python detect_face.py
```

> Requires a connected webcam. Tested on Python 3.8+.

---

## 📁 Repository Structure

```
Facial-Recognition-OpenCV/
├── detect_face.py                        # Main detection script (25 lines)
├── haarcascade_frontalface_default.xml   # Pre-trained Haar Cascade model
├── testimg.jpg                           # Static test image
└── README.md
```

---

## 📄 Code Walkthrough

```python
import cv2

# Load the pre-trained frontal face Haar Cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open webcam (device index 0 = default camera)
cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()                              # Capture frame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # Convert to greyscale

    # Run cascade detector: scaleFactor controls image pyramid step,
    # minNeighbours controls false positive suppression
    faces = face_cascade.detectMultiScale(gray, 1.1, 2)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw bounding box

    cv2.imshow('img', img)
    if cv2.waitKey(30) & 0xff == 27:  # ESC key exits
        break

cap.release()
```

---

## 🔗 Links

- 📖 **Viola-Jones algorithm paper:** [Rapid Object Detection using a Boosted Cascade of Simple Features (2001)](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf)
- 📖 **OpenCV Haar Cascade docs:** [docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html)
- 👤 **Author:** [github.com/Sarvesh-Ramani](https://github.com/Sarvesh-Ramani)
- 💼 **LinkedIn:** [linkedin.com/in/sarvesh-ramani](https://linkedin.com/in/sarvesh-ramani)

---

<div align="center">
  <sub>Classical CV — because knowing how the foundation works makes you a better ML engineer.</sub>
</div>
