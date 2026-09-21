# 🤖 AI Face Recognition & Identification System

<p align="center">

<strong>AI-powered face enrollment and identification using deep face embeddings</strong>

<br><br>

<img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/InsightFace-Face%20AI-purple?style=for-the-badge">
<img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-red?style=for-the-badge&logo=opencv&logoColor=white">
<img src="https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit">
<img src="https://img.shields.io/badge/ONNX%20Runtime-CPU-green?style=for-the-badge">

</p>

<p align="center">
<b>Detect → Embed → Compare → Identify</b>
</p>

---

## 📌 Overview

An end-to-end **AI face recognition and identification system** built with Python, InsightFace, OpenCV, and Streamlit.

The system:

- 👤 Enrolls people using images or webcam
- 🧠 Generates **512-dimensional face embeddings**
- 💾 Stores multiple embeddings per person
- 🔍 Recognizes new faces
- 📊 Uses **cosine similarity** for matching
- 🟢 Identifies known people
- 🔴 Rejects unknown people using a threshold
- ⚠️ Handles no-face and multiple-face inputs
- 💻 Runs locally using CPU inference
- 💰 Requires no paid API

---

## 🎯 How It Works

```text
Image / Webcam
      ↓
Face Detection
      ↓
Face Embedding
      ↓
512-D Vector
      ↓
Compare With Database
      ↓
Cosine Similarity
      ↓
Best Match
      ↓
Threshold = 0.30
      ↓
KNOWN / UNKNOWN
```

---

## 🧠 Model & Technology

| Technology | Purpose |
|---|---|
| Python | Application development |
| InsightFace `buffalo_l` | Face detection & embeddings |
| OpenCV | Image processing |
| NumPy | Vector calculations |
| ONNX Runtime | CPU inference |
| Streamlit | Web interface |
| scikit-learn | Evaluation |

### Face Embedding

Each detected face is converted into a 512-dimensional vector:

```text
Face Image
     ↓
InsightFace
     ↓
[0.12, -0.43, 0.87, ...]
```

The system compares these vectors instead of comparing raw images.

---

## 📊 Recognition Logic

The system uses cosine similarity:

```text
                 A · B
Similarity = ─────────────
              ||A|| × ||B||
```

Example:

```text
modi  → 0.21
virat → 0.73
```

Best match:

```text
virat → 0.73
```

With threshold `0.30`:

```text
0.73 >= 0.30
      ↓
🟢 KNOWN
```

If:

```text
Best Similarity = 0.12

0.12 < 0.30
      ↓
🔴 UNKNOWN
```

---

## 👤 Enrollment

Multiple samples can be stored for the same person.

```text
modi
 ├── Embedding 1
 ├── Embedding 2
 ├── Embedding 3
 └── Embedding 4
```

Multiple samples help handle variations in:

- Pose
- Lighting
- Expression
- Camera angle

---

## 📈 Evaluation

The development dataset contained:

- **12 genuine comparisons**
- **40 impostor comparisons**

### Similarity Results

| Type | Min | Max | Average |
|---|---:|---:|---:|
| Genuine | 0.3523 | 0.7517 | 0.6147 |
| Impostor | -0.0594 | 0.1788 | 0.0422 |

### Threshold Testing

```text
Threshold 0.20 → 100%
Threshold 0.25 → 100%
Threshold 0.30 → 100%
Threshold 0.35 → 100%
```

The selected development threshold is **0.30**.

> ⚠️ The observed 100% result applies only to the small local development dataset and is not a production accuracy claim.

---

## ⚠️ Failure Handling

The system handles:

```text
No Face
   ↓
❌ No face detected

Multiple Faces
   ↓
⚠️ Multiple faces detected

Low Similarity
   ↓
🔴 UNKNOWN
```

Other difficult conditions include poor lighting, low resolution, extreme face angles, and occlusion.

---

## 🖥️ Application

The Streamlit interface contains two main functions:

```text
👤 ENROLL
   ├── Upload Image
   └── Webcam

🔍 RECOGNITION
   ├── Upload Image
   └── Webcam
```

### Screenshots

Add your screenshots here:

```text
screenshots/
├── enrollment.png
├── recognition-known.png
└── recognition-unknown.png
```

## 🏠 Home Page

<img width="631" height="828" alt="Home Page" src="https://github.com/user-attachments/assets/a30fed0a-9c2f-4948-9394-2a366ade9414" />

## 👤 Enrollment Page

<img width="1024" height="1536" alt="Enrollment Page" src="https://github.com/user-attachments/assets/eeb532eb-c381-49f4-8632-f29739b4d9d4" />

## 🔍 Recognition Page

<img width="1920" height="1080" alt="Recognition Page" src="https://github.com/user-attachments/assets/3b8fdc5d-ad90-4d85-9056-3eaaad54055a" />




## 📁 Project Structure

```text
ai-face-recognition-system/
│
├── app.py
├── enroll.py
├── recognize.py
├── test_face.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── evaluation/
│   └── evaluate.py
│
├── dataset/
│   └── README.md
│
├── face_database/
│


---

## ⚙️ Installation

### 1. Clone

```bash
git clone https://github.com/YOUR_USERNAME/ai-face-recognition-system.git
cd ai-face-recognition-system
```

### 2. Create Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

## 🔐 Privacy

Face images and biometric embeddings are **not included in the public repository**.

The following are excluded:

```text
venv/
__pycache__/
dataset/*.jpg
dataset/*.png
face_database/*.pkl
```

This project performs inference locally and does not require a paid face-recognition API.

---

## 🚧 Limitations

Current limitations:

- Small evaluation dataset
- Local embedding storage
- No liveness detection
- No anti-spoofing
- CPU inference
- No production authentication

---

## 🚀 Future Improvements

- 🎥 Real-time webcam recognition
- 🛡️ Liveness detection
- 🔐 Encrypted biometric storage
- 🗄️ Secure database
- ⚡ Vector database/search
- 🎯 Advanced threshold calibration
- 📊 Larger evaluation dataset
- 📈 ROC / FAR / FRR analysis
- 🚀 GPU inference

---



## ⭐ What This Project Demonstrates

```text
AI / ML
   +
Computer Vision
   +
Python
   +
Deep Face Embeddings
   +
Similarity Matching
   +
Model Evaluation
   +
Streamlit
   +
Git & GitHub
```

---

## 👨‍💻 Author

**Prasad**  
Computer Science & Engineering Student

**Interests:** AI • Machine Learning • Python • Computer Vision • Cloud • DevOps

---

<p align="center">

<b>🚀 Detect. Embed. Compare. Identify.</b>

<br><br>

Python • InsightFace • OpenCV • NumPy • ONNX Runtime • Streamlit

</p>
