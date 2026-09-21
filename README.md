# 🤖 AI Face Recognition & Identification System

<p align="center">

<strong>AI-powered Face Enrollment and Identification using Deep Face Embeddings</strong>

<br><br>

<img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/InsightFace-Face%20AI-purple?style=for-the-badge">
<img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-red?style=for-the-badge&logo=opencv&logoColor=white">
<img src="https://img.shields.io/badge/Streamlit-Web%20Application-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">
<img src="https://img.shields.io/badge/ONNX%20Runtime-CPU-green?style=for-the-badge">
<img src="https://img.shields.io/badge/NumPy-Numerical%20Computing-blue?style=for-the-badge&logo=numpy">
<img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge">

</p>

<p align="center">

<b>Face Detection → Face Embedding → Similarity Matching → Identity Decision</b>

</p>

---

# 📌 Overview

**AI Face Recognition & Identification System** is an end-to-end computer vision and artificial intelligence application developed using Python.

The system allows users to enroll individuals and identify new faces by comparing deep facial embeddings.

Instead of comparing raw images directly, the system converts each detected face into a numerical feature representation called a **face embedding**.

The embeddings are then compared using **cosine similarity**.

The system supports:

- 👤 Face enrollment
- 📁 Image upload
- 🎥 Webcam capture
- 🧠 Deep face embeddings
- 📐 512-dimensional face representation
- 💾 Multiple embeddings per person
- 🔍 Face recognition
- 📊 Cosine similarity matching
- 🟢 Known-person identification
- 🔴 Unknown-person rejection
- ⚠️ No-face handling
- 👥 Multiple-face handling
- 🎯 Configurable recognition threshold
- 🌐 Streamlit web interface
- 💻 Local CPU inference
- 💰 No paid API required

The application uses the **InsightFace `buffalo_l` model** for face analysis and generates facial embeddings that can be compared against enrolled identities.

---

# 🎯 Project Objective

The main objective of this project is to develop a complete face identification pipeline that can:

1. Detect a face from an image or webcam.
2. Generate a deep facial embedding.
3. Store embeddings during enrollment.
4. Compare new embeddings with stored embeddings.
5. Calculate similarity using cosine similarity.
6. Find the best matching identity.
7. Apply a recognition threshold.
8. Identify the person as **KNOWN** or **UNKNOWN**.

The complete pipeline is:

```text
                 Input Image / Webcam
                         │
                         ▼
                  Face Detection
                         │
                         ▼
                  Face Alignment
                         │
                         ▼
                  Face Embedding
                         │
                         ▼
                  512-D Vector
                         │
                         ▼
                Database Comparison
                         │
                         ▼
                 Cosine Similarity
                         │
                         ▼
                  Highest Similarity
                         │
                         ▼
                  Threshold Decision
                         │
                 ┌───────┴───────┐
                 ▼               ▼
              🟢 KNOWN        🔴 UNKNOWN
