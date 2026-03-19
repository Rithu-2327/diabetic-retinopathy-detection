# 🧠 Diabetic Retinopathy Detection System

A production-oriented deep learning project for automated detection of **Diabetic Retinopathy (DR)** from retinal fundus images. This system demonstrates a complete machine learning pipeline — from data processing and model development to real-time inference in a local environment.

---

## 🚀 Highlights


* Built an **end-to-end ML pipeline** for multi-class medical image classification
* Developed a **custom CNN architecture** using TensorFlow/Keras
* Achieved ~72% accuracy on 5-class DR classification
* Implemented a **CLI-based inference system** for real-time predictions
* Resolved **framework compatibility and model deserialization issues** in local deployment
* Designed with a clear path toward **scalable deployment and performance optimization**

---

## 🧬 Problem Statement

Diabetic Retinopathy is a progressive eye disease that can lead to blindness if not diagnosed early. Manual screening is time-intensive and requires domain expertise.

This project aims to:

* Automate DR stage detection
* Assist in early diagnosis
* Provide a scalable, ML-driven screening approach

---

## 🏗️ System Architecture

| Component     | Implementation                      |
| ------------- | ----------------------------------- |
| Input         | Retinal fundus images               |
| Preprocessing | Resizing (224×224), normalization   |
| Model         | Custom CNN (Conv → Pool → Dense)    |
| Output        | DR stage classification (5 classes) |
| Interface     | Command-line prediction system      |

---

## 📊 Classification Categories

| Index | Class          |
| ----- | -------------- |
| 0     | Mild           |
| 1     | Moderate       |
| 2     | No_DR          |
| 3     | Proliferate_DR |
| 4     | Severe         |

---


## ▶️ Getting Started

### 1. Clone the repository

```
git clone <your-repo-link>
cd Diabetic-retinopathy-detection/app
```

### 2. Install dependencies

```
pip install -r ../requirements.txt
```

### 3. Run inference

```
python predict.py ../sample_images/sample1.jpg
```

---

## 📈 Model Performance

| Metric           | Value     |
| ---------------- | --------- |
| Accuracy         | ~72%      |
| Classes          | 5         |
| Input Resolution | 224 × 224 |

The current model serves as a strong baseline for multi-class retinal image classification and supports consistent real-time inference.

---

## 🧠 Engineering Insights

* Designed and validated a **custom CNN pipeline from scratch**
* Handled **model loading incompatibilities across TensorFlow/Keras versions**
* Built a **robust inference script independent of training environment**
* Ensured reproducibility with structured project organization

---

## 🚀 Roadmap

* Integrating **transfer learning (ResNet / EfficientNet)** for higher accuracy
* Applying **data augmentation and class balancing strategies**
* Extending to a **web-based interface (Streamlit/Flask)**
* Preparing for **cloud-based deployment and API integration**

---

## 📌 Status

🚧 Actively under development — focused on improving model performance and deployment readiness.

---



## 📜 License

Licensed under the MIT License.
