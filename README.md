#  Real-Time Vietnamese Sign Language Recognition

> **Python & Machine Learning Final Project – Summer 2026 | HCMUT**  
> **Team 17 – SAY2**

---

##  Overview

A real-time Vietnamese Sign Language Recognition system using **MediaPipe Holistic** for landmark extraction and deep learning models for temporal sequence classification.

The system captures webcam frames, extracts landmarks from the **left and right hands**, builds a 30-frame sliding window, and predicts one of **30 Vietnamese sign classes** with confidence-based temporal smoothing.

### Main objectives

- Real-time Vietnamese sign language recognition from webcam.
- Extract hand landmarks using **MediaPipe Holistic**.
- Compare **LSTM, BiLSTM, 1D-CNN, Transformer**.
- Evaluate models using standard classification metrics.
- Integrate the trained models into a **Streamlit Web App**.

---

##  Pipeline

```text
Webcam
   │
   ▼
MediaPipe Holistic
   │
   ├── Left Hand
   └── Right Hand
   │
   ▼
126 features / frame
   │
   ▼
30-frame Sliding Window
   │
   ▼
Sequence Model
   │
   ├── LSTM
   ├── BiLSTM
   ├── 1D-CNN
   └── Transformer
   │
   ▼
Confidence Threshold
+ Temporal Smoothing
   │
   ▼
Current Sign
   │
   ▼
Accumulated Sentence
   │
   ▼
Streamlit Web App
```

---

##  Dataset

**Source:** [`star092304/ViSignLanguage-Video`](https://huggingface.co/datasets/star092304/ViSignLanguage-Video)

| Item | Value |
|---|---:|
| Original videos | ~3,800 |
| Original classes | 100 |
| Selected classes | **30** |
| Frames / video | **30** |
| Random seed | **42** |

The project selects 30 sign classes with sufficient sample quantity and quality.

### Data preprocessing

Each video is:

1. Resampled to 30 frames.
2. Processed using MediaPipe Holistic.
3. Converted into left/right hand landmark sequences.
4. Missing hand landmarks are handled during preprocessing.
5. Stored as numerical sequences for model training.

---

##  Input Representation

The system uses **MediaPipe Holistic** with up to two hands per frame.

Each detected hand contains 21 landmarks:

```text
21 landmarks × (x, y, z)
= 63 features / hand
```

For two hands:

| Hand | Representation | Dimensions |
|---|---|---:|
| Left Hand | 21 × `(x, y, z)` | 63 |
| Right Hand | 21 × `(x, y, z)` | 63 |
| **Total** | | **126** |

Therefore, each frame is represented by:

```text
126 features
```

A sequence contains 30 consecutive frames:

```text
(30, 126)
```

The model input during inference is:

```text
(batch_size, 30, 126)
```

The model output is:

```text
(batch_size, 30)
```

where 30 represents the number of Vietnamese sign classes.

---

##  Models

The project compares four temporal models:

| Model | Architecture |
|---|---|
| LSTM | Long Short-Term Memory |
| BiLSTM | Bidirectional LSTM |
| 1D-CNN | Temporal 1D Convolution |
| Transformer | Transformer Encoder |

All models use the same:

```text
Input:  (30, 126)
Output: (30,)
```

and are evaluated using the same data split and test protocol.

---

##  Evaluation

The models are evaluated using:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Confusion Matrix


---

##  Web Application

The Streamlit application provides:

- Real-time webcam input via WebRTC.
- MediaPipe Holistic detection.
- Left/right hand landmark visualization.
- Model selection.
- Real-time sign prediction.
- Confidence score.
- Temporal smoothing.
- Duplicate prediction filtering.
- Sentence accumulation.
- Sliding-window reset.
- Sentence reset.

---

##  Project Structure

```text
Real-Time-Sign-Language-Recognition/
│
├── configs/
│   └── labels.json
│
├── data/
│
├── models/
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── preprocessing.ipynb
│   └── training.ipynb
│
├── results/
│   ├── metrics/
│   └── confusion_matrices/
│
├── src/
│   ├── app.py
│   ├── logic_processor.py
│   ├── data_preprocessing.py
│   ├── train_lstm.py
│   ├── train_bilstm.py
│   ├── train_cnn1d.py
│   ├── train_transformer.py
│   └── models/
│
├── requirements.txt
└── README.md
```

---

##  Installation

### 1. Clone repository

```bash
git clone https://github.com/TranTrong1008/Real-Time-Sign-Language-Recognition.git

cd Real-Time-Sign-Language-Recognition
```

### 2. Create environment

Python **3.11** is recommended.

```bash
conda create -n sign_lang_env python=3.11 -y

conda activate sign_lang_env
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt
```

---

##  Model Setup

Download the trained models from the project storage and place them inside:

```text
models/
├── lstm_model_hand.h5
├── bilstm_model_hand.h5
├── cnn1d_model_hand.h5
└── transformer_hands_best.keras
```

The model class order must match:

```text
configs/labels.json
```

---

##  Run the Web App

```bash
streamlit run src/app.py
```

Then open:

```text
http://localhost:8501
```

Select a model and start the webcam.

---

## 👥 Team

| Member | Role |
|---|---|
| **M1 – Nguyễn Quốc Trọng** | LSTM, BiLSTM, 1D-CNN & Model Evaluation |
| **M2 – Nguyễn Phương Thảo** | Data Engineering & Preprocessing |
| **M3 – Phan Võ Bảo Trâm** | Transformer & Temporal Logic |
| **M4 – Phạm Ngọc Thịnh** | Streamlit, WebRTC & Model Integration |

---


##  License

This project is developed for educational purposes as part of the **Python & Machine Learning – Summer 2026 HCMUT** final project.

Dataset and external resources remain subject to their respective licenses.