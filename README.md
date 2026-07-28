# Real-Time Sign Language Recognition using MediaPipe Hands and BiLSTM for Hearing-Impaired Support  

> **Python & Machine Learning Final Project – Summer 2026 HCMUT**




#  Project Overview

This project develops a **real-time Vietnamese Sign Language Recognition System** using **MediaPipe Hands** and a **Bidirectional LSTM (BiLSTM)** model.

The system captures hand movements through a webcam, extracts **21 hand landmarks** using MediaPipe Hands, converts them into sequential keypoints, and predicts Vietnamese sign language gestures using a trained BiLSTM model.

The final application provides real-time predictions with confidence scores and serves as an assistive communication tool for people with hearing impairments.



#  Objectives

- Recognize Vietnamese Sign Language in real time.
- Build an end-to-end Machine Learning pipeline.
- Apply Computer Vision with MediaPipe Hands.
- Learn temporal sequence modeling using BiLSTM.
- Deploy an interactive demo using Streamlit.



# 👥 Team Information

**Team 17 – Team SAY2**

| Member | Role |
|----------|------|
| Nguyễn Quốc Trọng | Project Lead |
| Nguyễn Phương Thảo | Data Engineer |
| Phan Võ Bảo Trâm | Backend Developer |
| Phạm Ngọc Thịnh | Frontend / Demo |



#  Project Structure
đang update.....




#  Dataset

Vietnamese Sign Language Hand Gesture Dataset

Source:

> Kaggle

Current size:

- **54 classes**
- **38 images per class**
- Image format: JPG

mở rộng tập dataset(ví dụ góc quay , độ sáng video ,....) : đang update..... 



#  System Pipeline

```text
                Webcam
                    │
                    ▼
                MediaPipe Hands
                (21 Hand Landmarks)
                    │
                    ▼
                Keypoint Sequence
                    │
                    ▼
                BiLSTM Model
                    │
                    ▼
                Prediction
                (Label + Confidence)
                    │
                    ▼
                Streamlit Interface
```



#  Input / Output

### Input

- Webcam video stream
- Continuous hand gesture sequence

### Output

- Predicted Vietnamese Sign Language label
- Confidence score

Example:





#  Machine Learning Model

## Feature Extraction

- MediaPipe Hands
- 21 landmarks per hand
- x, y, z coordinates

## Sequence Model

Bidirectional Long Short-Term Memory (BiLSTM)

Architecture

```
                    Input Sequence

                         ↓

                       BiLSTM

                         ↓

                       Dropout

                         ↓

                       Dense

                         ↓

                      Softmax
```



#  Technology Stack

| Component | Technology |
|------------|------------|
| Programming Language | Python |
| Computer Vision | OpenCV |
| Landmark Detection | MediaPipe Hands |
| Deep Learning | TensorFlow / Keras |
| API | FastAPI |
| Frontend | Streamlit |
| Deployment | Docker |
| CI/CD | GitHub Actions |



#  Evaluation Metrics

Model performance will be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Inference Latency (ms/frame)



#  API
đang update ......

#  Demo

The Streamlit application allows users to:

- Capture webcam video
- Detect hand landmarks in real time
- Display predicted sign language
- Show confidence score
- Visualize MediaPipe hand skeleton



#  Installation

Clone project

```bash
git clone https://github.com/TranTrong1008/Real-Time-Sign-Language-Recognition.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

update thêm ....



#  Requirements

```text
Python >= 3.11

tensorflow
opencv-python
mediapipe
numpy
pandas
scikit-learn
fastapi
streamlit
matplotlib

```




#  References

- MediaPipe Hands
- TensorFlow / Keras Documentation
- OpenCV Documentation
- Kaggle Dataset
- Scikit-learn Documentation



#  License

Dự án này được phát triển cho mục đích giáo dục là một phần của project cuối khóa **Python & Machine Learning – Summer 2026** .
