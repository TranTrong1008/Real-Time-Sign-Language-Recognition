import numpy as np
import streamlit as st
import pandas as pd
import cv2

# Khởi tạo bộ nhớ tạm (Session State) 
if "list_img" not in st.session_state:
    st.session_state.list_img = []
if "status" not in st.session_state:
    st.session_state.status = "stop"
st.markdown( # "CSS" giao diện streamlit
    """
    <style>
    .block-container { 
        padding-top: 2rem; 
    }
    html, body, [data-testid="stMarkdownContainer"] {
        font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, Helvetica, Arial, sans-serif;
    }
    h1 {
        font-weight: 600 !important;
        margin-bottom: 0.5rem !important;
    }
    h2 {
        font-weight: 600 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
    .custom-subtitle {
        border-left: 3px solid #ccc;
        padding-left: 10px;
        color: #555555;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }
    p {
        line-height: 1.6 !important;
        color: #222222;
    }
    </style>
    """,
    unsafe_allow_html=True
)


tab1, tab2, tab3 = st.tabs(["# Overview", "# Demo", "# Dataset"])
with tab1:
    st.title("Real-Time Sign Language Recognition using MediaPipe Hands and BiLSTM for Hearing-Impaired Support")
    st.markdown('<div class="custom-subtitle">Python & Machine Learning Final Project – Summer 2026 HCMUT</div>', unsafe_allow_html=True)
    st.header("Project Overview")
    st.write(
        "This project develops a **real-time Vietnamese Sign Language Recognition System** using **MediaPipe Hands** and a "
        "Bidirectional LSTM (BiLSTM) model."
    )
    st.write(
        "The system captures hand movements through a webcam, extracts **21 hand landmarks** using MediaPipe Hands, "
        "converts them into sequential keypoints, and predicts Vietnamese sign language gestures using a trained BiLSTM model."
    )
    st.write(
        "The final application provides real-time predictions with confidence scores and serves as an assistive communication "
        "tool for people with hearing impairments."
    )
with tab2:
    col1, col2 = st.columns([0.4, 0.6])
    with col1:
        st.header("Model dự đoán :")
        IMAGE_LIST = st.empty()
        with IMAGE_LIST.container():
            for img in st.session_state.list_img[:5]:
                st.image(img, use_container_width=True)
    with col2:
        st.header("Webcam Live :")
        run = st.checkbox("Bật/Tắt Webcam")
        FRAME = st.empty()
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        text_status = st.empty()
        with btn_col1:
            if st.button("Start",type="primary",use_container_width=True):
                st.session_state.status = "start"
        with btn_col2:
            if st.button("Stop",type="primary",use_container_width=True):
                st.session_state.status = "stop"
        with btn_col3:
            if st.button("Reset",type="primary",use_container_width=True):
                st.session_state.status = "reset"
                st.session_state.list_img.clear()
        # Hiển thị trạng thái
        text_status.write(f"Trạng thái hiện tại: "f"**{st.session_state.status.upper()}**")
        if run:
            camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            ret, frame = camera.read()
            if not ret:
                st.error("Không thể kết nối tới Webcam.")
            else:
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                FRAME.image(frame,use_container_width=True)
                #RESET
                if st.session_state.status == "reset":
                    st.session_state.list_img.clear()
                    st.session_state.status = "stop"
                #START
                elif st.session_state.status == "start":
                    if len(st.session_state.list_img) < 5:
                        st.session_state.list_img.append(frame)
            camera.release()
        else:
            st.write("Webcam đang tắt.")
       
