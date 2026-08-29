import os
import cv2
import numpy as np
import json
from pathlib import Path
from collections import deque

import mediapipe as mp
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh

def _unique_indices(connections):
    idx = set()
    for a, b in connections:
        idx.add(a); idx.add(b)
    return idx

_lips = _unique_indices(mp_face_mesh.FACEMESH_LIPS)
_left_eye = _unique_indices(mp_face_mesh.FACEMESH_LEFT_EYE)
_left_eyebrow = _unique_indices(mp_face_mesh.FACEMESH_LEFT_EYEBROW)
_right_eye = _unique_indices(mp_face_mesh.FACEMESH_RIGHT_EYE)
_right_eyebrow = _unique_indices(mp_face_mesh.FACEMESH_RIGHT_EYEBROW)

FACE_SUBSET_IDX = sorted(_lips | _left_eye | _left_eyebrow | _right_eye | _right_eyebrow)
N_FACE = len(FACE_SUBSET_IDX)

FACE_SCALE_I = FACE_SUBSET_IDX.index(33)
FACE_SCALE_J = FACE_SUBSET_IDX.index(263)

T = 30  # phải khớp đúng T dùng lúc training


def extract_frame_landmarks(image_rgb, holistic_model):
    results = holistic_model.process(image_rgb)

    if results.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark],
            dtype=np.float32,
        )
    else:
        pose = np.full((33, 4), np.nan, dtype=np.float32)

    if results.face_landmarks:
        all_face = results.face_landmarks.landmark
        face = np.array(
            [[all_face[i].x, all_face[i].y, all_face[i].z] for i in FACE_SUBSET_IDX],
            dtype=np.float32,
        )
    else:
        face = np.full((N_FACE, 3), np.nan, dtype=np.float32)

    if results.left_hand_landmarks:
        lh = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
            dtype=np.float32,
        )
    else:
        lh = np.full((21, 3), np.nan, dtype=np.float32)

    if results.right_hand_landmarks:
        rh = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
            dtype=np.float32,
        )
    else:
        rh = np.full((21, 3), np.nan, dtype=np.float32)

    return pose, face, lh, rh


# ============================================================
# THÊM MỚI: interpolate theo thời gian trên cả buffer/sequence
# (để đồng bộ pipeline train/infer)
# ============================================================
def fill_missing_and_resample_group(raw, T=30):
    """
    raw: array (num_frames_orig, K, D), có thể chứa NaN
    Trả về: array (T, K, D)
    - Cả nhóm NaN suốt buffer (vd tay absent 30 frame liên tục) -> zero-fill
    - NaN xen kẽ (occlusion tạm thời) -> interpolate theo trục thời gian
    - num_frames_orig == T (trường hợp real-time, buffer luôn đúng T frame)
      thì đây thực chất là resample T->T, nhưng vẫn dùng chung 1 hàm
      để nhất quán với pipeline training.
    """
    num_frames_orig = raw.shape[0]
    if np.all(np.isnan(raw)):
        return np.zeros((T,) + raw.shape[1:], dtype=np.float32)

    K, D = raw.shape[1], raw.shape[2]
    flat = raw.copy().reshape(num_frames_orig, K * D)

    for feat_idx in range(flat.shape[1]):
        col = flat[:, feat_idx]
        nan_mask = np.isnan(col)
        if nan_mask.any() and not nan_mask.all():
            valid_idx = np.where(~nan_mask)[0]
            col[nan_mask] = np.interp(np.where(nan_mask)[0], valid_idx, col[valid_idx])
        elif nan_mask.all():
            col[:] = 0.0
        flat[:, feat_idx] = col

    orig_idx = np.linspace(0, 1, num_frames_orig)
    target_idx = np.linspace(0, 1, T)
    resampled = np.zeros((T, K * D), dtype=np.float32)
    for feat_idx in range(flat.shape[1]):
        resampled[:, feat_idx] = np.interp(target_idx, orig_idx, flat[:, feat_idx])

    return resampled.reshape(T, K, D)


def normalize(raw, origin, scale_idx_pair, coord_dims=3):
    """
    raw: array (K, D) của 1 nhóm tại 1 frame
    """
    seq = raw.copy()
    if np.all(seq == 0):
        return seq

    coords = seq[:, :coord_dims]

    if origin == "centroid":
        origin_pt = coords.mean(axis=0, keepdims=True)
    else:
        origin_pt = coords[origin:origin + 1, :]

    centered = coords - origin_pt

    i, j = scale_idx_pair
    scale = np.linalg.norm(coords[j] - coords[i])
    scale = max(scale, 1e-6)

    seq[:, :coord_dims] = centered / scale
    return seq


def normalize_all(lh, rh):
    lh_n = normalize(lh, origin=0, scale_idx_pair=(0, 9), coord_dims=3)
    rh_n = normalize(rh, origin=0, scale_idx_pair=(0, 9), coord_dims=3)
    return lh_n, rh_n


def build_sequence_from_buffer(lh_buffer, rh_buffer):
    """
    lh_buffer, rh_buffer: list/deque of (21,3) array, độ dài T, có thể chứa NaN
    Trả về: (T, 126) sequence đã interpolate + normalize, sẵn sàng đưa vào model
    """
    lh_seq_raw = np.stack(list(lh_buffer))  # (T, 21, 3)
    rh_seq_raw = np.stack(list(rh_buffer))  # (T, 21, 3)

    # Interpolate theo thời gian (dùng cả buffer 30 frame)
    lh_filled = fill_missing_and_resample_group(lh_seq_raw, T=T)
    rh_filled = fill_missing_and_resample_group(rh_seq_raw, T=T)

    # Normalize từng frame
    lh_norm = np.stack([
        normalize(lh_filled[t], origin=0, scale_idx_pair=(0, 9), coord_dims=3)
        for t in range(T)
    ])
    rh_norm = np.stack([
        normalize(rh_filled[t], origin=0, scale_idx_pair=(0, 9), coord_dims=3)
        for t in range(T)
    ])

    sequence = np.concatenate([lh_norm.reshape(T, -1), rh_norm.reshape(T, -1)], axis=1)  # (T, 126)
    return sequence


# Tải label
RES_DIR = '../results/keypoints_output_holistic'
with open(f'{RES_DIR}/label_classes.json', 'r', encoding='utf-8') as f:
    class_names = json.load(f)
num_classes = len(class_names)
print(f"Đã tải {num_classes} nhãn thành công.")

import tensorflow as tf
model_path = "../models/cnn1d_model_hand.h5"
model = tf.keras.models.load_model(model_path)
print("Tải mô hình thành công!")

from PIL import Image, ImageDraw, ImageFont

COLORS = [
    (0, 255, 0),      # Green
    (0, 255, 255),    # Cyan 
    (255, 165, 0),    # Orange
    (255, 0, 255),    # Magenta
    (128, 128, 128)   # Gray
]


def prob_viz(res, class_names, input_frame, top_k=5):
    output_frame = input_frame.copy()
    top_indices = np.argsort(res)[-top_k:][::-1]

    output_pil = Image.fromarray(cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(output_pil)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)

    for rank, idx in enumerate(top_indices):
        prob = res[idx]
        label = class_names[idx]
        y = 70 + rank * 35

        color = COLORS[rank]

        draw.rectangle([(0, y), (int(prob * 200), y + 25)], fill=color)
        draw.text((205, y), f"{label}: {prob:.2f}", font=font, fill=(255, 255, 255))

    output_frame = cv2.cvtColor(np.array(output_pil), cv2.COLOR_RGB2BGR)
    return output_frame


font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 30)


def put_text_vietnamese(frame, text, position, font, color=(255, 255, 255)):
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)


# 1. New detection variables
lh_buffer = deque(maxlen=T)  # buffer RAW (có thể chứa NaN), chưa fill/normalize
rh_buffer = deque(maxlen=T)
sentence = []
predictions = []
threshold = 0.8

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue  # bỏ qua frame lỗi

        # đổi BGR -> RGB trước khi đưa vào MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pose_raw, face_raw, lh_raw, rh_raw = extract_frame_landmarks(frame_rgb, holistic)

        # Đưa vào buffer raw (còn NaN), không fill/normalize ngay từng frame 
        lh_buffer.append(lh_raw)
        rh_buffer.append(rh_raw)

        if len(lh_buffer) == T:
            # Đủ T frame -> interpolate theo thời gian trên cả buffer rồi normalize
            sequence = build_sequence_from_buffer(lh_buffer, rh_buffer)  # (T, 126)

            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            print(class_names[np.argmax(res)])
            predictions.append(np.argmax(res))

            if len(predictions) >= 10 and np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if class_names[np.argmax(res)] != sentence[-1]:
                            sentence.append(class_names[np.argmax(res)])
                    else:
                        sentence.append(class_names[np.argmax(res)])

            if len(sentence) > 5:
                sentence = sentence[-5:]

            frame = prob_viz(res, class_names, frame)

        cv2.rectangle(frame, (0, 0), (640, 40), (245, 117, 16), -1)
        frame = put_text_vietnamese(frame, ' '.join(sentence), (3, 3), font, (255, 255, 255))
        cv2.imshow('OpenCV Feed', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()