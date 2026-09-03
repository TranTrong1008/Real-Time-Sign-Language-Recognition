"""
sign_preprocessing.py - Module dùng chung cho việc ghi và đánh giá tập test real-time.

Chứa logic trích xuất/fill-missing/normalize cho cả 3 phiên bản (Hands / Pose+Hands / Holistic).
"""
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh

T = 30


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


def extract_frame_landmarks(image_rgb, holistic_model):
    """Trích xuất RAW landmark 1 frame — giữ NaN nếu không detect được.
    image_rgb PHẢI đã ở dạng RGB (không phải BGR)."""
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


def fill_missing_and_resample_group(raw, T=T):
    """raw: (num_frames_orig, K, D) có thể chứa NaN -> (T, K, D)."""
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
    """raw: (K, D) của 1 frame (đã qua interpolate)."""
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


NORMALIZE_PARAMS = {
    "pose": dict(origin=11, scale_idx_pair=(11, 12)),
    "face": dict(origin="centroid", scale_idx_pair=(FACE_SCALE_I, FACE_SCALE_J)),
    "left_hand": dict(origin=0, scale_idx_pair=(0, 9)),
    "right_hand": dict(origin=0, scale_idx_pair=(0, 9)),
}


def get_group_config(feature_dim):
    """Trả về [(tên_nhóm, K, D), ...] dựa theo feature_dim của từng phiên bản."""
    if feature_dim == 126:
        return [("left_hand", 21, 3), ("right_hand", 21, 3)]
    elif feature_dim == 258:
        return [("pose", 33, 4), ("left_hand", 21, 3), ("right_hand", 21, 3)]
    elif feature_dim == 534:
        remaining = feature_dim - 132 - 63 - 63
        assert remaining % 3 == 0, f"feature_dim={feature_dim} không khớp cấu trúc holistic"
        n_face = remaining // 3
        assert n_face == N_FACE, f"N_FACE hiện tại ({N_FACE}) khác với feature_dim={feature_dim} kỳ vọng"
        return [("pose", 33, 4), ("face", n_face, 3), ("left_hand", 21, 3), ("right_hand", 21, 3)]
    else:
        raise ValueError(f"feature_dim={feature_dim} không được hỗ trợ (chỉ 126/258/534).")


def build_feature_sequence(pose_seq_raw, face_seq_raw, lh_seq_raw, rh_seq_raw, feature_dim, T=T):
    """Dựng sequence (T, feature_dim) từ 4 nhóm RAW (mỗi nhóm (T,K,D), có thể NaN),
    tự chọn đúng nhóm cần dùng theo feature_dim (126/258/534) — cho phép suy ra cả
    3 phiên bản từ CÙNG 1 lần ghi Holistic đầy đủ."""
    group_config = get_group_config(feature_dim)
    raw_map = {"pose": pose_seq_raw, "face": face_seq_raw, "left_hand": lh_seq_raw, "right_hand": rh_seq_raw}

    filled = {name: fill_missing_and_resample_group(raw_map[name], T=T) for name, K, D in group_config}
    normed = {
        name: np.stack([normalize(filled[name][t], coord_dims=3, **NORMALIZE_PARAMS[name]) for t in range(T)])
        for name, K, D in group_config
    }
    seq = np.concatenate([normed[name].reshape(T, -1) for name, K, D in group_config], axis=1)
    seq = np.nan_to_num(seq, copy=False).astype(np.float32, copy=False)

    if seq.shape != (T, feature_dim):
        raise ValueError(f"Sequence có shape {seq.shape}, cần ({T}, {feature_dim}).")
    return seq
