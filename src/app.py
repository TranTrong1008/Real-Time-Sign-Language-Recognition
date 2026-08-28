"""thinh_Streamlit web application for real-time Vietnamese sign recognition."""
from __future__ import annotations
import json
import threading
from collections import deque
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from streamlit_autorefresh import st_autorefresh

try:
    import av
    import mediapipe as mp
    from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer
    WEBRTC_AVAILABLE = True
    WEBRTC_IMPORT_ERROR = ""
except ImportError as exc:  # Keep the Overview page usable before dependencies are installed.
    av = None
    mp = None
    VideoProcessorBase = object
    WEBRTC_AVAILABLE = False
    WEBRTC_IMPORT_ERROR = str(exc)


BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
SEQUENCE_LENGTH = 30
FEATURE_DIM = 534
CONFIDENCE_THRESHOLD = 0.8
STABLE_FRAMES = 10
MAX_SENTENCE_WORDS = 5

UNICODE_FONT_CANDIDATES = (
    Path("C:/Windows/Fonts/segoeui.ttf"),
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
)

MODEL_REGISTRY = {
    "LSTM": ("lstm_best.keras", "lstm_model.keras", "lstm_best.h5", "lstm_model.h5"),
    "BiLSTM": ("bilstm_best.keras", "bilstm_model.keras", "bilstm_best.h5", "bilstm_model.h5"),
    "1D-CNN": (
        "cnn1d_best.keras",
        "cnn1d_model.keras",
        "1d_cnn_best.keras",
        "cnn1d_best.h5",
        "cnn1d_model.h5",
        "1d_cnn_best.h5",
    ),
    "Transformer": ("transformer_best.keras", "transformer_best.h5"),
    "ST-GCN": ("stgcn_best.keras", "st_gcn_best.keras", "stgcn_best.h5", "st_gcn_best.h5"),
}

LABEL_FILES = (
    BASE_DIR / "configs" / "labels.json",
    BASE_DIR / "configs" / "labels.txt",
    BASE_DIR / "configs" / "classes.txt",
    MODELS_DIR / "labels.json",
    MODELS_DIR / "label_classes.npy",
    BASE_DIR / "data" / "label_classes.npy",
)

def discover_models() -> tuple[dict[str, Path], list[str]]:
    """Return the first existing artifact for each model and readable missing-model messages."""
    available: dict[str, Path] = {}
    missing: list[str] = []
    for display_name, filenames in MODEL_REGISTRY.items():
        path = next((MODELS_DIR / name for name in filenames if (MODELS_DIR / name).is_file()), None)
        if path is None:
            missing.append(f"{display_name}: chưa tìm thấy {', '.join(filenames)}")
        else:
            available[display_name] = path
    return available, missing

def load_labels() -> tuple[list[str] | None, str | None]:
    """Load ordered class names without allowing pickled NumPy objects."""
    for path in LABEL_FILES:
        if not path.is_file():
            continue
        try:
            if path.suffix == ".json":
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    raw = raw.get("labels", raw.get("classes"))
                labels = [str(value) for value in raw]
            elif path.suffix == ".npy":
                labels = [str(value) for value in np.load(path, allow_pickle=False).tolist()]
            else:
                labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if labels:
                return labels, str(path.relative_to(BASE_DIR))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return None, None


@st.cache_resource(show_spinner="Đang tải mô hình...")
def load_model(model_path: str) -> Any:
    """Load a Keras model once per process."""
    import tensorflow as tf

    return tf.keras.models.load_model(model_path, compile=False)


def validate_model(model: Any, labels: list[str] | None) -> tuple[bool, str]:
    """Reject artifacts that cannot consume the shared Holistic representation."""
    input_shape = getattr(model, "input_shape", None)
    output_shape = getattr(model, "output_shape", None)
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
    if not input_shape or len(input_shape) != 3 or tuple(input_shape[-2:]) != (SEQUENCE_LENGTH, FEATURE_DIM):
        return False, f"Input model {input_shape}, cần (None, {SEQUENCE_LENGTH}, {FEATURE_DIM})."
    if labels and output_shape and output_shape[-1] != len(labels):
        return False, f"Model có {output_shape[-1]} lớp nhưng metadata có {len(labels)} nhãn."
    return True, ""


def _connection_indices(connections: Any) -> list[int]:
    return sorted({index for edge in connections for index in edge})


def load_unicode_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a cross-platform font that can render Vietnamese characters."""
    for font_path in UNICODE_FONT_CANDIDATES:
        if font_path.is_file():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def draw_frame_overlay(image: np.ndarray, sentence: str, status: str) -> np.ndarray:
    """Draw Unicode text on an OpenCV BGR frame using Pillow."""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(rgb_image)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, image.shape[1], 72), fill=(28, 28, 28))
    draw.text((12, 5), sentence, font=load_unicode_font(22), fill=(255, 255, 255))
    draw.text((12, 43), status, font=load_unicode_font(16), fill=(80, 220, 130))
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def face_subset_indices() -> list[int]:
    """Build the same 92-point eyes/eyebrows/lips subset used by M2."""
    face_mesh = mp.solutions.face_mesh
    connections = (
        face_mesh.FACEMESH_LIPS
        | face_mesh.FACEMESH_LEFT_EYE
        | face_mesh.FACEMESH_LEFT_EYEBROW
        | face_mesh.FACEMESH_RIGHT_EYE
        | face_mesh.FACEMESH_RIGHT_EYEBROW
    )
    indices = _connection_indices(connections)
    if len(indices) != 92:
        raise RuntimeError(f"Face subset có {len(indices)} điểm, cần đúng 92 điểm.")
    return indices


def _safe_scale(value: float) -> float:
    return value if np.isfinite(value) and value > 1e-6 else 1.0


def _normalize_xyz(points: np.ndarray, origin: np.ndarray, scale: float) -> np.ndarray:
    normalized = points.copy()
    normalized[:, :3] = (normalized[:, :3] - origin[:3]) / _safe_scale(scale)
    return normalized


def extract_holistic_keypoints(results: Any, face_indices: list[int]) -> np.ndarray:
    """Convert one MediaPipe result into a normalized 534-value feature vector."""
    pose = np.zeros((33, 4), dtype=np.float32)
    face = np.zeros((92, 3), dtype=np.float32)
    left_hand = np.zeros((21, 3), dtype=np.float32)
    right_hand = np.zeros((21, 3), dtype=np.float32)

    if results.pose_landmarks:
        pose = np.asarray(
            [[p.x, p.y, p.z, p.visibility] for p in results.pose_landmarks.landmark], dtype=np.float32
        )
        shoulder_midpoint = (pose[11, :3] + pose[12, :3]) / 2.0
        shoulder_width = float(np.linalg.norm(pose[11, :3] - pose[12, :3]))
        pose = _normalize_xyz(pose, shoulder_midpoint, shoulder_width)

    if results.face_landmarks:
        landmarks = results.face_landmarks.landmark
        face = np.asarray([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in face_indices], dtype=np.float32)
        index_map = {original: subset for subset, original in enumerate(face_indices)}
        eye_width = float(np.linalg.norm(face[index_map[33]] - face[index_map[263]]))
        face = _normalize_xyz(face, face.mean(axis=0), eye_width)

    for destination, detected in (
        (left_hand, results.left_hand_landmarks),
        (right_hand, results.right_hand_landmarks),
    ):
        if detected:
            values = np.asarray([[p.x, p.y, p.z] for p in detected.landmark], dtype=np.float32)
            hand_scale = float(np.linalg.norm(values[0] - values[9]))
            destination[:] = _normalize_xyz(values, values[0], hand_scale)

    vector = np.concatenate((pose.ravel(), face.ravel(), left_hand.ravel(), right_hand.ravel()))
    if vector.shape != (FEATURE_DIM,):
        raise ValueError(f"Feature vector có shape {vector.shape}, cần ({FEATURE_DIM},).")
    return np.nan_to_num(vector, copy=False).astype(np.float32, copy=False)


class SignLanguageProcessor(VideoProcessorBase):
    """Thread-safe MediaPipe, inference, smoothing, and frame annotation pipeline."""

    def __init__(self, model: Any, labels: list[str] | None, model_name: str) -> None:
        self.model = model
        self.labels = labels
        self.model_name = model_name
        self.sequence: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)
        self.predictions: deque[int] = deque(maxlen=STABLE_FRAMES)
        self.sentence: deque[str] = deque(maxlen=MAX_SENTENCE_WORDS)
        self.current_label = ""
        self.confidence = 0.0
        self.frame_count = 0
        self.last_error = ""
        self.enabled = True
        self.lock = threading.Lock()
        self.face_indices = face_subset_indices()
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def reset(self) -> None:
        with self.lock:
            self.sequence.clear()
            self.predictions.clear()
            self.sentence.clear()
            self.current_label = ""
            self.confidence = 0.0
            self.last_error = ""

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "label": self.current_label,
                "confidence": self.confidence,
                "sentence": list(self.sentence),
                "frame_count": self.frame_count,
                "buffer_size": len(self.sequence),
                "error": self.last_error,
            }

    def _predict(self, keypoints: np.ndarray) -> None:
        self.sequence.append(keypoints)
        if len(self.sequence) < SEQUENCE_LENGTH:
            return
        batch = np.expand_dims(np.asarray(self.sequence, dtype=np.float32), axis=0)
        probabilities = np.asarray(self.model.predict(batch, verbose=0))[0]
        class_index = int(np.argmax(probabilities))
        self.confidence = float(probabilities[class_index])
        self.predictions.append(class_index)
        is_stable = len(self.predictions) == STABLE_FRAMES and len(set(self.predictions)) == 1
        if not is_stable or self.confidence < CONFIDENCE_THRESHOLD:
            self.current_label = ""
            return
        label = self.labels[class_index] if self.labels and class_index < len(self.labels) else f"Class_{class_index}"
        self.current_label = label
        if not self.sentence or self.sentence[-1] != label:
            self.sentence.append(label)

    def recv(self, frame: Any) -> Any:
        image = frame.to_ndarray(format="bgr24")
        with self.lock:
            self.frame_count += 1
            if self.enabled:
                try:
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = self.holistic.process(rgb)
                    self._predict(extract_holistic_keypoints(results, self.face_indices))
                    mp.solutions.drawing_utils.draw_landmarks(
                        image, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS
                    )
                    mp.solutions.drawing_utils.draw_landmarks(
                        image, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS
                    )
                    mp.solutions.drawing_utils.draw_landmarks(
                        image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS
                    )
                    self.last_error = ""
                except Exception as exc:  # A bad frame must not terminate the WebRTC worker.
                    self.last_error = str(exc)

            sentence = " ".join(self.sentence) or "Đang chờ cử chỉ..."
            image = draw_frame_overlay(
                image,
                sentence,
                f"{self.model_name} | confidence: {self.confidence:.1%} | frame: {self.frame_count}",
            )
        return av.VideoFrame.from_ndarray(image, format="bgr24")

    def __del__(self) -> None:
        holistic = getattr(self, "holistic", None)
        if holistic is not None:
            holistic.close()


def initialize_state() -> None:
    defaults = {
        "recognition_enabled": True,
        "reset_counter": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_overview() -> None:
    st.title("Nhận diện ngôn ngữ ký hiệu Việt Nam thời gian thực")
    st.caption("Python & Machine Learning Final Project – Summer 2026 HCMUT")
    st.subheader("Tổng quan dự án")
    st.write(
        "Ứng dụng nhận luồng webcam, trích xuất các điểm đặc trưng bằng **MediaPipe Holistic** "
        "và nhận diện từ bằng mô hình học sâu có kết quả tốt nhất."
    )
    st.info(
        "Webcam → Holistic (534 đặc trưng) → chuỗi 30 frame → mô hình → "
        "lọc confidence/độ ổn định → ghép từ thành câu"
    )
    st.subheader("Các mô hình được so sánh")
    st.write("LSTM · BiLSTM · 1D-CNN · Transformer · ST-GCN")
    st.write(
        "Mỗi mô hình được đánh giá trên cùng signer-based test split bằng Accuracy, Macro Precision, "
        "Macro Recall, Macro F1-score và Confusion Matrix."
    )


def render_demo() -> None:
    available, missing = discover_models()
    labels, label_source = load_labels()

    if not WEBRTC_AVAILABLE:
        st.error("Chưa thể mở webcam vì thiếu dependency: " + WEBRTC_IMPORT_ERROR)
        st.code("pip install -r requirements.txt", language="bash")
        return
    if not available:
        st.error("Chưa tìm thấy model nào trong thư mục models/. Giao diện vẫn hoạt động nhưng chưa thể suy luận.")
        with st.expander("Tên file model được hỗ trợ"):
            st.write("\n\n".join(missing))
        return

    left, right = st.columns([0.38, 0.62], gap="large")
    with left:
        st.subheader("Kết quả nhận diện")
        model_name = st.selectbox("Chọn mô hình", list(available), key="selected_model")
        st.caption(f"File: models/{available[model_name].name}")
        if labels:
            st.success(f"Đã nạp {len(labels)} nhãn từ {label_source}.")
        else:
            st.warning("Thiếu metadata nhãn; kết quả tạm hiển thị Class_<index>.")

        enabled = st.toggle("Bật nhận diện", value=st.session_state.recognition_enabled)
        st.session_state.recognition_enabled = enabled
        if st.button("Reset câu và bộ đệm", use_container_width=True):
            st.session_state.reset_counter += 1

        with st.expander("Model chưa khả dụng"):
            if missing:
                for message in missing:
                    st.write(f"- {message}")
            else:
                st.write("Đã tìm thấy đủ 5 model.")

    try:
        model = load_model(str(available[model_name]))
        valid, validation_error = validate_model(model, labels)
    except Exception as exc:
        valid, validation_error, model = False, f"Không thể load model: {exc}", None

    with right:
        st.subheader("Webcam Live")
        if not valid:
            st.error(validation_error)
            return

        processor_factory = lambda: SignLanguageProcessor(model, labels, model_name)
        context = webrtc_streamer(
            key=f"sign-language-{model_name}",
            video_processor_factory=processor_factory,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    processor = context.video_processor
    if processor is not None:
        processor.enabled = enabled
        reset_token = getattr(processor, "reset_token", -1)
        if reset_token != st.session_state.reset_counter:
            processor.reset()
            processor.reset_token = st.session_state.reset_counter
        snapshot = processor.snapshot()
        with left:
            st.metric("Từ hiện tại", snapshot["label"] or "—")
            st.progress(snapshot["confidence"], text=f"Confidence: {snapshot['confidence']:.1%}")
            st.text_area("Câu tích lũy", " ".join(snapshot["sentence"]), disabled=True)
            st.caption(
                f"Frame: {snapshot['frame_count']} · Buffer: {snapshot['buffer_size']}/{SEQUENCE_LENGTH}"
            )
            if snapshot["error"]:
                st.warning("Frame gần nhất gặp lỗi: " + snapshot["error"])
        if context.state.playing:
            st_autorefresh(
                interval=500,
                key=f"recognition-status-refresh-{model_name}",
            )
    else:
        with left:
            st.info("Nhấn START trong khung webcam để bắt đầu nhận diện.")


def main() -> None:
    st.set_page_config(page_title="Vietnamese Sign Language Recognition", page_icon="🤟", layout="wide")
    initialize_state()
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
        [data-testid="stMetricValue"] {font-size: 1.8rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    overview_tab, demo_tab = st.tabs(["Overview", "Demo"])
    with overview_tab:
        render_overview()
    with demo_tab:
        render_demo()


if __name__ == "__main__":
    main()
