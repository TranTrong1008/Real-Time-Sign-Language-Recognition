from __future__ import annotations
import json
import threading
from collections import deque
from functools import lru_cache
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
FEATURE_DIM = 126
CONFIDENCE_THRESHOLD = 0.6
STABLE_FRAMES = 10
MAX_SENTENCE_WORDS = 5


PROCESS_WIDTH = 480     # Frame will be resize to this width before going to Holistic, to reduce computation cost
HOLISTIC_MODEL_COMPLEXITY = 0   # may reduce accuracy of Holistic a bit, but much quicker


UNICODE_FONT_CANDIDATES = (
    Path("C:/Windows/Fonts/segoeui.ttf"),
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
)

MODEL_REGISTRY = {
    "LSTM": ("lstm_model_hand.h5",),
    "BiLSTM": ("bilstm_model_hand.h5",),
    "1D-CNN": ("cnn1d_model_hand.h5",),
    "Transformer": ("transformer_hands_best.keras",),
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


@lru_cache(maxsize=8)
def load_unicode_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a cross-platform font that can render Vietnamese characters.
    Cache to make sure that font is loaded once only for each frame.
    """
    for font_path in UNICODE_FONT_CANDIDATES:
        if font_path.is_file():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def draw_frame_overlay(rgb_image: np.ndarray, sentence: str, status: str) -> np.ndarray:
    """Draw Unicode text on an already-RGB frame and return a BGR frame for WebRTC.
    Input: rgb_image (converted in recv())
    """
    canvas = Image.fromarray(rgb_image)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, rgb_image.shape[1], 72), fill=(28, 28, 28))
    draw.text((12, 5), sentence, font=load_unicode_font(22), fill=(255, 255, 255))
    draw.text((12, 43), status, font=load_unicode_font(16), fill=(80, 220, 130))
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def _safe_scale(value: float) -> float:
    return value if np.isfinite(value) and value > 1e-6 else 1.0


def _normalize_xyz(points: np.ndarray, origin: np.ndarray, scale: float) -> np.ndarray:
    normalized = points.copy()
    normalized[:, :3] = (normalized[:, :3] - origin[:3]) / _safe_scale(scale)
    return normalized


def extract_holistic_keypoints(results: Any) -> tuple[np.ndarray, bool, bool]:
    """Convert one MediaPipe result into a normalized 126-value feature vector."""
    left_hand = np.zeros((21, 3), dtype=np.float32)
    right_hand = np.zeros((21, 3), dtype=np.float32)
    left_detected = results.left_hand_landmarks is not None
    right_detected = results.right_hand_landmarks is not None

    for destination, detected in (
        (left_hand, results.left_hand_landmarks),
        (right_hand, results.right_hand_landmarks),
    ):
        if detected:
            values = np.asarray([[p.x, p.y, p.z] for p in detected.landmark], dtype=np.float32)
            hand_scale = float(np.linalg.norm(values[0] - values[9]))
            destination[:] = _normalize_xyz(values, values[0], hand_scale)

    vector = np.concatenate((left_hand.ravel(), right_hand.ravel()))
    if vector.shape != (FEATURE_DIM,):
        raise ValueError(f"Feature vector có shape {vector.shape}, cần ({FEATURE_DIM},).")
    vector = np.nan_to_num(vector, copy=False).astype(np.float32, copy=False)
    return vector, left_detected, right_detected


def interpolate_missing_frames(
    window: np.ndarray, left_valid: np.ndarray, right_valid: np.ndarray
) -> np.ndarray:
    """Linearly interpolate frames where a hand wasn't detected in the sliding window."""
    window = window.copy()
    frame_indices = np.arange(window.shape[0])
    for valid_mask, col_slice in ((left_valid, slice(0, 63)), (right_valid, slice(63, 126))):
        valid_idx = frame_indices[valid_mask]
        if valid_idx.size == 0 or valid_idx.size == window.shape[0]:
            continue
        block = window[:, col_slice]
        for col in range(block.shape[1]):
            block[:, col] = np.interp(frame_indices, valid_idx, block[valid_idx, col])
        window[:, col_slice] = block
    return window


class SignLanguageProcessor(VideoProcessorBase):
    """Thread-safe MediaPipe, inference, smoothing, and frame annotation pipeline."""

    def __init__(self, model: Any, labels: list[str] | None, model_name: str) -> None:
        self.model = model
        self.labels = labels
        self.model_name = model_name
        self.sequence: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)
        self.left_valid: deque[bool] = deque(maxlen=SEQUENCE_LENGTH)
        self.right_valid: deque[bool] = deque(maxlen=SEQUENCE_LENGTH)
        self.predictions: deque[int] = deque(maxlen=STABLE_FRAMES)
        self.sentence: deque[str] = deque(maxlen=MAX_SENTENCE_WORDS)
        self.current_label = ""
        self.confidence = 0.0
        self.frame_count = 0
        self.last_error = ""
        self.enabled = True
        self.lock = threading.Lock()
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=HOLISTIC_MODEL_COMPLEXITY,     # model_complexity=0 to reduce Holistic cost per-frame
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        # Warm-up when initializing, so that the first prediction 
        # won't add more build/trace cost from TensorFlow
        try:
            warmup_batch = np.zeros((1, SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
            self.model(warmup_batch, training=False)
        except Exception:
            pass

    def reset(self) -> None:
        with self.lock:
            self.sequence.clear()
            self.left_valid.clear()
            self.right_valid.clear()
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

    def _predict(self, keypoints: np.ndarray, left_detected: bool, right_detected: bool) -> None:
        self.sequence.append(keypoints)
        self.left_valid.append(left_detected)
        self.right_valid.append(right_detected)
        if len(self.sequence) < SEQUENCE_LENGTH:
            return
        window = interpolate_missing_frames(
            np.asarray(self.sequence, dtype=np.float32),
            np.asarray(self.left_valid, dtype=bool),
            np.asarray(self.right_valid, dtype=bool),
        )
        batch = np.expand_dims(window, axis=0)
        # Cal model directly to reduce overhead of model.predict()
        probabilities = np.asarray(self.model(batch, training=False))[0]
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

        # Downscale before processing in order to reduce the cost of Holistic + 
        # overlay drawing + encoding frame again, without affecting feature vector.
        height, width = image.shape[:2]
        if width > PROCESS_WIDTH:   # resize if frame is too big
            scale = PROCESS_WIDTH / width
            image = cv2.resize(image, (PROCESS_WIDTH, int(height * scale)), interpolation=cv2.INTER_AREA)

        with self.lock:
            self.frame_count += 1
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.enabled:
                try:
                    results = self.holistic.process(rgb)
                    keypoints, left_detected, right_detected = extract_holistic_keypoints(results)
                    self._predict(keypoints, left_detected, right_detected)
                    self.last_error = ""
                except Exception as exc:  # A bad frame must not terminate the WebRTC worker.
                    self.last_error = str(exc)

            sentence = " ".join(self.sentence) or "Đang chờ cử chỉ..."
            # Directly pass rgb_frame, avoid converting BGR->RGB again.
            image = draw_frame_overlay(
                rgb,
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
        "Webcam → Holistic, chỉ lấy Hands (126 đặc trưng) → chuỗi 30 frame → mô hình → "
        "lọc confidence/độ ổn định → ghép từ thành câu"
    )
    st.subheader("Các mô hình được so sánh")
    st.write("LSTM · BiLSTM · 1D-CNN · Transformer")
    st.write(
        "Mỗi mô hình được đánh giá trên cùng tập test bằng Accuracy, Macro Precision, "
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
                st.write("Đã tìm thấy đủ 4 model.")

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
            media_stream_constraints={
                "video": {"width": {"ideal": 640}, "frameRate": {"ideal": 24, "max": 30}},
                "audio": False,
            },
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
            # Increase interval 500ms -> 800ms to reduce Streamlit rerun frequency
            st_autorefresh(
                interval=800,
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
