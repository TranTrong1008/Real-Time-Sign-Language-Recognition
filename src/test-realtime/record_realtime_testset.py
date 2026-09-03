"""
record_realtime_testset.py - Ghi lại tập test real-time từ webcam.

Ghi 5 mẫu cho mỗi class bằng MediaPipe Holistic, lưu raw landmarks.

Cách chạy:
    python record_realtime_testset.py --labels_file ../../configs/labels.json --output_dir realtime_test_set --samples_per_class 5

Điều khiển trong lúc chạy:
    r : bắt đầu ghi 1 sample cho class hiện tại (có đếm ngược 3s)
    n : bỏ qua, chuyển sang class tiếp theo
    q : thoát toàn bộ chương trình
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sign_preprocessing import extract_frame_landmarks, mp_holistic, T


def get_vietnamese_font(size=28):
    """
    Tìm font TrueType hỗ trợ Unicode/tiếng Việt.
    """
    font_paths = [
        # Windows
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\tahoma.ttf",

        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",

        # macOS
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
    ]

    for font_path in font_paths:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size)

    raise FileNotFoundError(
        "Không tìm thấy font hỗ trợ tiếng Việt. "
        "Hãy cài Arial/DejaVu Sans hoặc cập nhật đường dẫn trong get_vietnamese_font()."
    )


FONT = get_vietnamese_font(28)


def draw_status(frame, text, color=(255, 255, 255), y=30):
    """
    Vẽ text Unicode/tiếng Việt lên frame OpenCV bằng Pillow.
    OpenCV cv2.putText() không hỗ trợ Unicode tốt, nên dùng PIL.ImageDraw.
    """

    # OpenCV: BGR -> PIL: RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    draw = ImageDraw.Draw(pil_image)

    x = 10

    # PIL dùng RGB, OpenCV dùng BGR
    text_color = (color[2], color[1], color[0])

    # Vẽ viền đen để chữ dễ nhìn trên webcam
    stroke_width = 4
    draw.text(
        (x, y),
        text,
        font=FONT,
        fill=text_color,
        stroke_width=stroke_width,
        stroke_fill=(0, 0, 0),
    )

    # PIL RGB -> OpenCV BGR
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return frame


def record_one_sample(cap, holistic_model, class_name, sample_idx):
    """Đếm ngược 3s rồi ghi liên tục đúng T frame."""

    countdown_start = time.time()

    while time.time() - countdown_start < 3.0:
        ret, frame = cap.read()
        if not ret:
            continue

        remaining = 3.0 - (time.time() - countdown_start)

        text = (
            f"[{class_name}] mẫu {sample_idx} - "
            f"Sẵn sàng: {remaining:.1f}s"
        )

        frame = draw_status(
            frame,
            text,
            (0, 165, 255),
        )

        cv2.imshow("Ghi tập test real-time", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return None

    pose_list, face_list, lh_list, rh_list = [], [], [], []

    while len(pose_list) < T:
        ret, frame = cap.read()

        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pose, face, lh, rh = extract_frame_landmarks(
            frame_rgb,
            holistic_model,
        )

        pose_list.append(pose)
        face_list.append(face)
        lh_list.append(lh)
        rh_list.append(rh)

        text = (
            f"[{class_name}] mẫu {sample_idx} - "
            f"ĐANG GHI {len(pose_list)}/{T}"
        )

        frame = draw_status(
            frame,
            text,
            (0, 0, 255),
        )

        cv2.imshow("Ghi tập test real-time", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return None

    return {
        "pose_raw": np.stack(pose_list),
        "face_raw": np.stack(face_list),
        "lh_raw": np.stack(lh_list),
        "rh_raw": np.stack(rh_list),
        "label": class_name,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Ghi tập test real-time cho đánh giá 3 phiên bản model"
    )

    parser.add_argument(
        "--labels_file",
        required=True,
        help="Đường dẫn labels.json (list tên class)",
    )

    parser.add_argument(
        "--output_dir",
        default="realtime_test_set",
        help="Thư mục lưu sample",
    )

    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=5,
        help="Số lần ghi mỗi class",
    )

    args = parser.parse_args()

    # Đọc labels với UTF-8
    with open(args.labels_file, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    print(f"Đã tải {len(class_names)} class.")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Không thể mở webcam.")
        return

    quit_all = False

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic_model:

        for class_name in class_names:

            if quit_all:
                break

            class_dir = output_root / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            existing = len(
                list(class_dir.glob("sample_*.npz"))
            )

            sample_idx = existing
            recorded_this_class = 0

            while recorded_this_class < args.samples_per_class:

                ret, frame = cap.read()

                if not ret:
                    continue

                text = (
                    f"Class: {class_name} "
                    f"({recorded_this_class}/{args.samples_per_class}) "
                    f"- [r] ghi  [n] bỏ qua  [q] thoát"
                )

                frame = draw_status(
                    frame,
                    text,
                    (0, 255, 0),
                )

                cv2.imshow(
                    "Ghi tập test real-time",
                    frame,
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    quit_all = True
                    break

                elif key == ord("n"):
                    break

                elif key == ord("r"):

                    result = record_one_sample(
                        cap,
                        holistic_model,
                        class_name,
                        sample_idx,
                    )

                    if result is None:
                        quit_all = True
                        break

                    out_path = (
                        class_dir
                        / f"sample_{sample_idx:03d}.npz"
                    )

                    np.savez_compressed(
                        out_path,
                        **result,
                    )

                    print(f"Đã lưu: {out_path}")

                    sample_idx += 1
                    recorded_this_class += 1

    cap.release()
    cv2.destroyAllWindows()

    print("Hoàn tất ghi tập test real-time.")


if __name__ == "__main__":
    main()
