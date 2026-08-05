import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Cấu hình tham số (Thay đổi giá trị này khi có dữ liệu thật)
NUM_CLASSES = 100  # CẦN XÁC NHẬN: Số lượng từ vựng thực tế trong ViSignLanguage-Video
INPUT_SHAPE = (30, 126)


def build_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """Khởi tạo kiến trúc mô hình LSTM"""
    model = Sequential()

    # Các lớp LSTM (Yêu cầu return_sequences=True cho 2 lớp đầu)
    model.add(
        LSTM(64, return_sequences=True, activation="relu", input_shape=input_shape)
    )
    model.add(LSTM(128, return_sequences=True, activation="relu"))
    model.add(LSTM(64, return_sequences=False, activation="relu"))

    # Các lớp Dense (Fully Connected)
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))

    # LỚP OUTPUT BẮT BUỘC: units = num_classes, activation = 'softmax'
    model.add(Dense(num_classes, activation="softmax"))

    return model


if __name__ == "__main__":
    # ĐƯỜNG DẪN TỚI FILE DUMMY DATA
    # Sử dụng đường dẫn tương đối từ thư mục gốc của project
    dummy_path = os.path.join("data", "dummy_data.npy")

    # Tự động tạo file dummy_data.npy nếu M2 chưa kịp cung cấp
    if not os.path.exists(dummy_path):
        print(f"[*] Không tìm thấy {dummy_path}. Đang tự động tạo dummy data...")
        os.makedirs("data", exist_ok=True)
        # Tạo 5 video giả (samples), mỗi video 30 frames, mỗi frame 126 tọa độ
        dummy_data = np.random.rand(5, INPUT_SHAPE[0], INPUT_SHAPE[1])
        np.save(dummy_path, dummy_data)

    # 1. Nạp dữ liệu
    X_test = np.load(dummy_path)
    print(f"[*] Đã nạp dummy_data.npy thành công. Input Shape: {X_test.shape}")

    # 2. Xây dựng mô hình
    model = build_model()
    print("\n[*] BẢNG TÓM TẮT KIẾN TRÚC MÔ HÌNH:")
    model.summary()

    # 3. Chạy kiểm thử Forward Pass
    print("\n[*] Đang chạy thử Forward Pass với dummy data...")
    predictions = model.predict(X_test)
    print(f"[*] Output Shape của mô hình: {predictions.shape}")

    # 4. Đối chiếu kết quả
    expected_output_shape = (X_test.shape[0], NUM_CLASSES)
    if predictions.shape == expected_output_shape:
        print(f"\nPASSED: Output shape chính xác {expected_output_shape}.")
    else:
        print(
            f"\nFAILED: Output shape {predictions.shape} không khớp với {expected_output_shape}."
        )
