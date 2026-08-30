import numpy as np
from model_config import INPUT_SHAPE, NUM_CLASSES
from architectures import build_lstm_model, build_bilstm_model, build_cnn1d_model

print(
    f"[*] CẤU HÌNH HIỆN TẠI: INPUT_SHAPE = {INPUT_SHAPE}, NUM_CLASSES = {NUM_CLASSES}\n"
)

models = {
    "LSTM": build_lstm_model(),
    "BiLSTM": build_bilstm_model(),
    "1D-CNN": build_cnn1d_model(),
}

# Tạo 1 sample dummy data dựa trên config tĩnh để test forward pass
dummy_X = np.random.rand(1, INPUT_SHAPE[0], INPUT_SHAPE[1])

for name, model in models.items():
    print(f"=== {name} MODEL SUMMARY ===")
    model.summary()

    # Kiểm tra Forward Pass
    output = model.predict(dummy_X, verbose=0)
    print(f"-> {name} Dummy Input Shape: {dummy_X.shape}")
    print(f"-> {name} Dummy Output Shape: {output.shape}")

    # Kiểm chứng số lượng nơ-ron lớp cuối
    assert (
        output.shape[-1] == NUM_CLASSES
    ), f"LỖI: Output shape của {name} là {output.shape[-1]}, nhưng NUM_CLASSES là {NUM_CLASSES}!"
    print(f"-> [OK] {name} pass bài test kiến trúc.\n")
