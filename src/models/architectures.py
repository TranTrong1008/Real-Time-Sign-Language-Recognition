from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Bidirectional,
    Conv1D,
    GlobalAveragePooling1D,
    Dense,
    Dropout,
    BatchNormalization,
)
from model_config import INPUT_SHAPE, NUM_CLASSES


def build_lstm_model():
    model = Sequential(
        [
            # Bỏ activation="relu" để dùng mặc định tanh, thêm Dropout giữa các lớp
            LSTM(64, return_sequences=True, input_shape=INPUT_SHAPE),
            Dropout(0.2),
            LSTM(128, return_sequences=False),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dropout(0.5),  # Regularization mạnh ở lớp Dense cuối
            Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def build_bilstm_model():
    model = Sequential(
        [
            Bidirectional(
                LSTM(64, return_sequences=True),
                input_shape=INPUT_SHAPE,
            ),
            Dropout(0.2),
            Bidirectional(LSTM(128, return_sequences=False)),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dropout(0.5),
            Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def build_cnn1d_model():
    model = Sequential(
        [
            Conv1D(
                filters=64, kernel_size=3, activation="relu", input_shape=INPUT_SHAPE
            ),
            BatchNormalization(),  # Giúp CNN ổn định đặc trưng
            Conv1D(filters=128, kernel_size=3, activation="relu"),
            BatchNormalization(),
            GlobalAveragePooling1D(),
            Dense(64, activation="relu"),
            Dropout(0.5),
            Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model
