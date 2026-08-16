from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Bidirectional,
    Conv1D,
    GlobalAveragePooling1D,
    Dense,
    Dropout,
)
from model_config import INPUT_SHAPE, NUM_CLASSES

# Loss function sử dụng: sparse_categorical_crossentropy (vì label là số nguyên, không one-hot)


def build_lstm_model():
    model = Sequential(
        [
            LSTM(64, return_sequences=True, activation="relu", input_shape=INPUT_SHAPE),
            LSTM(128, return_sequences=False, activation="relu"),
            Dense(64, activation="relu"),
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
                LSTM(64, return_sequences=True, activation="relu"),
                input_shape=INPUT_SHAPE,
            ),
            Bidirectional(LSTM(128, return_sequences=False, activation="relu")),
            Dense(64, activation="relu"),
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
            Conv1D(filters=128, kernel_size=3, activation="relu"),
            GlobalAveragePooling1D(),
            Dense(64, activation="relu"),
            Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model
