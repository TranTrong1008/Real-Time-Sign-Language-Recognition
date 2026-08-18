import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# Setup directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "holistic")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def build_transformer(input_shape=(30, 534), num_classes=25, d_model=128, num_heads=4, ff_dim=256):
    inputs = layers.Input(shape=input_shape)
    
    # Linear projection to latent dimension d_model
    x = layers.Dense(d_model)(inputs)
    
    # Learnable positional embedding for temporal sequence
    positions = tf.range(start=0, limit=input_shape[0], delta=1)
    pos_emb = layers.Embedding(input_dim=input_shape[0], output_dim=d_model)(positions)
    x = x + pos_emb

    # Transformer Encoder Block
    # 1. Multi-Head Self-Attention with residual connection
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(x, x)
    attn = layers.Dropout(0.2)(attn)
    x1 = layers.LayerNormalization(epsilon=1e-6)(x + attn)
    
    # 2. Feed-Forward Network with residual connection
    ffn = layers.Dense(ff_dim, activation="relu")(x1)
    ffn = layers.Dense(d_model)(ffn)
    ffn = layers.Dropout(0.2)(ffn)
    x = layers.LayerNormalization(epsilon=1e-6)(x1 + ffn)

    # Classification head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs=inputs, outputs=outputs, name="SignLanguage_Transformer")

def main():
    # Load holistic keypoint dataset
    print("[INFO] Loading dataset from data/holistic/...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train_aug.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train_aug.npy"))
    X_val   = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val   = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    # Convert one-hot encoded labels to integer indices if necessary
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        y_train = np.argmax(y_train, axis=1)
        y_val = np.argmax(y_val, axis=1)
        y_test_true = np.argmax(y_test, axis=1)
    else:
        y_test_true = y_test

    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(np.unique(np.concatenate([y_train, y_val, y_test_true])))
    print(f"[INFO] Input shape: {input_shape} | Classes: {num_classes}")

    # Build and compile model
    model = build_transformer(input_shape=input_shape, num_classes=num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    # Training callbacks
    model_save_path = os.path.join(MODELS_DIR, "transformer_best.keras")
    my_callbacks = [
        callbacks.ModelCheckpoint(model_save_path, save_best_only=True, monitor="val_accuracy", verbose=1),
        callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1)
    ]

    # Train model
    print("\n[INFO] Starting Transformer training...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=my_callbacks
    )

    # Evaluate on independent test set (Signer-based split)
    print("\n[INFO] Evaluating on test set...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_test_true, y_pred)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_test_true, y_pred, average='macro', zero_division=0
    )

    print("\n" + "=" * 50)
    print("       TRANSFORMER EVALUATION RESULTS (TEST SET)      ")
    print("=" * 50)
    print(f"Accuracy:        {acc * 100:.2f}%")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall:    {macro_recall:.4f}")
    print(f"Macro F1-score:  {macro_f1:.4f}")
    print("=" * 50)

    # Export classification report
    report = classification_report(y_test_true, y_pred, digits=4, zero_division=0)
    metrics_path = os.path.join(RESULTS_DIR, "transformer_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\nMacro Precision: {macro_precision:.4f}\nMacro Recall: {macro_recall:.4f}\nMacro F1-score: {macro_f1:.4f}\n\n")
        f.write(report)

    # Plot and save confusion matrix
    cm = confusion_matrix(y_test_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix - Transformer Model")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "transformer_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    print(f"\n[SUCCESS] Best model saved to: {model_save_path}")
    print(f"[SUCCESS] Metrics and confusion matrix saved to: {RESULTS_DIR}")

if __name__ == "__main__":
    main()