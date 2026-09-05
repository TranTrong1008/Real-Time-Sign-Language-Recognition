"""
evaluate_realtime_testset.py - Đánh giá accuracy 3 phiên bản model Transformer
(Hands / Pose+Hands / Holistic) trên tập test real-time đã ghi.

Mỗi sample .npz chứa raw landmarks Holistic đầy đủ (pose+face+2 tay), từ đó
suy ra cả 3 bộ feature (126/258/534) để đánh giá công bằng trên cùng 1 lần
thực hiện ký hiệu thật, không phải 3 lần quay riêng biệt.

Cách chạy:
    python evaluate_realtime_testset.py --test_set_dir ../../data/realtime_test_set --labels_file ../../configs/labels.json --model_hands ../../models/transformer_hands_best.keras --model_pose_hands ../../models/transformer_pose_hands_best.keras --model_holistic ../../models/transformer_best.keras --output_dir realtime_eval_output
"""
import argparse
import json
from pathlib import Path

import numpy as np

from sign_preprocessing import build_feature_sequence, T

VERSIONS = [
    ("hands", 126),
    ("pose_hands", 258),
    ("holistic", 534),
]


def load_samples(test_set_dir: str):
    """Trả về list of dict {pose_raw, face_raw, lh_raw, rh_raw, label}."""
    samples = []
    for npz_path in sorted(Path(test_set_dir).rglob("sample_*.npz")):
        data = np.load(npz_path, allow_pickle=False)
        samples.append({
            "pose_raw": data["pose_raw"],
            "face_raw": data["face_raw"],
            "lh_raw": data["lh_raw"],
            "rh_raw": data["rh_raw"],
            "label": str(data["label"]),
            "path": str(npz_path),
        })
    return samples


def evaluate_version(samples, model, class_to_idx, feature_dim):
    """Chạy 1 phiên bản model trên toàn bộ samples, trả về y_true, y_pred, per_sample_records."""
    y_true, y_pred, records = [], [], []

    for s in samples:
        seq = build_feature_sequence(
            s["pose_raw"], s["face_raw"], s["lh_raw"], s["rh_raw"], feature_dim=feature_dim, T=T
        )
        batch = np.expand_dims(seq, axis=0)
        probabilities = np.asarray(model(batch, training=False))[0]
        pred_idx = int(np.argmax(probabilities))
        true_idx = class_to_idx[s["label"]]

        y_true.append(true_idx)
        y_pred.append(pred_idx)
        records.append({
            "path": s["path"],
            "true_label": s["label"],
            "pred_idx": pred_idx,
            "confidence": float(probabilities[pred_idx]),
            "correct": pred_idx == true_idx,
        })

    return np.array(y_true), np.array(y_pred), records


def compute_metrics(y_true, y_pred, num_classes):
    """Accuracy tổng + macro precision/recall/f1"""
    accuracy = float(np.mean(y_true == y_pred))

    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        support = np.sum(y_true == c)
        if support == 0:
            continue  # class không xuất hiện trong tập test real-time này
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        precisions.append(precision); recalls.append(recall); f1s.append(f1)

    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precisions)) if precisions else 0.0,
        "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Đánh giá 3 phiên bản model trên tập test real-time")
    parser.add_argument("--test_set_dir", required=True)
    parser.add_argument("--labels_file", required=True)
    parser.add_argument("--model_hands", required=True)
    parser.add_argument("--model_pose_hands", required=True)
    parser.add_argument("--model_holistic", required=True)
    parser.add_argument("--output_dir", default="realtime_eval_output")
    args = parser.parse_args()

    with open(args.labels_file, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    num_classes = len(class_names)

    samples = load_samples(args.test_set_dir)
    print(f"Đã tải {len(samples)} sample từ tập test real-time.")
    missing_labels = {s["label"] for s in samples} - set(class_to_idx)
    if missing_labels:
        raise ValueError(f"Có label trong tập test không khớp labels.json: {missing_labels}")

    import tensorflow as tf
    model_paths = {
        "hands": args.model_hands,
        "pose_hands": args.model_pose_hands,
        "holistic": args.model_holistic,
    }

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    for version_name, feature_dim in VERSIONS:
        print(f"\n=== Đánh giá phiên bản '{version_name}' (feature_dim={feature_dim}) ===")
        model = tf.keras.models.load_model(model_paths[version_name], compile=False)

        y_true, y_pred, records = evaluate_version(samples, model, class_to_idx, feature_dim)
        metrics = compute_metrics(y_true, y_pred, num_classes)
        all_metrics[version_name] = metrics

        print(f"Accuracy      : {metrics['accuracy']:.4f}")
        print(f"Macro Precision: {metrics['macro_precision']:.4f}")
        print(f"Macro Recall  : {metrics['macro_recall']:.4f}")
        print(f"Macro F1      : {metrics['macro_f1']:.4f}")

        with open(output_root / f"{version_name}_predictions.json", "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("SO SÁNH 3 PHIÊN BẢN TRÊN TẬP TEST REAL-TIME")
    print("=" * 60)
    print(f"{'Version':<15} {'Accuracy':>10} {'Macro-F1':>10}")
    for version_name, _ in VERSIONS:
        m = all_metrics[version_name]
        print(f"{version_name:<15} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f}")

    with open(output_root / "summary_comparison.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu chi tiết vào: {output_root}/")


if __name__ == "__main__":
    main()
