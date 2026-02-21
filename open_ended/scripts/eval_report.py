#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained RF model with test split report and confusion matrix."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/asl_landmarks.csv"),
        help="Input CSV path.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/rf_asl.joblib"),
        help="Trained model path.",
    )
    parser.add_argument(
        "--report_dir",
        type=Path,
        default=Path("reports"),
        help="Directory to store evaluation artifacts.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducible split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not (0.0 < args.test_size < 1.0):
        raise ValueError("--test_size must be between 0 and 1.")

    df = pd.read_csv(args.csv)
    if "label" not in df.columns:
        raise ValueError("CSV must include a 'label' column.")

    feature_cols = [c for c in df.columns if c != "label"]
    if len(feature_cols) != 42:
        raise ValueError(f"Expected 42 feature columns, got {len(feature_cols)}.")

    X = df[feature_cols].values
    y = df["label"].astype(str).values

    class_counts = pd.Series(y).value_counts()
    low_count = class_counts[class_counts < 2]
    if not low_count.empty:
        detail = ", ".join([f"{k}:{int(v)}" for k, v in low_count.items()])
        raise ValueError(
            "Each class needs at least 2 samples for stratified split. "
            f"Too small classes: {detail}"
        )

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    model = joblib.load(args.model)
    y_pred = model.predict(X_test)

    labels = [str(c) for c in model.classes_]
    accuracy = accuracy_score(y_test, y_pred)
    report_str = classification_report(
        y_test, y_pred, labels=labels, digits=4, zero_division=0
    )
    report_dict = classification_report(
        y_test, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    # Lowest-performing labels by F1 score (support > 0 only).
    per_label = []
    for label in labels:
        item = report_dict.get(label, {})
        support = int(item.get("support", 0))
        if support > 0:
            per_label.append(
                {
                    "label": label,
                    "f1": float(item.get("f1-score", 0.0)),
                    "precision": float(item.get("precision", 0.0)),
                    "recall": float(item.get("recall", 0.0)),
                    "support": support,
                }
            )
    worst5 = sorted(per_label, key=lambda x: (x["f1"], x["support"]))[:5]

    args.report_dir.mkdir(parents=True, exist_ok=True)
    cm_out = args.report_dir / "confusion_matrix.png"
    metrics_out = args.report_dir / "metrics.txt"

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig_size = max(12, int(len(labels) * 0.45))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title="ASL Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
    fig.tight_layout()
    fig.savefig(cm_out, dpi=200)
    plt.close(fig)

    lines = []
    lines.append(f"CSV: {args.csv}")
    lines.append(f"Model: {args.model}")
    lines.append(f"Rows: {len(df)}")
    lines.append(f"Test rows: {len(y_test)}")
    lines.append(f"Accuracy: {accuracy:.4f}")
    lines.append("")
    lines.append("Lowest 5 labels by F1:")
    for item in worst5:
        lines.append(
            f"  {item['label']}: f1={item['f1']:.4f}, "
            f"precision={item['precision']:.4f}, recall={item['recall']:.4f}, "
            f"support={item['support']}"
        )
    lines.append("")
    lines.append("Classification report:")
    lines.append(report_str)
    metrics_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Accuracy: {accuracy:.4f}")
    print("Lowest 5 labels by F1:")
    for item in worst5:
        print(
            f"  {item['label']}: f1={item['f1']:.4f}, "
            f"precision={item['precision']:.4f}, recall={item['recall']:.4f}, "
            f"support={item['support']}"
        )
    print(f"Saved confusion matrix: {cm_out}")
    print(f"Saved metrics: {metrics_out}")


if __name__ == "__main__":
    main()
