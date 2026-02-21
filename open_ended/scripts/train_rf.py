#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RandomForest on ASL hand-landmark CSV."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/asl_landmarks.csv"),
        help="Input CSV path.",
    )
    parser.add_argument(
        "--model_out",
        type=Path,
        default=Path("models/rf_asl.joblib"),
        help="Output model path.",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=300,
        help="Number of trees for RandomForestClassifier.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.n_estimators <= 0:
        raise ValueError("--n_estimators must be > 0.")
    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

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

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4, zero_division=0)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_out)

    classes_path = args.model_out.parent / "label_classes.txt"
    classes_path.write_text("\n".join(model.classes_) + "\n", encoding="utf-8")

    print(f"Loaded rows: {len(df)}")
    print(f"Train rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(report)
    print(f"Saved model: {args.model_out}")
    print(f"Saved classes: {classes_path}")


if __name__ == "__main__":
    main()
