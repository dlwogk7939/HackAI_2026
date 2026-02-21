#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RandomForest on pooled WLASL video features."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/words_features.csv"),
        help="Input pooled-feature CSV path.",
    )
    parser.add_argument(
        "--model_out",
        type=Path,
        default=Path("models_words/rf_words.joblib"),
        help="Output model path.",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=500,
        help="Number of trees.",
    )
    parser.add_argument(
        "--use_explicit_split",
        action="store_true",
        default=False,
        help="Use CSV split column (train/val/test). Default is False.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test ratio for stratified train_test_split.",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Number of CV folds (0 or 1 disables CV).",
    )
    parser.add_argument(
        "--min_per_class",
        type=int,
        default=2,
        help="Minimum samples per class to keep.",
    )
    parser.add_argument(
        "--drop_rare",
        dest="drop_rare",
        action="store_true",
        default=True,
        help="Drop classes with sample count < --min_per_class (default: True).",
    )
    parser.add_argument(
        "--no_drop_rare",
        dest="drop_rare",
        action="store_false",
        help="Disable rare-class dropping.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    feat_cols = [c for c in df.columns if c.startswith("f")]
    if not feat_cols:
        fail("No feature columns found (expected columns starting with 'f').")

    def sort_key(col: str):
        suffix = col[1:]
        if suffix.isdigit():
            return (0, int(suffix), col)
        return (1, 0, col)

    feat_cols = sorted(feat_cols, key=sort_key)
    return feat_cols


def print_class_counts(name: str, labels) -> None:
    counts = pd.Series(labels).value_counts().sort_index()
    print(f"{name} class counts ({len(counts)} classes):")
    for label, cnt in counts.items():
        print(f"  {label}: {int(cnt)}")


def ensure_train_class_counts(y_train) -> None:
    counts = pd.Series(y_train).value_counts()
    low = counts[counts < 2]
    if not low.empty:
        details = ", ".join([f"{k}:{int(v)}" for k, v in low.items()])
        fail(
            "Chosen train set has classes with fewer than 2 samples. "
            f"Fix data or split strategy. Offending classes: {details}"
        )


def main() -> None:
    args = parse_args()

    if not args.csv.exists():
        fail(f"CSV not found: {args.csv}")
    if args.n_estimators <= 0:
        fail("--n_estimators must be > 0.")
    if not (0.0 < args.test_size < 1.0):
        fail("--test_size must be between 0 and 1.")
    if args.cv < 0:
        fail("--cv must be >= 0.")
    if args.min_per_class <= 0:
        fail("--min_per_class must be > 0.")

    df = pd.read_csv(args.csv)

    required_cols = {"label_gloss"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        fail(f"Missing required columns: {missing}")

    df["label_gloss"] = df["label_gloss"].astype(str)

    counts_before = df["label_gloss"].value_counts()
    rare_counts = counts_before[counts_before < args.min_per_class]
    if args.drop_rare and not rare_counts.empty:
        print("Dropping rare labels (count < --min_per_class):")
        dropped_total = 0
        for label, cnt in rare_counts.sort_index().items():
            print(f"  {label}: {int(cnt)}")
            dropped_total += int(cnt)
        df = df[~df["label_gloss"].isin(rare_counts.index)].copy()
        print(f"Dropped samples total: {dropped_total}")

    if df.empty:
        fail("Dataset is empty after filtering.")
    if df["label_gloss"].nunique() < 2:
        fail("Need at least 2 classes after filtering.")

    feature_cols = get_feature_columns(df)
    X_all = df[feature_cols].astype("float32").values
    y_all = df["label_gloss"].astype(str).values

    if args.use_explicit_split:
        if "split" not in df.columns:
            fail("--use_explicit_split was passed but 'split' column is missing.")

        split_series = df["split"].astype(str).str.lower()
        train_mask = split_series == "train"
        test_mask = split_series == "test"
        val_mask = split_series == "val"

        if int(train_mask.sum()) == 0 or int(test_mask.sum()) == 0:
            fail("Explicit split requires non-empty train and test rows.")

        X_train = df.loc[train_mask, feature_cols].astype("float32").values
        y_train = df.loc[train_mask, "label_gloss"].astype(str).values
        X_test = df.loc[test_mask, feature_cols].astype("float32").values
        y_test = df.loc[test_mask, "label_gloss"].astype(str).values

        print("Using explicit split from CSV.")
        print(
            f"Split sizes: train={int(train_mask.sum())}, "
            f"val={int(val_mask.sum())}, test={int(test_mask.sum())}"
        )
    else:
        print("Using stratified train_test_split (default behavior).")
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_all,
                y_all,
                test_size=args.test_size,
                random_state=args.seed,
                stratify=y_all,
            )
        except ValueError as e:
            fail(f"train_test_split failed: {e}")

    print(f"Total samples used: {len(y_all)}")
    print(f"Train size: {len(y_train)}")
    print(f"Test size: {len(y_test)}")

    ensure_train_class_counts(y_train)

    print_class_counts("Train", y_train)
    print_class_counts("Test", y_test)

    if args.cv > 1:
        class_counts = pd.Series(y_all).value_counts()
        min_class_count = int(class_counts.min())
        effective_cv = args.cv
        if min_class_count < effective_cv:
            effective_cv = min_class_count
            print(
                f"Requested cv={args.cv}, but smallest class has {min_class_count} samples. "
                f"Using cv={effective_cv}."
            )
        if effective_cv < 2:
            print("Skipping CV: effective folds < 2.")
            effective_cv = 0

        if effective_cv >= 2:
            cv_model = RandomForestClassifier(
                n_estimators=args.n_estimators,
                n_jobs=-1,
                random_state=args.seed,
            )
            skf = StratifiedKFold(n_splits=effective_cv, shuffle=True, random_state=args.seed)
            try:
                cv_scores = cross_val_score(
                    cv_model, X_all, y_all, cv=skf, scoring="accuracy", n_jobs=-1
                )
            except (PermissionError, OSError) as e:
                print(f"CV parallel execution unavailable ({e}). Retrying with n_jobs=1.")
                cv_scores = cross_val_score(
                    cv_model, X_all, y_all, cv=skf, scoring="accuracy", n_jobs=1
                )
            print(
                f"CV accuracy ({effective_cv}-fold): "
                f"{cv_scores.mean():.4f} +/- {cv_scores.std():.4f}"
            )

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        n_jobs=-1,
        random_state=args.seed,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_out)

    classes_path = args.model_out.parent / "label_classes.txt"
    classes_path.write_text("\n".join(model.classes_) + "\n", encoding="utf-8")

    print(f"Saved model: {args.model_out}")
    print(f"Saved classes: {classes_path}")


if __name__ == "__main__":
    main()

# Example usage:
# python scripts_words/train_rf_pool.py --csv data/words_features.csv --model_out models_words/rf_words.joblib --cv 5
# python scripts_words/train_rf_pool.py --csv data/words_features.csv --use_explicit_split
