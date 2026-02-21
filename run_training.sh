#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./run_training.sh [options]

Pipeline:
  1) Build manifest from dataset/words/training
  2) Extract pooled hand features from videos
  3) Train RandomForest model

Options:
  --python PATH            Python executable (default: python3)
  --training-dir PATH      Training root (default: dataset/words/training)
  --manifest-out PATH      Manifest CSV output (default: data/words_manifest.csv)
  --features-out PATH      Features CSV output (default: data/words_features_seg3_2h.csv)
  --model-out PATH         Model output (default: models_words/rf_words_seg3_2h.joblib)
  --pool MODE              Pool mode: mean|seg3 (default: seg3)
  --max-frames N           Max valid frames/sample (default: 80)
  --sample-fps N           Sampling fps (default: 10)
  --min-detected-ratio F   Min detected ratio detected_any/total_frames (default: 0.8)
  --cv N                   CV folds for training (default: 5)
  --n-estimators N         RF trees (default: 500)
  --test-size F            Test size in train/test split (default: 0.2)
  --seed N                 Random seed (default: 42)
  --debug_detect           Print per-video detection summaries and write detect report CSV
  --two-hands              Enable two-hand frame features (default: enabled)
  --one-hand               Use one-hand frame features
  -h, --help               Show this help

Example:
  ./run_training.sh --pool seg3 --cv 5
EOF
}

PYTHON_BIN="python3"
TRAINING_DIR="dataset/words/training"
MANIFEST_OUT="data/words_manifest.csv"
MODEL_OUT="models_words/rf_words_seg3_2h.joblib"
FEATURES_OUT="data/words_features_seg3_2h.csv"
POOL="seg3"
MAX_FRAMES="80"
SAMPLE_FPS="10"
MIN_DETECTED_RATIO="0.8"
CV_FOLDS="5"
N_ESTIMATORS="500"
TEST_SIZE="0.2"
SEED="42"
TWO_HANDS="1"
DEBUG_DETECT="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --training-dir) TRAINING_DIR="$2"; shift 2 ;;
    --manifest-out) MANIFEST_OUT="$2"; shift 2 ;;
    --features-out) FEATURES_OUT="$2"; shift 2 ;;
    --model-out) MODEL_OUT="$2"; shift 2 ;;
    --pool) POOL="$2"; shift 2 ;;
    --max-frames) MAX_FRAMES="$2"; shift 2 ;;
    --sample-fps) SAMPLE_FPS="$2"; shift 2 ;;
    --min-detected-ratio) MIN_DETECTED_RATIO="$2"; shift 2 ;;
    --min-detected) MIN_DETECTED_RATIO="$2"; shift 2 ;; # backward-compatible alias
    --cv) CV_FOLDS="$2"; shift 2 ;;
    --n-estimators) N_ESTIMATORS="$2"; shift 2 ;;
    --test-size) TEST_SIZE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --debug_detect|--debug-detect) DEBUG_DETECT="1"; shift ;;
    --two-hands) TWO_HANDS="1"; shift ;;
    --one-hand) TWO_HANDS="0"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ ! -d "$TRAINING_DIR" ]]; then
  echo "ERROR: training dir not found: $TRAINING_DIR" >&2
  exit 1
fi

if [[ "$POOL" != "mean" && "$POOL" != "seg3" ]]; then
  echo "ERROR: --pool must be mean or seg3" >&2
  exit 1
fi

mkdir -p "$(dirname "$MANIFEST_OUT")" "$(dirname "$FEATURES_OUT")" "$(dirname "$MODEL_OUT")"

echo "[1/3] Build manifest from training videos..."
"$PYTHON_BIN" scripts_words/build_manifest_training.py \
  --training_dir "$TRAINING_DIR" \
  --out "$MANIFEST_OUT"

row_count="$("$PYTHON_BIN" - <<PY
import csv
from pathlib import Path
p = Path("$MANIFEST_OUT")
with p.open("r", encoding="utf-8", newline="") as f:
    n = sum(1 for _ in csv.reader(f)) - 1
print(max(0, n))
PY
)"

if [[ "$row_count" -le 0 ]]; then
  echo "ERROR: manifest has no rows. Add videos under $TRAINING_DIR/<label>/ first." >&2
  exit 1
fi

echo "[2/3] Extract pooled features..."
feature_cmd=(
  "$PYTHON_BIN" scripts_words/make_features_csv.py
  --manifest "$MANIFEST_OUT"
  --out "$FEATURES_OUT"
  --pool "$POOL"
  --max_frames "$MAX_FRAMES"
  --sample_fps "$SAMPLE_FPS"
  --min_detected_ratio "$MIN_DETECTED_RATIO"
)
if [[ "$TWO_HANDS" == "1" ]]; then
  feature_cmd+=(--two_hands)
fi
if [[ "$DEBUG_DETECT" == "1" ]]; then
  feature_cmd+=(--debug_detect)
fi
"${feature_cmd[@]}"

echo "[3/3] Train RandomForest..."
"$PYTHON_BIN" scripts_words/train_rf_pool.py \
  --csv "$FEATURES_OUT" \
  --model_out "$MODEL_OUT" \
  --cv "$CV_FOLDS" \
  --n_estimators "$N_ESTIMATORS" \
  --test_size "$TEST_SIZE" \
  --seed "$SEED"

echo "Done."
echo "  Manifest: $MANIFEST_OUT"
echo "  Features: $FEATURES_OUT"
echo "  Model:    $MODEL_OUT"
