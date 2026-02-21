# A for the W - HackAI_2026

Real-time sign word recognition from webcam/video using MediaPipe hand landmarks + Random Forest, with optional face emotion and emotional TTS.

## Team
- Team Name: `A for the W`
- Members:
  - `Jaeha Lee`
  - `Caden Burklund`

## What This Project Does
- Learns sign-word classes from videos in `dataset/words/training/<label>/*.mp4`
- Extracts hand landmarks per frame with MediaPipe Hands
- Builds pooled temporal features (`seg3` by default)
- Trains a Random Forest classifier for word prediction
- Runs realtime webcam inference with stabilized predictions
- Optionally adds Face Mesh-based emotion (`happy / neutral / surprised`) and emotional TTS

## Tech Stack
- Python 3.11
- OpenCV
- MediaPipe Hands + Face Mesh
- NumPy / Pandas
- scikit-learn (RandomForest)
- joblib
- matplotlib (reports)
- macOS `afplay` + `say` for TTS

## Project Layout
```text
dataset/words/training/<label>/*.mp4   # training videos
data/                                   # generated CSVs / reports
models_words/                           # trained RF models
scripts_words/                          # manifest, features, train, realtime
voice/
  happy/
  neutral/
  surprised/                            # optional emotion-specific mp3 clips
```

## How To Run

### 1) Setup (macOS, Python 3.11)
```bash
cd /Users/dlwogk7939/Documents/GitHub/HackAI_2026
source venv311/bin/activate
pip install -r requirements.txt
```

### 2) Train the Model
```bash
./run_training.sh
```

Outputs:
- `data/words_manifest.csv`
- `data/words_features_seg3_2h.csv`
- `models_words/rf_words_seg3_2h.joblib`

### 3) Run Realtime Demo
```bash
python scripts_words/realtime_sequence_pool.py \
  --model models_words/rf_words_seg3_2h.joblib \
  --pool seg3 \
  --mirror \
  --enable_emotion
```

### 4) Bigger Display (Optional)
```bash
python scripts_words/realtime_sequence_pool.py \
  --model models_words/rf_words_seg3_2h.joblib \
  --pool seg3 \
  --mirror \
  --enable_emotion \
  --display_width 1800 \
  --display_height 1100 \
  --fullscreen
```

## Controls (Realtime)
- `q`: quit
- `r`: reset buffers/history
- `t` or `=`: threshold up
- `T` or `-`: threshold down

## Emotion + TTS
- Emotion classes: `happy`, `neutral`, `surprised`
- If `--enable_emotion` is on, emotion is estimated from Face Mesh landmarks
- TTS speaks only on stable label changes (anti-spam logic)
- MP3 voice clips are used first; fallback is macOS `say`

Voice clip naming:
```text
voice/happy/happy_<label>.mp3
voice/neutral/neutral_<label>.mp3
voice/surprised/surprised_<label>.mp3
```

Example with custom macOS fallback voice:
```bash
python scripts_words/realtime_sequence_pool.py \
  --model models_words/rf_words_seg3_2h.joblib \
  --enable_emotion \
  --tts_voice Yuna
```

## Debug Feature Extraction Skips
If you want detailed keep/skip reasons during feature extraction:
```bash
./run_training.sh --debug_detect
```

Report file:
```text
data/detect_report_seg3_2h.csv
```
