from __future__ import annotations

import subprocess
from pathlib import Path

EMOTIONS = {"happy", "neutral", "surprised"}
LABEL_ALIASES = {
    "j": ["jay"],
    "jay": ["j"],
}

NEUTRAL_RATE = 180
HAPPY_RATE = 210
SURPRISED_RATE = 220


def build_emotional_utterance(label: str, emotion: str) -> tuple[str, int]:
    if emotion == "happy":
        return f"{label}!", HAPPY_RATE
    if emotion == "surprised":
        return f"{label}?!", SURPRISED_RATE
    return label, NEUTRAL_RATE


def _normalize_label(label: str) -> str:
    key = label.strip().lower().replace(" ", "_").replace("-", "_")
    return "".join(ch for ch in key if ch.isalnum() or ch == "_")


def _label_candidates(label: str) -> list[str]:
    key = _normalize_label(label)
    candidates = [key]
    for alias in LABEL_ALIASES.get(key, []):
        if alias not in candidates:
            candidates.append(alias)
    return candidates


def _find_voice_clip(label: str, emotion: str, voice_dir: str | Path) -> Path | None:
    emo = emotion.lower().strip()
    if emo not in EMOTIONS:
        emo = "neutral"
    root = Path(voice_dir)
    emo_dir = root / emo
    for cand in _label_candidates(label):
        named = emo_dir / f"{emo}_{cand}.mp3"
        if named.exists():
            return named
        plain = emo_dir / f"{cand}.mp3"
        if plain.exists():
            return plain
    return None


def _play_clip(path: Path) -> bool:
    try:
        subprocess.Popen(
            ["afplay", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def speak_emotional(
    label: str,
    emotion: str,
    voice: str | None = None,
    voice_dir: str | Path = "voice",
) -> None:
    clip = _find_voice_clip(label=label, emotion=emotion, voice_dir=voice_dir)
    if clip is not None and _play_clip(clip):
        return

    text, rate = build_emotional_utterance(label, emotion)
    cmd = ["say", "-r", str(rate)]
    if voice:
        cmd.extend(["-v", voice])
    cmd.append(text)
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        # Keep realtime loop alive even if TTS is unavailable.
        pass
