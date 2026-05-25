# ============================================================
# core/emotion_engine.py
# Real-Time Mock Interview Evaluator
# ============================================================
# Wav2Vec2-based audio emotion/prosody classifier.
# Model is lazy-loaded once and cached by Streamlit to prevent
# repeated 360MB downloads or re-initialization overhead.
# ============================================================

import logging
import gc
import numpy as np
import streamlit as st
from transformers import pipeline as hf_pipeline

from config import EMOTION_MODEL_ID, TARGET_SAMPLE_RATE, EMOTION_LABEL_MAP, EMOTION_LABEL_FALLBACK

logger = logging.getLogger(__name__)


# ─── Model Loading (Cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_emotion_pipeline():
    """
    Load the Wav2Vec2 audio classification pipeline once and cache it for the
    entire Streamlit session. @st.cache_resource ensures the model is NOT
    reloaded on every rerun — critical for performance.

    Auto-detects CUDA GPU (device=0) and falls back to CPU (device=-1).
    """
    import torch
    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU (CUDA)" if device == 0 else "CPU"
    logger.info(f"Loading Emotion Model [{EMOTION_MODEL_ID}] on {device_name}...")

    classifier = hf_pipeline(
        task="audio-classification",
        model=EMOTION_MODEL_ID,
        device=device,
    )
    logger.info("Emotion model loaded successfully.")
    return classifier


# ─── Public Analysis Function ─────────────────────────────────────────────────
def analyze_emotion(waveform: np.ndarray) -> dict:
    """
    Run the Wav2Vec2 emotion classifier on a preprocessed audio waveform.

    The pipeline expects:
    - A 1D float32 numpy array
    - At 16kHz sample rate (enforced by audio_processor.py)

    Args:
        waveform: Preprocessed 1D float32 numpy array at 16kHz.

    Returns:
        dict with keys:
            - "label"      (str):   Raw model label (e.g., "happy")
            - "score"      (float): Confidence score 0.0–1.0
            - "icon"       (str):   Emoji for the UI
            - "rating"     (str):   "positive" | "neutral" | "negative" | "critical"
            - "coaching"   (str):   Actionable coaching message
            - "all_scores" (list):  Full ranked list of (label, score) tuples
    """
    if waveform is None or waveform.size == 0:
        raise ValueError("Empty waveform passed to emotion engine.")

    classifier = _load_emotion_pipeline()

    try:
        # The HuggingFace audio-classification pipeline accepts a numpy array directly.
        # top_k=None returns scores for ALL emotion labels — used for full breakdown.
        results = classifier(
            {"raw": waveform, "sampling_rate": TARGET_SAMPLE_RATE},
            top_k=None,
        )
        # results: [{"label": "happy", "score": 0.91}, {"label": "sad", "score": 0.05}, ...]
        # Already sorted by score descending.

    except Exception as e:
        logger.error(f"Emotion pipeline failed: {e}")
        raise RuntimeError(f"Emotion analysis failed: {e}") from e
    finally:
        # Explicit cleanup of any intermediate tensors
        gc.collect()

    # ── Extract top prediction ───────────────────────────────────────────
    top = results[0]
    raw_label  = top["label"].lower().strip()
    confidence = round(top["score"] * 100, 1)  # Convert to percentage

    # ── Map label to coaching metadata ───────────────────────────────────
    metadata = EMOTION_LABEL_MAP.get(raw_label, EMOTION_LABEL_FALLBACK)

    # ── Build full score breakdown for UI charts ─────────────────────────
    all_scores = [(r["label"].capitalize(), round(r["score"] * 100, 1)) for r in results]

    logger.info(f"Emotion detected: {raw_label} ({confidence}%)")

    return {
        "label":      raw_label.capitalize(),
        "score":      confidence,
        "icon":       metadata["icon"],
        "rating":     metadata["rating"],
        "coaching":   metadata["coaching"],
        "all_scores": all_scores,
    }
