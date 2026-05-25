# ============================================================
# core/transcription_engine.py
# Real-Time Mock Interview Evaluator
# ============================================================
# OpenAI Whisper-based Automatic Speech Recognition (ASR).
# Uses HuggingFace pipeline for consistent API and GPU support.
# Model is cached via @st.cache_resource to prevent reloading.
# ============================================================

import gc
import time
import logging
import numpy as np
import streamlit as st
from transformers import pipeline as hf_pipeline

from config import WHISPER_MODEL_ID, TARGET_SAMPLE_RATE

logger = logging.getLogger(__name__)


# ─── Model Loading (Cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_whisper_pipeline():
    """
    Load the Whisper ASR model once and cache it for the entire session.
    Supports chunked long-form transcription (chunk_length_s=30) to handle
    recordings longer than the model's native 30s context window.

    chunk_length_s=30 + stride_length_s=5 enables overlap-and-add decoding
    for smooth transcription at longer durations.
    """
    import torch
    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU (CUDA)" if device == 0 else "CPU"
    logger.info(f"Loading Whisper Model [{WHISPER_MODEL_ID}] on {device_name}...")

    asr = hf_pipeline(
        task="automatic-speech-recognition",
        model=WHISPER_MODEL_ID,
        chunk_length_s=30,       # Process audio in 30-second chunks
        stride_length_s=5,       # 5-second overlap between chunks for smooth output
        return_timestamps=False, # We only need raw text, not word timestamps
        device=device,
    )
    logger.info("Whisper model loaded successfully.")
    return asr


# ─── Public Transcription Function ───────────────────────────────────────────
def transcribe(waveform: np.ndarray) -> dict:
    """
    Transcribe a preprocessed audio waveform to text using Whisper.

    The pipeline accepts:
    - A dict with "raw" (numpy array) and "sampling_rate" keys, OR
    - A 1D float32 numpy array directly

    Args:
        waveform: Preprocessed 1D float32 numpy array at 16kHz.

    Returns:
        dict with keys:
            - "text"       (str):   Full transcript text string
            - "latency_ms" (float): Transcription wall-clock time in milliseconds
            - "word_count" (int):   Approximate word count from transcript
    """
    if waveform is None or waveform.size == 0:
        raise ValueError("Empty waveform passed to transcription engine.")

    asr = _load_whisper_pipeline()

    start_time = time.perf_counter()

    try:
        # Pass as dict to ensure sampling_rate is respected even if the
        # pipeline's internal feature extractor tries to re-sample.
        result = asr({"raw": waveform, "sampling_rate": TARGET_SAMPLE_RATE})

        # result format: {"text": "...", "chunks": [...]} (chunks only if timestamps=True)
        transcript_text = result.get("text", "").strip()

    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}")
        raise RuntimeError(f"Transcription failed: {e}") from e
    finally:
        gc.collect()

    latency_ms = round((time.perf_counter() - start_time) * 1000, 1)
    word_count  = len(transcript_text.split()) if transcript_text else 0

    logger.info(
        f"Transcription complete: {word_count} words in {latency_ms}ms"
    )

    return {
        "text":       transcript_text,
        "latency_ms": latency_ms,
        "word_count": word_count,
    }
