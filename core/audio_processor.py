# ============================================================
# core/audio_processor.py
# Real-Time Mock Interview Evaluator
# ============================================================
# Handles all raw audio ingestion, resampling, and buffer
# cleanup. Designed to be memory-safe on both local machines
# and remote server deployments.
# ============================================================

import io
import gc
import logging
import numpy as np
import librosa
import soundfile as sf

from config import TARGET_SAMPLE_RATE, MAX_AUDIO_DURATION_SEC, MIN_AUDIO_DURATION_SEC

logger = logging.getLogger(__name__)


def process_audio(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    Convert raw audio bytes (from Streamlit's st.audio_input) into a
    clean, resampled float32 numpy waveform suitable for Wav2Vec2 and Whisper.

    Strategy:
    - Use an in-memory BytesIO buffer — NEVER write to disk.
    - Resample to TARGET_SAMPLE_RATE (16kHz) using Librosa's high-quality
      resampler (res_type='kaiser_fast').
    - Convert to mono by averaging channels if stereo.
    - Enforce duration bounds to prevent pipeline overloads.
    - Explicitly delete the BytesIO buffer and call gc.collect() after
      extraction to free memory immediately.

    Args:
        audio_bytes: Raw bytes from st.audio_input() — typically WAV format.

    Returns:
        Tuple of (waveform: np.ndarray[float32], sample_rate: int)

    Raises:
        ValueError: If audio is too short, too long, or unreadable.
    """
    if not audio_bytes:
        raise ValueError("No audio data received. Please record audio first.")

    # ── Step 1: Wrap bytes in an in-memory buffer (no disk I/O) ──────────
    audio_buffer = io.BytesIO(audio_bytes)

    try:
        # ── Step 2: Load audio via SoundFile first to get raw data ───────
        # soundfile is faster than librosa for initial loading.
        # We pass the BytesIO directly — no temp file needed.
        raw_waveform, original_sr = sf.read(audio_buffer, dtype="float32", always_2d=True)

        # raw_waveform shape: (num_samples, num_channels)
        # Convert to mono by averaging across channels (axis=1)
        if raw_waveform.ndim > 1:
            raw_waveform = np.mean(raw_waveform, axis=1)  # shape: (num_samples,)

        logger.info(f"Loaded audio: {original_sr} Hz, {raw_waveform.shape[0]} samples")

    except Exception as e:
        raise ValueError(f"Failed to decode audio bytes: {e}") from e
    finally:
        # ── CRITICAL: Dispose BytesIO buffer immediately ──────────────────
        # Even if an exception occurs, we must free the buffer.
        audio_buffer.close()
        del audio_buffer
        gc.collect()

    # ── Step 3: Validate duration bounds ─────────────────────────────────
    duration_sec = raw_waveform.shape[0] / original_sr

    if duration_sec < MIN_AUDIO_DURATION_SEC:
        del raw_waveform
        gc.collect()
        raise ValueError(
            f"Recording too short ({duration_sec:.1f}s). "
            f"Minimum required: {MIN_AUDIO_DURATION_SEC}s."
        )

    if duration_sec > MAX_AUDIO_DURATION_SEC:
        # Truncate to max duration instead of hard failing
        max_samples = int(MAX_AUDIO_DURATION_SEC * original_sr)
        raw_waveform = raw_waveform[:max_samples]
        logger.warning(f"Audio truncated to {MAX_AUDIO_DURATION_SEC}s max duration.")

    # ── Step 4: Resample to 16kHz if needed ──────────────────────────────
    if original_sr != TARGET_SAMPLE_RATE:
        resampled_waveform = librosa.resample(
            y=raw_waveform,
            orig_sr=original_sr,
            target_sr=TARGET_SAMPLE_RATE,
            res_type="kaiser_fast",  # Balanced quality/speed tradeoff
        )
        # Free the pre-resample buffer
        del raw_waveform
        gc.collect()
        logger.info(f"Resampled: {original_sr} Hz → {TARGET_SAMPLE_RATE} Hz")
    else:
        resampled_waveform = raw_waveform

    # ── Step 5: Ensure float32 dtype (required by both models) ───────────
    waveform = resampled_waveform.astype(np.float32)
    del resampled_waveform
    gc.collect()

    final_duration = waveform.shape[0] / TARGET_SAMPLE_RATE
    logger.info(f"Audio ready: {waveform.shape[0]} samples @ {TARGET_SAMPLE_RATE} Hz ({final_duration:.2f}s)")

    return waveform, TARGET_SAMPLE_RATE


def get_audio_duration(waveform: np.ndarray, sample_rate: int) -> float:
    """
    Helper: Calculate duration in seconds from a waveform array.

    Args:
        waveform: 1D float32 numpy array.
        sample_rate: Sample rate of the waveform.

    Returns:
        Duration in seconds as a float.
    """
    return waveform.shape[0] / sample_rate
