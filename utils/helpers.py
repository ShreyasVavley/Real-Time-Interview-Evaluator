# ============================================================
# utils/helpers.py
# Real-Time Mock Interview Evaluator
# ============================================================
# Shared utility functions: CSS injection, score normalization,
# overall grade calculation, and session report generation.
# ============================================================

import os
import json
import datetime
import streamlit as st


def inject_css() -> None:
    """
    Read the external CSS file and inject it into the Streamlit page
    using st.markdown with unsafe_allow_html=True.

    The CSS file path is resolved relative to this utils/ directory,
    so it works regardless of where `streamlit run` is called from.
    """
    css_path = os.path.join(
        os.path.dirname(__file__),   # utils/
        "..", "ui", "styles.css"     # → ui/styles.css
    )
    css_path = os.path.normpath(css_path)

    try:
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file not found at: {css_path}")


def compute_overall_grade(
    emotion_rating: str,
    sentiment_label: str,
    fluency_score: float,
    word_count: int,
) -> dict:
    """
    Combine all four analytical dimensions into a single Overall Performance Grade.

    Scoring Rubric (each component contributes 25 points max):
    - Tone/Emotion (25 pts):  positive=25, neutral=18, negative=10, critical=0
    - Sentiment    (25 pts):  Positive=25, Neutral=15, Negative=5
    - Fluency      (25 pts):  Scaled from fluency_score (0–100) → 0–25
    - Length       (25 pts):  Optimal range 80–200 words; penalized outside

    Total: 0–100 → mapped to letter grade A+/A/B/C/D/F

    Args:
        emotion_rating:  "positive" | "neutral" | "negative" | "critical"
        sentiment_label: "Positive" | "Neutral" | "Negative" | "N/A"
        fluency_score:   0.0–100.0 from lexical_analyzer
        word_count:      Integer word count from transcription

    Returns:
        dict with "score", "letter_grade", "color", "summary"
    """
    # ── Tone Score ────────────────────────────────────────────────────────
    tone_map = {"positive": 25, "neutral": 18, "negative": 10, "critical": 0}
    tone_score = tone_map.get(emotion_rating.lower(), 12)

    # ── Sentiment Score ───────────────────────────────────────────────────
    sentiment_map = {"Positive": 25, "Neutral": 15, "Negative": 5, "N/A": 12}
    sentiment_score = sentiment_map.get(sentiment_label, 12)

    # ── Fluency Score (scaled 0–100 → 0–25) ──────────────────────────────
    fluency_component = round((fluency_score / 100.0) * 25, 1)

    # ── Word Count / Length Score ─────────────────────────────────────────
    # Optimal: 80–200 words for a strong interview answer
    if 80 <= word_count <= 200:
        length_score = 25
    elif 50 <= word_count < 80 or 200 < word_count <= 280:
        length_score = 18
    elif 30 <= word_count < 50 or 280 < word_count <= 350:
        length_score = 12
    else:
        length_score = 5   # Too short (<30) or excessively long (>350)

    # ── Aggregate ─────────────────────────────────────────────────────────
    total = round(tone_score + sentiment_score + fluency_component + length_score, 1)
    total = max(0.0, min(100.0, total))   # Clamp to valid range

    # ── Letter Grade Mapping ──────────────────────────────────────────────
    if total >= 90:
        letter, color = "A+", "#00d4a0"
        summary = "Exceptional performance. Your delivery is confident, fluent, and positive."
    elif total >= 80:
        letter, color = "A",  "#00d4ff"
        summary = "Excellent performance. Minor refinements will elevate your delivery further."
    elif total >= 70:
        letter, color = "B",  "#7b5ea7"
        summary = "Good performance. Focus on the flagged areas to reach top-tier delivery."
    elif total >= 60:
        letter, color = "C",  "#f0c420"
        summary = "Average performance. Significant room for improvement in tone and fluency."
    elif total >= 50:
        letter, color = "D",  "#ff8c42"
        summary = "Below average. Practice regularly using the coaching feedback provided."
    else:
        letter, color = "F",  "#ff4757"
        summary = "Needs significant work. Review all coaching points and re-record your answer."

    # Contextual override for very short test recordings
    if word_count < 30:
        summary = f"Your response was too short ({word_count} words). A strong interview answer should be 80-200 words. Please elaborate more."


    return {
        "score":        total,
        "letter_grade": letter,
        "color":        color,
        "summary":      summary,
        "components": {
            "tone":      tone_score,
            "sentiment": sentiment_score,
            "fluency":   fluency_component,
            "length":    length_score,
        },
    }


def format_timestamp() -> str:
    """Return a human-readable timestamp for session history labels."""
    return datetime.datetime.now().strftime("%H:%M:%S")


def truncate_text(text: str, max_chars: int = 80) -> str:
    """Truncate text for compact history card display."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


def safe_html(text: str) -> str:
    """
    Escape HTML special characters in plain text before injecting into
    st.markdown HTML blocks — prevents XSS from transcript content.
    """
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
