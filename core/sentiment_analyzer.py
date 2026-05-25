# ============================================================
# core/sentiment_analyzer.py
# Real-Time Mock Interview Evaluator
# ============================================================
# TextBlob-based sentiment polarity and subjectivity scoring.
# Lightweight, no model download required — runs instantly.
# ============================================================

import logging
from textblob import TextBlob
from config import SENTIMENT_POSITIVE_THRESHOLD, SENTIMENT_NEGATIVE_THRESHOLD

logger = logging.getLogger(__name__)


def analyze_sentiment(transcript: str) -> dict:
    """
    Extract sentiment polarity and subjectivity from the Whisper transcript
    using TextBlob's pattern-based sentiment analyzer.

    TextBlob Sentiment:
    - polarity:     float in [-1.0, +1.0]
                    -1.0 = very negative, 0.0 = neutral, +1.0 = very positive
    - subjectivity: float in [0.0, 1.0]
                    0.0 = very objective, 1.0 = very subjective/opinionated

    Interview Context Interpretation:
    - A high positive polarity (>0.3) signals enthusiasm and confidence.
    - Near-zero polarity is neutral — acceptable but may lack energy.
    - Negative polarity flags language that may undermine the candidate.
    - High subjectivity is acceptable in behavioral answers (personal stories),
      but low subjectivity is preferred for technical/factual questions.

    Args:
        transcript: Plain text string from the transcription engine.

    Returns:
        dict with keys:
            - "polarity"          (float): Raw polarity score [-1.0 → +1.0]
            - "subjectivity"      (float): Raw subjectivity score [0.0 → 1.0]
            - "polarity_pct"      (float): Polarity normalized to [0 → 100] for gauge
            - "label"             (str):   "Positive" | "Neutral" | "Negative"
            - "label_color"       (str):   Hex color for the rating badge
            - "label_icon"        (str):   Emoji indicator
            - "coaching"          (str):   Actionable coaching message
            - "sentence_scores"   (list):  Per-sentence polarity breakdown
    """
    if not transcript or not transcript.strip():
        return _empty_result()

    blob = TextBlob(transcript)
    polarity     = round(blob.sentiment.polarity, 4)
    subjectivity = round(blob.sentiment.subjectivity, 4)

    # ── Normalize polarity to 0–100 scale for Plotly gauge ───────────────
    # polarity is in [-1, 1] → map to [0, 100]
    polarity_pct = round((polarity + 1.0) / 2.0 * 100, 1)

    # ── Classify and get coaching ─────────────────────────────────────────
    label, label_color, label_icon, coaching = _classify_sentiment(
        polarity, subjectivity
    )

    # ── Per-sentence breakdown (for detailed view) ────────────────────────
    sentence_scores = []
    for sentence in blob.sentences:
        sentence_scores.append({
            "text":     str(sentence),
            "polarity": round(sentence.sentiment.polarity, 3),
        })

    logger.info(
        f"Sentiment: polarity={polarity}, subjectivity={subjectivity}, label={label}"
    )

    return {
        "polarity":        polarity,
        "subjectivity":    subjectivity,
        "polarity_pct":    polarity_pct,
        "label":           label,
        "label_color":     label_color,
        "label_icon":      label_icon,
        "coaching":        coaching,
        "sentence_scores": sentence_scores,
    }


# ─── Internal Helpers ─────────────────────────────────────────────────────────

def _classify_sentiment(polarity: float, subjectivity: float) -> tuple:
    """
    Map raw TextBlob polarity and subjectivity to interview coaching labels.

    Returns:
        Tuple of (label, color_hex, icon, coaching_message)
    """
    # ── Polarity classification ───────────────────────────────────────────
    if polarity > SENTIMENT_POSITIVE_THRESHOLD:
        label      = "Positive"
        label_color = "#00d4a0"   # Teal-green
        label_icon  = "🟢"
        base_coaching = (
            "Your language conveys confidence and positivity — excellent for interviews! "
            "Continue using achievement-focused and solution-oriented vocabulary."
        )
    elif polarity < SENTIMENT_NEGATIVE_THRESHOLD:
        label       = "Negative"
        label_color = "#ff4757"   # Red
        label_icon  = "🔴"
        base_coaching = (
            "Negative language detected. Replace words like 'failed', 'couldn't', 'problem' "
            "with 'learned', 'opportunity', 'challenge'. Reframe past experiences positively."
        )
    else:
        label       = "Neutral"
        label_color = "#f0c420"   # Amber
        label_icon  = "🟡"
        base_coaching = (
            "Neutral tone detected. While professional, try adding more positive language "
            "to convey enthusiasm. Use impact statements like 'I achieved...', 'I drove...'."
        )

    # ── Subjectivity addendum ─────────────────────────────────────────────
    # High subjectivity = very opinionated — can feel less credible in technical contexts.
    if subjectivity > 0.7:
        base_coaching += (
            " Note: Your response is highly personal/opinionated. For technical questions, "
            "balance with objective data and measurable outcomes."
        )

    return label, label_color, label_icon, base_coaching


def _empty_result() -> dict:
    """Return a safe empty result when transcript is blank."""
    return {
        "polarity":        0.0,
        "subjectivity":    0.0,
        "polarity_pct":    50.0,
        "label":           "N/A",
        "label_color":     "#888888",
        "label_icon":      "⚪",
        "coaching":        "No transcript available for sentiment analysis.",
        "sentence_scores": [],
    }
