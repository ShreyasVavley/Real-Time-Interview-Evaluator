# ============================================================
# core/lexical_analyzer.py
# Real-Time Mock Interview Evaluator
# ============================================================
# Custom filler word detection and fluency scoring algorithm.
# Uses lightweight regex tokenization — no heavy NLP dependency.
# Designed for sub-millisecond execution on transcript strings.
# ============================================================

import re
import logging
from config import FILLER_WORDS, FILLER_DENSITY_EXCELLENT, FILLER_DENSITY_GOOD, FILLER_DENSITY_NEEDS_WORK

logger = logging.getLogger(__name__)


def analyze_lexical(transcript: str, duration_sec: float = 0.0) -> dict:
    """
    Parse the Whisper transcript to count filler words, calculate density,
    and assign a fluency score with actionable coaching feedback.

    Algorithm:
    1. Lowercase the entire transcript for case-insensitive matching.
    2. Handle multi-word fillers (e.g., "you know", "i mean") first, before
       single-word tokenization, to prevent double-counting.
    3. Tokenize remaining text using regex word boundaries.
    4. Cross-reference each token against the FILLER_WORDS list.
    5. Calculate density as: (total_fillers / total_words) * 100.
    6. Fluency score: 100 - density, clamped to [0, 100].

    Args:
        transcript: Plain text string from the transcription engine.

    Returns:
        dict with keys:
            - "total_words"       (int):   Total word count in transcript
            - "filler_counts"     (dict):  Per-filler-word occurrence counts
            - "total_fillers"     (int):   Sum of all filler occurrences
            - "filler_density_pct"(float): Fillers as % of total words
            - "fluency_score"     (float): 0–100 fluency rating
            - "rating"            (str):   "Excellent" | "Good" | "Needs Work" | "Critical"
            - "rating_color"      (str):   Hex color for the UI badge
            - "coaching"          (str):   Actionable coaching message
            - "highlighted_html"  (str):   Transcript HTML with fillers marked in red
    """
    if not transcript or not transcript.strip():
        return _empty_result()

    text_lower = transcript.lower().strip()

    # ── Step 1: Initialize filler count dict ─────────────────────────────
    filler_counts = {word: 0 for word in FILLER_WORDS}

    # ── Step 2: Detect multi-word fillers first ───────────────────────────
    # e.g., "you know", "i mean", "sort of", "kind of"
    # We scan the lowercased text and count non-overlapping occurrences.
    multi_word_fillers = [f for f in FILLER_WORDS if " " in f]
    working_text = text_lower  # We'll strip multi-word matches before tokenizing

    for phrase in multi_word_fillers:
        # Use regex to find whole-phrase matches (not partial word matches)
        pattern = r'\b' + re.escape(phrase) + r'\b'
        matches = re.findall(pattern, working_text)
        filler_counts[phrase] = len(matches)
        # Replace matches with placeholder to prevent double-counting during tokenization
        working_text = re.sub(pattern, "__FILLER__", working_text)

    # ── Step 3: Tokenize remaining text ──────────────────────────────────
    # Extract all word tokens (alphabetic only) from the de-multi-worded text.
    tokens = re.findall(r'\b[a-z]+\b', working_text)
    # Filter out our placeholder tokens
    tokens = [t for t in tokens if t != "__filler__"]

    # ── Step 4: Count single-word fillers ────────────────────────────────
    single_word_fillers = [f for f in FILLER_WORDS if " " not in f]
    for token in tokens:
        if token in single_word_fillers:
            filler_counts[token] = filler_counts.get(token, 0) + 1

    # ── Step 5: Compute aggregate metrics ────────────────────────────────
    # Total word count = single-word tokens + (multi-word filler phrase counts × phrase length)
    multi_word_total = sum(
        filler_counts[f] * len(f.split()) for f in multi_word_fillers
    )
    total_words = len(tokens) + multi_word_total

    # Remove fillers with zero count for cleaner output
    filler_counts_nonzero = {k: v for k, v in filler_counts.items() if v > 0}

    total_fillers = sum(filler_counts.values())

    # Guard against division by zero for very short transcripts
    if total_words == 0:
        return _empty_result()

    filler_density_pct = round((total_fillers / total_words) * 100, 2)

    # Fluency score: inverse of density, clamped [0, 100]
    fluency_score = round(max(0.0, min(100.0, 100.0 - filler_density_pct)), 1)

    # ── Step 6: Assign rating based on density thresholds ────────────────
    rating, rating_color, coaching = _get_rating(filler_density_pct)

    # ── Step 7: Generate highlighted HTML for transcript display ─────────
    highlighted_html = _highlight_fillers(transcript)

    wpm = 0.0
    if duration_sec > 0:
        wpm = round((total_words / duration_sec) * 60, 1)

    logger.info(
        f"Lexical analysis: {total_words} words, {total_fillers} fillers "
        f"({filler_density_pct}%), fluency={fluency_score}, wpm={wpm}"
    )

    return {
        "total_words":        total_words,
        "filler_counts":      filler_counts_nonzero,
        "total_fillers":      total_fillers,
        "filler_density_pct": filler_density_pct,
        "fluency_score":      fluency_score,
        "rating":             rating,
        "rating_color":       rating_color,
        "coaching":           coaching,
        "highlighted_html":   highlighted_html,
        "wpm":                wpm,
    }


# ─── Internal Helpers ─────────────────────────────────────────────────────────

def _get_rating(density: float) -> tuple[str, str, str]:
    """Map density percentage to rating label, badge color, and coaching text."""
    if density < FILLER_DENSITY_EXCELLENT:
        return (
            "Excellent",
            "#00d4a0",   # Teal green
            "Outstanding fluency! Your speech flows naturally with minimal hesitation. "
            "Continue practicing deliberate pauses instead of filler words.",
        )
    elif density < FILLER_DENSITY_GOOD:
        return (
            "Good",
            "#f0c420",   # Amber
            "Minor filler usage detected. Practice replacing fillers with a brief, "
            "confident pause (1–2 seconds). Silence is more powerful than 'um'.",
        )
    elif density < FILLER_DENSITY_NEEDS_WORK:
        return (
            "Needs Work",
            "#ff8c42",   # Orange
            "Noticeable filler words are interrupting your message. Try the 'pause drill': "
            "record yourself and replace every filler with a silent pause until it feels natural.",
        )
    else:
        return (
            "Critical",
            "#ff4757",   # Red
            "Heavy filler usage is significantly undermining your credibility. "
            "Focus on slowing your speech rate by 20% — rushing triggers filler words. "
            "Practice one answer per day with zero fillers allowed.",
        )


def _highlight_fillers(text: str) -> str:
    """
    Return the transcript with filler words wrapped in HTML spans for
    red-highlighted display in the Streamlit UI.

    Uses case-insensitive regex replacement to preserve original casing.
    """
    highlighted = text

    # Sort fillers by length descending to match longer phrases first
    sorted_fillers = sorted(FILLER_WORDS, key=len, reverse=True)

    for filler in sorted_fillers:
        pattern = r'(?<!\w)' + re.escape(filler) + r'(?!\w)'
        replacement = (
            f'<span style="color:#ff4757; font-weight:700; '
            f'background:rgba(255,71,87,0.15); border-radius:3px; '
            f'padding:1px 4px;">{filler}</span>'
        )
        highlighted = re.sub(pattern, replacement, highlighted, flags=re.IGNORECASE)

    return highlighted


def _empty_result() -> dict:
    """Return a safe empty result when transcript is blank."""
    return {
        "total_words":        0,
        "filler_counts":      {},
        "total_fillers":      0,
        "filler_density_pct": 0.0,
        "fluency_score":      100.0,
        "rating":             "N/A",
        "rating_color":       "#888888",
        "coaching":           "No transcript detected. Please record audio and try again.",
        "highlighted_html":   "",
        "wpm":                0.0,
    }
