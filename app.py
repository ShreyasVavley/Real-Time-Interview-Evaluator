# ============================================================
# app.py — Main Entry Point
# Real-Time Mock Interview Evaluator
# ============================================================
# Orchestrates the full application: layout, model loading,
# analysis pipeline, session state, and history management.
#
# Run with: streamlit run app.py
# ============================================================

import gc
import sys
import logging
import traceback
from typing import Optional
import streamlit as st

# ── Configure logging before any other imports ────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Page config MUST be the first Streamlit call in the script ────────────────
from config import APP_TITLE, APP_ICON, APP_LAYOUT, MAX_HISTORY_ITEMS, WHISPER_MODEL_ID

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=APP_LAYOUT,
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": None,
        "Report a bug": None,
        "About": (
            "**MockIQ — AI Interview Evaluator**\n\n"
            "Powered by Wav2Vec2, Whisper, and TextBlob.\n"
            "All processing is 100% local — no data leaves your machine."
        ),
    },
)

# ── Now safe to import modules that use st.cache_resource ────────────────────
from utils.helpers import inject_css, format_timestamp
from core.audio_processor import process_audio, get_audio_duration
from core.emotion_engine import analyze_emotion, _load_emotion_pipeline
from core.transcription_engine import transcribe, _load_whisper_pipeline
from core.lexical_analyzer import analyze_lexical
from core.sentiment_analyzer import analyze_sentiment
from core.llm_engine import evaluate_content, load_llm_pipeline
from core.grammar_engine import analyze_grammar, load_spacy_model
from core.video_engine import analyze_video, warmup_deepface
from ui.input_terminal import render_input_terminal
from ui.analytics_dashboard import (
    render_dashboard,
    render_empty_dashboard,
    render_session_history,
)


# ─── CSS Injection ────────────────────────────────────────────────────────────
inject_css()


# ─── Session State Initialization ────────────────────────────────────────────
def _init_session_state() -> None:
    """Initialize all session state keys with safe defaults on first run."""
    defaults = {
        "history":               [],     # List of past analysis results
        "last_results":          None,   # Most recent full analysis dict
        "active_prompt":         None,   # Currently displayed interview prompt
        "last_category":         None,   # Category of the active prompt
        "selected_category_idx": 0,      # Dropdown index for persistence
        "models_loaded":         False,  # Track if warm-up is complete
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

_init_session_state()


# ─── Sidebar ─────────────────────────────────────────────────────────────────
def _render_sidebar() -> None:
    """Render the application sidebar with settings and session history."""

    # ── App Branding ──────────────────────────────────────────────────────
    st.sidebar.markdown(
        """
        <div style="text-align:center; padding: 1rem 0 1.5rem;">
            <div style="font-size:2.5rem;">🎙️</div>
            <div style="font-size:1.1rem; font-weight:800;
                        background: linear-gradient(135deg, #00d4ff, #7b5ea7);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                        background-clip: text; letter-spacing:-0.02em;">
                MockIQ
            </div>
            <div style="font-size:0.72rem; color:var(--text-muted); margin-top:0.2rem;">
                AI Interview Evaluator
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")

    # ── Model Status ──────────────────────────────────────────────────────
    st.sidebar.markdown(
        '<p style="font-size:0.7rem; font-weight:700; letter-spacing:0.12em; '
        'text-transform:uppercase; color:var(--text-muted);">Model Status</p>',
        unsafe_allow_html=True,
    )

    if st.session_state["models_loaded"]:
        st.sidebar.markdown(
            """
            <div style="font-size:0.8rem; color:#00d4a0;">✅ Emotion Model — Ready</div>
            <div style="font-size:0.8rem; color:#00d4a0; margin-top:0.25rem;">✅ Whisper STT — Ready</div>
            <div style="font-size:0.8rem; color:#00d4a0; margin-top:0.25rem;">✅ Sentiment Engine — Ready</div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            '<div style="font-size:0.8rem; color:#f0c420;">⏳ Loading models...</div>',
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("---")

    # ── Session History ───────────────────────────────────────────────────
    st.sidebar.markdown(
        '<p style="font-size:0.7rem; font-weight:700; letter-spacing:0.12em; '
        'text-transform:uppercase; color:var(--text-muted);">Session History</p>',
        unsafe_allow_html=True,
    )

    render_session_history(st.session_state["history"])

    # Clear history button
    if st.session_state["history"]:
        st.sidebar.markdown("<br/>", unsafe_allow_html=True)
        if st.sidebar.button("🗑️ Clear History", use_container_width=True, key="clear_history"):
            st.session_state["history"] = []
            st.session_state["last_results"] = None
            st.rerun()

    st.sidebar.markdown("---")

    # ── Info Footer ───────────────────────────────────────────────────────
    st.sidebar.markdown(
        f"""
        <div style="font-size:0.7rem; color:var(--text-muted); text-align:center;
                    line-height:1.7;">
            <div>🔒 100% local — no data sent externally</div>
            <div>🤗 Powered by HuggingFace Transformers</div>
            <div style="margin-top:0.5rem; opacity:0.5;">
                Whisper: {WHISPER_MODEL_ID.split('/')[-1]}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── App Header ───────────────────────────────────────────────────────────────
def _render_header() -> None:
    """Render the top application branding header."""
    st.markdown(
        """
        <div class="app-header">
            <div class="app-logo">🎙️</div>
            <div>
                <h1 style="margin:0;">MockIQ</h1>
                <p class="app-tagline">
                    Real-Time AI Interview Evaluator · Emotion · Fluency · Sentiment · Coaching
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── Model Warm-Up ───────────────────────────────────────────────────────────
def _warmup_models() -> None:
    """
    Pre-load both ML models into the @st.cache_resource cache at app startup.
    This prevents the first analysis from having a long cold-start delay.
    Shows a progress spinner while loading.
    """
    if st.session_state["models_loaded"]:
        return  # Already warmed up — skip

    with st.spinner("🔄 Loading AI models on first run... This may take 1–2 minutes."):
        try:
            _load_emotion_pipeline()
            _load_whisper_pipeline()
            load_llm_pipeline()
            load_spacy_model()
            warmup_deepface()
            st.session_state["models_loaded"] = True
            logger.info("All models loaded and cached successfully.")
        except Exception as e:
            st.error(
                f"⚠️ Model loading failed: {e}\n\n"
                "Ensure your internet connection is active for first-run model download, "
                "then restart the app."
            )
            logger.error(f"Model warm-up failed: {traceback.format_exc()}")


# ─── Core Analysis Pipeline ───────────────────────────────────────────────────
def _run_analysis(audio_bytes: bytes, video_bytes: Optional[bytes] = None) -> Optional[dict]:
    """
    Execute the full 4-engine analysis pipeline on raw audio bytes.

    Pipeline order:
    1. audio_processor  → waveform (shared input for engines 2 & 3)
    2. emotion_engine   → tone/prosody classification
    3. transcription    → Whisper STT → transcript text
    4. lexical_analyzer → filler detection (depends on transcript)
    5. sentiment_analyzer → TextBlob polarity (depends on transcript)

    All intermediate objects are explicitly deleted and gc.collect() is
    called after each step to ensure clean memory management.

    Args:
        audio_bytes: Raw bytes from st.audio_input()

    Returns:
        Combined results dict, or None if any engine fails.
    """
    results = {}

    # ── Progress bar UI ───────────────────────────────────────────────────
    progress_bar = st.progress(0, text="Processing audio...")

    try:
        # ── Step 1: Audio Processing ──────────────────────────────────────
        progress_bar.progress(10, text="🎵 Resampling audio to 16kHz...")
        waveform, sample_rate = process_audio(audio_bytes)
        duration = get_audio_duration(waveform, sample_rate)
        logger.info(f"Audio processed: {duration:.2f}s @ {sample_rate}Hz")

        # ── Step 2: Emotion Analysis (Wav2Vec2) ───────────────────────────
        progress_bar.progress(30, text="🧠 Analyzing vocal tone (Wav2Vec2)...")
        emotion_result = analyze_emotion(waveform)
        logger.info(f"Emotion: {emotion_result['label']} ({emotion_result['score']}%)")

        # ── Step 3: Transcription (Whisper) ───────────────────────────────
        progress_bar.progress(55, text="📝 Transcribing speech (Whisper)...")
        transcript_result = transcribe(waveform)
        logger.info(f"Transcript: '{transcript_result['text'][:80]}...' ({transcript_result['word_count']} words)")

        # ── Free waveform after both audio-dependent engines are done ─────
        del waveform
        gc.collect()

        transcript_text = transcript_result.get("text", "")

        # ── Step 4: Lexical Analysis ──────────────────────────────────────
        progress_bar.progress(65, text="🔤 Detecting filler words...")
        lexical_result = analyze_lexical(transcript_text, duration)
        logger.info(f"Lexical: {lexical_result['total_fillers']} fillers, {lexical_result['filler_density_pct']}% density")

        # ── Step 5: Sentiment Analysis ────────────────────────────────────
        progress_bar.progress(70, text="💬 Analyzing sentiment...")
        sentiment_result = analyze_sentiment(transcript_text)
        logger.info(f"Sentiment: {sentiment_result['label']} ({sentiment_result['polarity']:+.3f})")
        
        # ── Step 6: Grammar Analysis ──────────────────────────────────────
        progress_bar.progress(75, text="📚 Analyzing grammar & vocabulary...")
        grammar_result = analyze_grammar(transcript_text)
        logger.info(f"Grammar: Richness={grammar_result['lexical_richness']}, Passive={grammar_result['passive_voice_count']}")
        
        # ── Step 7: Content Evaluation (LLM) ──────────────────────────────
        progress_bar.progress(85, text="🤖 Grading content (LLM)...")
        active_prompt = st.session_state.get("active_prompt", "")
        content_result = evaluate_content(transcript_text, active_prompt)
        logger.info(f"Content Feedback generated.")
        
        # ── Step 8: Video Analysis (DeepFace) ─────────────────────────────
        if video_bytes:
            progress_bar.progress(95, text="👁️ Analyzing facial expressions...")
            video_result = analyze_video(video_bytes)
            logger.info("Video analysis complete.")
        else:
            video_result = {"smile_pct": 0, "nervous_pct": 0, "coaching": "No video provided for analysis."}

        progress_bar.progress(100, text="✅ Analysis complete!")

        results = {
            "emotion":    emotion_result,
            "transcript": transcript_result,
            "lexical":    lexical_result,
            "sentiment":  sentiment_result,
            "grammar":    grammar_result,
            "content":    content_result,
            "video":      video_result,
            "duration_sec": round(duration, 1),
        }

    except ValueError as ve:
        # User-facing errors (bad audio, too short, etc.)
        st.error(f"⚠️ {ve}")
        logger.warning(f"Analysis validation error: {ve}")
        progress_bar.empty()
        return None

    except RuntimeError as re:
        # Model pipeline errors
        st.error(
            f"🔴 Analysis engine error: {re}\n\n"
            "Try re-recording your audio or restarting the app."
        )
        logger.error(f"Runtime error during analysis: {traceback.format_exc()}")
        progress_bar.empty()
        return None

    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        logger.error(f"Unexpected error: {traceback.format_exc()}")
        progress_bar.empty()
        return None

    finally:
        # Ensure progress bar is cleaned up even on failure
        import time
        time.sleep(0.3)
        progress_bar.empty()
        gc.collect()

    return results


def _save_to_history(results: dict) -> None:
    """
    Append a summarized version of the analysis results to session history.
    Keeps only the last MAX_HISTORY_ITEMS entries.

    Args:
        results: Full analysis results dict from _run_analysis()
    """
    from utils.helpers import compute_overall_grade

    emotion   = results.get("emotion", {})
    sentiment = results.get("sentiment", {})
    lexical   = results.get("lexical", {})
    transcript_data = results.get("transcript", {})

    grade = compute_overall_grade(
        emotion_rating=emotion.get("rating", "neutral"),
        sentiment_label=sentiment.get("label", "Neutral"),
        fluency_score=lexical.get("fluency_score", 50.0),
        word_count=transcript_data.get("word_count", 0),
    )

    history_entry = {
        "timestamp":       format_timestamp(),
        "grade":           grade,
        "emotion":         emotion,
        "sentiment_label": sentiment.get("label", "N/A"),
        "fluency_score":   lexical.get("fluency_score", 0),
        "transcript":      transcript_data.get("text", ""),
        "word_count":      transcript_data.get("word_count", 0),
    }

    st.session_state["history"].append(history_entry)

    # Keep only the last N entries
    if len(st.session_state["history"]) > MAX_HISTORY_ITEMS:
        st.session_state["history"] = st.session_state["history"][-MAX_HISTORY_ITEMS:]


# ─── Main Application ─────────────────────────────────────────────────────────
def main() -> None:
    """
    Main application function. Renders the full page layout and wires up
    all components into a cohesive interactive experience.
    """

    # Render sidebar (always visible)
    _render_sidebar()

    # Render top header
    _render_header()

    # Pre-load models silently on first run
    _warmup_models()

    # ── Two-column layout ─────────────────────────────────────────────────
    # Left (input): 2 units wide | Right (analytics): 3 units wide
    col_left, col_right = st.columns([2, 3], gap="large")

    with col_left:
        st.markdown('<div class="column-label">⬤ INPUT TERMINAL</div>', unsafe_allow_html=True)

        # Render input terminal and capture outputs
        audio_bytes, video_bytes, analyze_clicked = render_input_terminal()

    with col_right:
        st.markdown('<div class="column-label">⬤ ANALYTICS DASHBOARD</div>', unsafe_allow_html=True)

        # ── Handle Analyze button click ───────────────────────────────────
        if analyze_clicked and audio_bytes:
            logger.info("Analysis triggered by user.")

            with st.spinner(""):
                results = _run_analysis(audio_bytes, video_bytes)

            if results:
                # Cache results for re-renders (Streamlit reruns don't re-analyze)
                st.session_state["last_results"] = results
                _save_to_history(results)
                st.rerun()   # Rerun to refresh the sidebar history and clear spinner

        # ── Render dashboard based on available results ───────────────────
        last_results = st.session_state.get("last_results")

        if last_results:
            render_dashboard(last_results)
        else:
            render_empty_dashboard()


# ─── Entry Point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
