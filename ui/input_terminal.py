# ============================================================
# ui/input_terminal.py
# Real-Time Mock Interview Evaluator
# ============================================================
# Left column UI: Interview prompt display, category selector,
# audio recorder, and analysis trigger button.
# All state is managed through st.session_state for persistence
# across Streamlit reruns.
# ============================================================

import random
from typing import Optional
import streamlit as st
from config import PROMPT_BANK


def render_input_terminal() -> tuple:
    """
    Render the left-column Input Terminal panel.

    Contains:
    1. Section header
    2. Prompt category selector (dropdown)
    3. Active prompt display with refresh button
    4. Audio recorder (st.audio_input — built-in Streamlit)
    5. Analyze button

    Returns:
        Tuple of:
            - audio_bytes (bytes | None): Raw audio from recorder, or None
            - analyze_clicked (bool): True if the Analyze button was pressed
    """

    # ── Section Header ────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="section-header">
            <span style="font-size:1.2rem;">🎙️</span>
            <span class="section-title">Input Terminal</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 1. Interview Category Selector ───────────────────────────────────
    categories = list(PROMPT_BANK.keys())

    selected_category = st.selectbox(
        label="Question Category",
        options=categories,
        index=st.session_state.get("selected_category_idx", 0),
        key="category_selector",
        help="Select the type of interview question you want to practice.",
    )

    # Track category index for persistence
    st.session_state["selected_category_idx"] = categories.index(selected_category)

    # ── 1.5 Job Description Generator ─────────────────────────────────────
    with st.expander("📝 Dynamic JD Question Generator", expanded=False):
        st.markdown(
            '<p style="font-size:0.8rem; color:var(--text-secondary);">'
            'Paste a job description to automatically generate custom behavioral questions.</p>',
            unsafe_allow_html=True
        )
        jd_text = st.text_area("Job Description", height=100, label_visibility="collapsed")
        if st.button("Generate Custom Questions", use_container_width=True):
            if jd_text:
                with st.spinner("🧠 Analyzing JD..."):
                    from core.llm_engine import generate_questions
                    new_qs = generate_questions(jd_text)
                    if new_qs:
                        if "Custom JD" not in PROMPT_BANK:
                            PROMPT_BANK["Custom JD"] = []
                        PROMPT_BANK["Custom JD"].extend(new_qs)
                        
                        # Update session state to select the new category and prompt
                        categories = list(PROMPT_BANK.keys())
                        st.session_state["selected_category_idx"] = categories.index("Custom JD")
                        st.session_state["active_prompt"] = new_qs[0]
                        st.session_state["last_category"] = "Custom JD"
                        st.rerun()

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── 2. Prompt Display ────────────────────────────────────────────────
    st.markdown(
        """
        <div class="section-header">
            <span style="font-size:0.95rem;">📋</span>
            <span class="section-title">Interview Prompt</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize or re-roll prompt when category changes
    current_prompts = PROMPT_BANK.get(selected_category, ["No prompts available."])
    prev_category = st.session_state.get("last_category", None)

    if prev_category != selected_category or "active_prompt" not in st.session_state:
        st.session_state["active_prompt"] = random.choice(current_prompts)
        st.session_state["last_category"] = selected_category

    # Refresh prompt button
    col_prompt, col_refresh = st.columns([5, 1])
    with col_refresh:
        if st.button("🔀", help="Get a new random prompt", key="refresh_prompt"):
            remaining = [p for p in current_prompts if p != st.session_state["active_prompt"]]
            if remaining:
                st.session_state["active_prompt"] = random.choice(remaining)
            else:
                st.session_state["active_prompt"] = random.choice(current_prompts)
            st.rerun()

    active_prompt = st.session_state.get("active_prompt", "Loading prompt...")
    st.markdown(f'<div class="prompt-box">"{active_prompt}"</div>', unsafe_allow_html=True)

    st.markdown(
        f'<p style="font-size:0.72rem; color:var(--text-muted); margin-top:0.3rem;">'
        f'{len(current_prompts)} prompts in this category · Click 🔀 to refresh</p>',
        unsafe_allow_html=True,
    )

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── 3. Audio & Video Inputs ───────────────────────────────────────────
    st.markdown(
        """
        <div class="section-header">
            <span style="font-size:0.95rem;">🎤</span>
            <span class="section-title">Record Your Response</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    audio_file = st.audio_input("Microphone Input", key="audio_recorder")
    audio_bytes: Optional[bytes] = audio_file.read() if audio_file else None

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-header">
            <span style="font-size:0.95rem;">📹</span>
            <span class="section-title">Video Analysis (Optional)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    video_file = st.file_uploader("Upload a video (.mp4) for facial tracking.", type=["mp4", "mov"])
    video_bytes: Optional[bytes] = video_file.read() if video_file else None

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── 4. Analyze Button ────────────────────────────────────────────────
    analyze_clicked = st.button(
        label="⚡ Analyze Response",
        type="primary",
        use_container_width=True,
        key="analyze_btn",
        disabled=(audio_bytes is None), 
    )

    if audio_bytes is None:
        st.markdown(
            '<p style="text-align:center; font-size:0.75rem; color:var(--text-muted); margin-top:0.4rem;">'
            'Record audio first to enable analysis</p>',
            unsafe_allow_html=True,
        )

    return audio_bytes, video_bytes, analyze_clicked
