# ============================================================
# ui/analytics_dashboard.py
# Real-Time Mock Interview Evaluator
# ============================================================
# Right column UI: Renders all post-analysis metrics, charts,
# transcript with highlighted fillers, and coaching feedback.
# Uses Plotly for animated, interactive charts.
# ============================================================

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from utils.helpers import compute_overall_grade, format_timestamp, truncate_text


# ─── Plotly Theme Shared Config ───────────────────────────────────────────────
_PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#8892a4"),
    margin=dict(l=10, r=10, t=30, b=10),
)


def render_empty_dashboard() -> None:
    """
    Render the analytics panel in its default (empty) state — shown before
    any analysis has been run. Provides a visual placeholder so the layout
    doesn't feel broken on first load.
    """
    st.markdown(
        """
        <div class="section-header">
            <span style="font-size:1.2rem;">📊</span>
            <span class="section-title">Analytics Dashboard</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="glass-card" style="text-align:center; padding: 3rem 2rem;">
            <div style="font-size:3rem; margin-bottom:1rem;">🎙️</div>
            <h2 style="color:var(--text-secondary); font-size:1rem; text-transform:none;
                       letter-spacing:normal; font-weight:500;">
                Record and analyze your response to see your performance metrics here.
            </h2>
            <p style="font-size:0.82rem; color:var(--text-muted); margin-top:0.5rem;">
                Emotion · Sentiment · Fluency · Transcript · Coaching
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard(results: dict) -> None:
    """
    Render the full analytics dashboard with all metrics and charts.

    Args:
        results: Combined dict from all four analysis engines:
            - "emotion":    output of emotion_engine.analyze_emotion()
            - "transcript": output of transcription_engine.transcribe()
            - "lexical":    output of lexical_analyzer.analyze_lexical()
            - "sentiment":  output of sentiment_analyzer.analyze_sentiment()
    """
    emotion   = results.get("emotion", {})
    transcript_data = results.get("transcript", {})
    lexical   = results.get("lexical", {})
    sentiment = results.get("sentiment", {})
    grammar   = results.get("grammar", {})
    content   = results.get("content", {})
    video     = results.get("video", {})

    transcript_text = transcript_data.get("text", "")
    word_count      = transcript_data.get("word_count", 0)
    latency_ms      = transcript_data.get("latency_ms", 0)
    wpm             = lexical.get("wpm", 0.0)

    # ── Overall Grade ─────────────────────────────────────────────────────
    grade = compute_overall_grade(
        emotion_rating=emotion.get("rating", "neutral"),
        sentiment_label=sentiment.get("label", "Neutral"),
        fluency_score=lexical.get("fluency_score", 50.0),
        word_count=word_count,
    )

    st.markdown(
        """
        <div class="section-header">
            <span style="font-size:1.2rem;">📊</span>
            <span class="section-title">Analytics Dashboard</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── PDF Report Download ───────────────────────────────────────────────
    col_grade, col_pdf = st.columns([4, 1])
    with col_pdf:
        from utils.pdf_generator import generate_pdf_report
        pdf_bytes = generate_pdf_report(grade, emotion, sentiment, lexical, content, grammar, video, transcript_text)
        st.download_button(
            label="📄 Download PDF",
            data=pdf_bytes,
            file_name="MockIQ_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    # ── Overall Grade Banner ──────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="glass-card panel-active" style="
            display:flex; align-items:center; justify-content:space-between;
            padding: 1rem 1.5rem; margin-bottom:1rem;">
            <div>
                <div style="font-size:0.68rem; font-weight:700; letter-spacing:0.15em;
                            text-transform:uppercase; color:var(--text-muted);">
                    Overall Performance
                </div>
                <div style="font-size:0.9rem; color:var(--text-secondary); margin-top:0.25rem;">
                    {grade['summary']}
                </div>
            </div>
            <div style="text-align:center; flex-shrink:0; margin-left:1.5rem;">
                <div style="font-size:2.8rem; font-weight:800; color:{grade['color']};
                            line-height:1; letter-spacing:-0.04em;">
                    {grade['letter_grade']}
                </div>
                <div style="font-size:0.72rem; color:{grade['color']}; opacity:0.8;
                            font-weight:600; margin-top:0.2rem;">
                    {grade['score']:.0f} / 100
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 4 Key Metric Cards ────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">{emotion['icon']}</div>
                <div class="metric-value" style="font-size:1.3rem; color:#00d4ff;">
                    {emotion['label']}
                </div>
                <div class="metric-label">Tone</div>
                <div style="font-size:0.72rem; color:var(--text-muted); margin-top:0.3rem;">
                    {emotion['score']}% confidence
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">{sentiment['label_icon']}</div>
                <div class="metric-value" style="font-size:1.3rem; color:{sentiment['label_color']};">
                    {sentiment['label']}
                </div>
                <div class="metric-label">Sentiment</div>
                <div style="font-size:0.72rem; color:var(--text-muted); margin-top:0.3rem;">
                    {sentiment['polarity']:+.3f} polarity
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">🗣️</div>
                <div class="metric-value">{lexical['fluency_score']:.0f}%</div>
                <div class="metric-label">Fluency</div>
                <div style="font-size:0.72rem; color:{lexical['rating_color']}; margin-top:0.3rem;
                            font-weight:600;">
                    {lexical['rating']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">⏱️</div>
                <div class="metric-value">{wpm:.0f}</div>
                <div class="metric-label">WPM</div>
                <div style="font-size:0.72rem; color:var(--text-muted); margin-top:0.3rem;">
                    Words per minute
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Two-column chart row ──────────────────────────────────────────────
    chart_left, chart_right = st.columns([1.2, 1])

    with chart_left:
        _render_filler_chart(lexical)

    with chart_right:
        _render_sentiment_gauge(sentiment)

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Emotion Breakdown Bar ─────────────────────────────────────────────
    _render_emotion_breakdown(emotion)

    st.markdown("<br/>", unsafe_allow_html=True)
    
    # ── Secondary Metrics (Grammar & Video) ───────────────────────────────
    st.markdown(
        '<div class="section-header">'
        '<span style="font-size:0.9rem;">📚</span>'
        '<span class="section-title">Additional Insights</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Grammar Score**: {grammar.get('grammar_score', 0):.1f}/100")
        st.markdown(f"**Lexical Richness**: {grammar.get('lexical_richness', 0):.2f}")
        st.markdown(f"**Passive Voice**: {grammar.get('passive_voice_count', 0)} instances")
        
    with col2:
        if video.get('smile_pct') is not None:
            st.markdown(f"**Smile Time**: {video.get('smile_pct', 0):.1f}%")
            st.markdown(f"**Nervous Expressions**: {video.get('nervous_pct', 0):.1f}%")
        else:
            st.markdown("No video analysis provided.")
            
    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Transcript ────────────────────────────────────────────────────────
    _render_transcript(lexical, transcript_text)

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Content Coaching ──────────────────────────────────────────────────
    if content.get("content_feedback"):
        st.info(f"**🤖 Content Evaluation (Did you answer the prompt?):**\n\n{content['content_feedback']}")
        st.markdown("<br/>", unsafe_allow_html=True)

    # ── Coaching Feedback ─────────────────────────────────────────────────
    _render_coaching(emotion, sentiment, lexical, grade, grammar, video)

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Session Info Footer ───────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="text-align:right; font-size:0.7rem; color:var(--text-muted); padding-top:0.5rem;
                    border-top: 1px solid var(--border-subtle); margin-top:0.5rem;">
            Analysis completed · {format_timestamp()} ·
            Grade components: Tone {grade['components']['tone']}/25 ·
            Sentiment {grade['components']['sentiment']}/25 ·
            Fluency {grade['components']['fluency']:.0f}/25 ·
            Length {grade['components']['length']}/25
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── Chart Renderers ─────────────────────────────────────────────────────────

def _render_filler_chart(lexical: dict) -> None:
    """Render horizontal bar chart of filler word counts using Plotly."""
    st.markdown(
        '<div class="section-header">'
        '<span style="font-size:0.9rem;">🔤</span>'
        '<span class="section-title">Filler Word Breakdown</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    filler_counts = lexical.get("filler_counts", {})

    if not filler_counts:
        st.markdown(
            '<div style="text-align:center; padding:1.5rem; color:var(--accent-green); '
            'font-weight:600; font-size:0.9rem;">🎉 Zero filler words detected!</div>',
            unsafe_allow_html=True,
        )
        return

    # Sort by count descending
    sorted_items = sorted(filler_counts.items(), key=lambda x: x[1], reverse=True)
    labels  = [item[0] for item in sorted_items]
    counts  = [item[1] for item in sorted_items]

    # Color bars based on count severity
    bar_colors = []
    for c in counts:
        if c <= 1:
            bar_colors.append("#00d4a0")    # Green — minor
        elif c <= 3:
            bar_colors.append("#f0c420")    # Amber — moderate
        else:
            bar_colors.append("#ff4757")    # Red — heavy

    fig = go.Figure(go.Bar(
        x=counts,
        y=labels,
        orientation="h",
        marker=dict(
            color=bar_colors,
            line=dict(width=0),
            opacity=0.85,
        ),
        text=[str(c) for c in counts],
        textposition="outside",
        textfont=dict(color="#f0f4ff", size=12, family="Inter"),
    ))

    fig.update_layout(
        **_PLOTLY_LAYOUT,
        height=max(160, len(labels) * 42),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(color="#f0f4ff", size=12, family="Inter"),
        ),
        bargap=0.35,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Density summary
    density = lexical.get("filler_density_pct", 0)
    total_f = lexical.get("total_fillers", 0)
    rating_color = lexical.get("rating_color", "#888")
    st.markdown(
        f'<p style="font-size:0.78rem; color:var(--text-secondary);">'
        f'<strong style="color:{rating_color};">{total_f} fillers</strong> detected · '
        f'<strong style="color:{rating_color};">{density}%</strong> density</p>',
        unsafe_allow_html=True,
    )


def _render_sentiment_gauge(sentiment: dict) -> None:
    """Render animated Plotly gauge for sentiment polarity."""
    st.markdown(
        '<div class="section-header">'
        '<span style="font-size:0.9rem;">🎯</span>'
        '<span class="section-title">Sentiment Gauge</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    polarity_pct = sentiment.get("polarity_pct", 50.0)
    polarity     = sentiment.get("polarity", 0.0)
    label        = sentiment.get("label", "Neutral")
    label_color  = sentiment.get("label_color", "#f0c420")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=polarity_pct,
        number=dict(
            suffix="%",
            font=dict(color=label_color, size=22, family="Inter"),
        ),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickwidth=1,
                tickcolor="rgba(255,255,255,0.1)",
                tickfont=dict(color="#4a5568", size=10),
                tickvals=[0, 25, 50, 75, 100],
                ticktext=["−1.0", "−0.5", "0.0", "+0.5", "+1.0"],
            ),
            bar=dict(
                color=label_color,
                thickness=0.25,
            ),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0, 35],  color="rgba(255, 71, 87, 0.15)"),
                dict(range=[35, 65], color="rgba(240,196, 32, 0.10)"),
                dict(range=[65, 100],color="rgba(0, 212, 160, 0.15)"),
            ],
            threshold=dict(
                line=dict(color=label_color, width=3),
                thickness=0.8,
                value=polarity_pct,
            ),
        ),
        title=dict(
            text=f"{label}  ({polarity:+.3f})",
            font=dict(color="#8892a4", size=12, family="Inter"),
        ),
    ))

    fig.update_layout(
        **_PLOTLY_LAYOUT,
        height=200,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_emotion_breakdown(emotion: dict) -> None:
    """Render horizontal bar chart for all emotion scores."""
    all_scores = emotion.get("all_scores", [])
    if not all_scores:
        return

    st.markdown(
        '<div class="section-header">'
        '<span style="font-size:0.9rem;">🧠</span>'
        '<span class="section-title">Emotion Probability Distribution</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    labels  = [item[0] for item in all_scores]
    scores  = [item[1] for item in all_scores]

    # Gradient from accent-primary to accent-secondary
    bar_colors = [f"rgba(0, 212, 255, {max(0.2, s/100):.2f})" for s in scores]
    # Highlight the top emotion
    bar_colors[0] = "#00d4ff"

    fig = go.Figure(go.Bar(
        x=scores,
        y=labels,
        orientation="h",
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[f"{s:.1f}%" for s in scores],
        textposition="outside",
        textfont=dict(color="#8892a4", size=10, family="Inter"),
    ))

    fig.update_layout(
        **_PLOTLY_LAYOUT,
        height=max(140, len(labels) * 30),
        xaxis=dict(
            range=[0, 115],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(color="#f0f4ff", size=11, family="Inter"),
        ),
        bargap=0.3,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_transcript(lexical: dict, transcript_text: str) -> None:
    """Render the transcript with filler words highlighted in red."""
    st.markdown(
        '<div class="section-header">'
        '<span style="font-size:0.9rem;">📝</span>'
        '<span class="section-title">Transcript</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    highlighted_html = lexical.get("highlighted_html", "")

    if not highlighted_html and not transcript_text:
        st.markdown(
            '<div class="transcript-box" style="color:var(--text-muted);">'
            'No transcript available.</div>',
            unsafe_allow_html=True,
        )
        return

    # Use highlighted HTML if available, otherwise plain text
    display_html = highlighted_html if highlighted_html else transcript_text

    st.markdown(
        f'<div class="transcript-box">{display_html}</div>',
        unsafe_allow_html=True,
    )

    if lexical.get("total_fillers", 0) > 0:
        st.markdown(
            '<p style="font-size:0.75rem; color:var(--text-muted); margin-top:0.4rem;">'
            '<span style="color:#ff4757; font-weight:700;">Red</span> '
            '= filler word detected</p>',
            unsafe_allow_html=True,
        )


def _render_coaching(emotion: dict, sentiment: dict, lexical: dict, grade: dict, grammar: dict, video: dict) -> None:
    """Render the consolidated AI Coaching Feedback panel."""
    st.markdown(
        '<div class="section-header">'
        '<span style="font-size:0.9rem;">💡</span>'
        '<span class="section-title">AI Coaching Feedback</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    coaching_items = [
        {
            "icon": emotion.get("icon", "🎙️"),
            "label": "Tone & Delivery",
            "text": emotion.get("coaching", ""),
        },
        {
            "icon": sentiment.get("label_icon", "🟡"),
            "label": "Language Sentiment",
            "text": sentiment.get("coaching", ""),
        },
        {
            "icon": "🗣️",
            "label": "Speech Fluency",
            "text": lexical.get("coaching", ""),
        },
        {
            "icon": "📝",
            "label": "Grammar & Vocab",
            "text": grammar.get("coaching", ""),
        },
        {
            "icon": "📹",
            "label": "Body Language",
            "text": video.get("coaching", ""),
        },
        {
            "icon": "🏆",
            "label": "Overall Performance",
            "text": grade.get("summary", ""),
        },
    ]

    items_html = ""
    for item in coaching_items:
        if item["text"]:
            items_html += f"""
<div class="coaching-item">
    <div class="coaching-icon">{item['icon']}</div>
    <div>
        <div style="font-size:0.7rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:var(--text-muted); margin-bottom:0.2rem;">
            {item['label']}
        </div>
        <div style="font-size:0.86rem; color:var(--text-primary);">
            {item['text']}
        </div>
    </div>
</div>"""

    st.markdown(
        f'<div class="coaching-panel">{items_html}</div>',
        unsafe_allow_html=True,
    )


# ─── Session History Sidebar ─────────────────────────────────────────────────

def render_session_history(history: list) -> None:
    """
    Render past analysis sessions in the sidebar for quick comparison.

    Args:
        history: List of dicts, each containing a past analysis result.
                 Stored in st.session_state["history"].
    """
    if not history:
        st.sidebar.markdown(
            '<p style="font-size:0.8rem; color:var(--text-muted); text-align:center;">'
            'No sessions yet. Analyze a response to see history here.</p>',
            unsafe_allow_html=True,
        )
        return

    for i, session in enumerate(reversed(history)):
        grade   = session.get("grade", {})
        emotion = session.get("emotion", {})
        ts      = session.get("timestamp", "—")
        snippet = truncate_text(session.get("transcript", ""), 60)

        st.sidebar.markdown(
            f"""
            <div class="history-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-weight:700; color:{grade.get('color','#888')};">
                        {grade.get('letter_grade','?')} · {grade.get('score',0):.0f}pts
                    </span>
                    <span style="font-size:0.68rem; color:var(--text-muted);">
                        {ts}
                    </span>
                </div>
                <div style="font-size:0.78rem; color:var(--text-secondary); margin-top:0.3rem;">
                    {emotion.get('icon','🎙️')} {emotion.get('label','—')} ·
                    {session.get('sentiment_label','—')} sentiment
                </div>
                <div style="font-size:0.75rem; color:var(--text-muted); margin-top:0.25rem;
                            font-style:italic; line-height:1.4;">
                    "{snippet}"
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
