# ============================================================
# config.py — Global Configuration & Constants
# Real-Time Mock Interview Evaluator
# ============================================================
# All magic strings, model identifiers, and thresholds live
# here. To upgrade a model, change only this file.

# ─── Model Identifiers ──────────────────────────────────────
EMOTION_MODEL_ID = "superb/wav2vec2-base-superb-er"  # Public, verified — 4 emotion classes: neu/hap/ang/sad
WHISPER_MODEL_ID = "openai/whisper-tiny"   # Options: whisper-base, whisper-small
LLM_MODEL_ID = "Qwen/Qwen1.5-0.5B-Chat"

# ─── Audio Processing Constants ─────────────────────────────
TARGET_SAMPLE_RATE = 16_000          # Hz — required by Wav2Vec2 & Whisper
MAX_AUDIO_DURATION_SEC = 120         # Hard cap per recording session
MIN_AUDIO_DURATION_SEC = 1           # Minimum to avoid pipeline errors

# ─── Lexical Analytics ──────────────────────────────────────
FILLER_WORDS = [
    "uh", "um", "ah", "like", "basically", "actually",
    "you know", "i mean", "sort of", "kind of", "right", "okay"
]

# Filler density thresholds (percentage)
FILLER_DENSITY_EXCELLENT = 5.0    # < 5%  → Excellent
FILLER_DENSITY_GOOD      = 10.0   # 5–10% → Good
FILLER_DENSITY_NEEDS_WORK = 20.0  # 10–20%→ Needs Work
                                   # > 20% → Critical

# ─── Sentiment Thresholds (TextBlob polarity) ────────────────
SENTIMENT_POSITIVE_THRESHOLD  =  0.15   # > 0.15 → Positive
SENTIMENT_NEGATIVE_THRESHOLD  = -0.15   # < -0.15 → Negative
                                         # else  → Neutral

# ─── Emotion Label Mapping ──────────────────────────────────
# Maps raw HuggingFace model output labels to coaching context.
# Adjust if using a different emotion model with different labels.
EMOTION_LABEL_MAP = {
    # Positive signals
    "hap":     {"icon": "😊", "rating": "positive", "coaching": "Great energy! Conveys enthusiasm and confidence."},
    "happy":   {"icon": "😊", "rating": "positive", "coaching": "Great energy! Conveys enthusiasm and confidence."},
    "surprised": {"icon": "😲", "rating": "positive", "coaching": "Engaged tone detected — keep it authentic."},
    # Neutral signals
    "neu":     {"icon": "😐", "rating": "neutral",  "coaching": "Calm delivery is good. Add vocal variation to show passion."},
    "neutral": {"icon": "😐", "rating": "neutral",  "coaching": "Calm delivery is good. Add vocal variation to show passion."},
    # Negative signals
    "sad":     {"icon": "😢", "rating": "negative", "coaching": "Voice signals low confidence — try power posing before recording."},
    "fearful": {"icon": "😨", "rating": "negative", "coaching": "Detected anxiety in tone — slow your breathing and pause more."},
    # Critical signals
    "ang":     {"icon": "😠", "rating": "critical",  "coaching": "Aggressive tone detected — soften vocal delivery immediately."},
    "angry":   {"icon": "😠", "rating": "critical",  "coaching": "Aggressive tone detected — soften vocal delivery immediately."},
    "disgusted": {"icon": "🤢", "rating": "critical", "coaching": "Negative tone may alienate interviewers — reframe your language."},
}
# Fallback for labels not in the map
EMOTION_LABEL_FALLBACK = {"icon": "🎙️", "rating": "neutral", "coaching": "Unable to classify tone — ensure clear audio."}

# ─── Interview Prompt Bank ──────────────────────────────────
PROMPT_BANK = {
    "Behavioral": [
        "Tell me about a time you faced a significant challenge at work and how you overcame it.",
        "Describe a situation where you had to work with a difficult team member. What was the outcome?",
        "Give an example of a goal you set and how you achieved it.",
        "Tell me about a time you failed. What did you learn from that experience?",
        "Describe a moment when you had to make a tough decision with limited information.",
        "Tell me about a time you demonstrated leadership without having a formal title.",
    ],
    "Technical": [
        "Walk me through a complex system or project you designed from scratch.",
        "Describe your approach to debugging a production issue under time pressure.",
        "Explain a technical concept you recently learned and how you applied it.",
        "How do you approach code reviews and maintaining code quality in a team?",
        "Describe your experience with designing scalable, distributed systems.",
        "How do you balance technical debt with shipping features on time?",
    ],
    "HR / Culture Fit": [
        "Where do you see yourself professionally in the next 3 to 5 years?",
        "Why are you leaving your current role, and what are you looking for in your next opportunity?",
        "What is your greatest professional strength and how does it add value?",
        "What is an area you are actively working to improve, and how are you doing it?",
        "How do you handle working under tight deadlines or high-pressure environments?",
        "Describe your ideal team culture and working environment.",
    ],
    "Leadership": [
        "Describe your philosophy on giving and receiving constructive feedback.",
        "Tell me about a time you had to influence stakeholders without direct authority.",
        "How do you prioritize tasks when everything feels equally urgent?",
        "Describe how you have mentored or developed someone on your team.",
        "Tell me about a strategic initiative you drove and the impact it created.",
        "How do you align a team around a shared goal when there is disagreement?",
    ],
}

# ─── App Visual Config ───────────────────────────────────────
APP_TITLE         = "MockIQ — AI Interview Evaluator"
APP_ICON          = "🎙️"
APP_LAYOUT        = "wide"
MAX_HISTORY_ITEMS = 3   # Number of past sessions to keep in st.session_state
