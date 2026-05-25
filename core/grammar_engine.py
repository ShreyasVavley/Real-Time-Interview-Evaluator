# ============================================================
# core/grammar_engine.py
# Real-Time Mock Interview Evaluator
# ============================================================
# Evaluates grammatical complexity and vocabulary richness
# using spaCy NLP.
# ============================================================

import logging
import streamlit as st
import spacy

logger = logging.getLogger(__name__)

@st.cache_resource(show_spinner=False)
def load_spacy_model():
    """Load the English spaCy model."""
    logger.info("Loading spaCy grammar model...")
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("spaCy model loaded successfully.")
        return nlp
    except Exception as e:
        logger.error(f"Failed to load spaCy: {e}. Attempting to download...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

def analyze_grammar(transcript: str) -> dict:
    """
    Analyze the transcript for lexical richness (unique vocabulary)
    and grammatical flags like passive voice.
    """
    if not transcript or len(transcript.split()) < 5:
        return {
            "lexical_richness": 0,
            "passive_voice_count": 0,
            "grammar_score": 0,
            "coaching": "Transcript too short to analyze grammar."
        }

    nlp = load_spacy_model()
    doc = nlp(transcript)
    
    # 1. Lexical Richness (Unique lemmas vs total words, excluding stop words)
    words = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    total_content_words = len(words)
    unique_words = len(set(words))
    
    richness_ratio = (unique_words / total_content_words) if total_content_words > 0 else 0
    
    # Scale richness to 0-100 (where 0.7 ratio is 100%)
    richness_score = min(100.0, (richness_ratio / 0.7) * 100)
    
    # 2. Passive Voice Detection
    passive_count = 0
    for token in doc:
        if token.dep_ == "auxpass" or token.dep_ == "csubjpass" or token.dep_ == "nsubjpass":
            passive_count += 1
            
    # Grammar Score (out of 100)
    grammar_score = richness_score - (passive_count * 5)
    grammar_score = max(0.0, min(100.0, grammar_score))
    
    # Coaching
    coaching = []
    if richness_ratio < 0.4:
        coaching.append("Try to vary your vocabulary more. You are repeating the same action words.")
    else:
        coaching.append("Great vocabulary variety!")
        
    if passive_count > 2:
        coaching.append(f"Detected {passive_count} instances of passive voice. Use active, direct statements (e.g., 'I drove the project' instead of 'the project was driven by me').")
        
    return {
        "lexical_richness": round(richness_ratio, 2),
        "passive_voice_count": passive_count,
        "grammar_score": round(grammar_score, 1),
        "coaching": " ".join(coaching)
    }
