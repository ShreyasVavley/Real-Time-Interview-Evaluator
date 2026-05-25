# ============================================================
# core/video_engine.py
# Real-Time Mock Interview Evaluator
# ============================================================
# Evaluates facial expressions and body language from an
# uploaded video file using OpenCV and DeepFace.
# ============================================================

import os
import cv2
import logging
import tempfile
import streamlit as st
from deepface import DeepFace

logger = logging.getLogger(__name__)

@st.cache_resource(show_spinner=False)
def warmup_deepface():
    """Warmup DeepFace by analyzing a dummy array so models download."""
    logger.info("Warming up DeepFace models...")
    try:
        import numpy as np
        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
        DeepFace.analyze(dummy_img, actions=['emotion'], enforce_detection=False)
        logger.info("DeepFace warmed up.")
    except Exception as e:
        logger.warning(f"DeepFace warmup skipped: {e}")

def analyze_video(video_bytes: bytes) -> dict:
    """
    Extracts frames from the video (1 frame per second) and 
    analyzes facial expressions using DeepFace.
    """
    if not video_bytes:
        return {"error": "No video provided"}
        
    # Write video bytes to a temporary file
    temp_fd, temp_path = tempfile.mkstemp(suffix=".mp4")
    try:
        with os.fdopen(temp_fd, 'wb') as f:
            f.write(video_bytes)
            
        cap = cv2.VideoCapture(temp_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30
        
        frame_interval = int(fps * 1.5) # Extract 1 frame every 1.5 seconds
        
        frame_count = 0
        emotions_tracked = {"happy": 0, "neutral": 0, "nervous": 0, "total": 0}
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                try:
                    # Analyze frame
                    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                    # result is a list of dicts if multiple faces, take the first one
                    if isinstance(result, list):
                        result = result[0]
                    
                    dom_emotion = result.get('dominant_emotion', 'neutral')
                    
                    emotions_tracked["total"] += 1
                    if dom_emotion == "happy":
                        emotions_tracked["happy"] += 1
                    elif dom_emotion in ["fear", "sad", "angry"]:
                        emotions_tracked["nervous"] += 1
                    else:
                        emotions_tracked["neutral"] += 1
                        
                except Exception as e:
                    pass # Ignore frames where face isn't detected
                    
            frame_count += 1
            
            # Cap at 20 frames to avoid super long processing times on CPU
            if emotions_tracked["total"] >= 20:
                break
                
        cap.release()
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    total = emotions_tracked["total"]
    if total == 0:
        return {
            "smile_pct": 0,
            "nervous_pct": 0,
            "coaching": "Could not detect a clear face in the video. Make sure your lighting is good."
        }
        
    smile_pct = (emotions_tracked["happy"] / total) * 100
    nervous_pct = (emotions_tracked["nervous"] / total) * 100
    
    coaching = []
    if smile_pct < 15:
        coaching.append("You maintained a very stern expression. Try to smile warmly during introductions to build rapport.")
    elif smile_pct > 60:
        coaching.append("Excellent positive body language! Your frequent smiling conveys enthusiasm.")
    else:
        coaching.append("Good professional expression. Remember to smile when talking about your achievements.")
        
    if nervous_pct > 30:
        coaching.append("Your micro-expressions indicated some tension or anxiety. Remember to take deep breaths.")
        
    return {
        "smile_pct": round(smile_pct, 1),
        "nervous_pct": round(nervous_pct, 1),
        "coaching": " ".join(coaching)
    }
