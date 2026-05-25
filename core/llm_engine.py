# ============================================================
# core/llm_engine.py
# Real-Time Mock Interview Evaluator
# ============================================================
# Uses a lightweight local LLM to evaluate answer content
# and generate dynamic interview questions based on JDs.
# ============================================================

import logging
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from config import LLM_MODEL_ID

logger = logging.getLogger(__name__)

@st.cache_resource(show_spinner=False)
def load_llm_pipeline():
    """Load a lightweight local LLM pipeline on CPU."""
    logger.info(f"Loading Generative LLM [{LLM_MODEL_ID}] on CPU...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        device_map="cpu",
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    
    # Create the text-generation pipeline
    llm_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    logger.info("LLM loaded successfully.")
    return llm_pipe

def evaluate_content(transcript: str, prompt: str) -> dict:
    """
    Evaluate if the user actually answered the interview prompt.
    Uses the local LLM to generate a short critique.
    """
    if not transcript or len(transcript.split()) < 10:
        return {"content_feedback": "Response too short to evaluate content."}

    llm_pipe = load_llm_pipeline()
    
    # Construct prompt
    system_prompt = "You are an expert interview coach. Be concise and direct."
    user_message = f"Prompt: {prompt}\nCandidate Answer: {transcript}\nDid the candidate successfully answer the prompt using the STAR method? Keep your critique under 3 sentences."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    try:
        # Generate text
        output = llm_pipe(messages, max_new_tokens=100)
        # Extract the assistant's reply (depends on the pipeline output format)
        if isinstance(output, list) and "generated_text" in output[0]:
            # For chat templates, generated_text is a list of dicts or string
            gen_text = output[0]["generated_text"]
            if isinstance(gen_text, list):
                response = gen_text[-1]["content"]
            else:
                response = gen_text.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
        else:
            response = str(output)
            
        return {"content_feedback": response}
    except Exception as e:
        logger.error(f"LLM evaluation failed: {e}")
        return {"content_feedback": "Content evaluation temporarily unavailable."}

def generate_questions(job_description: str) -> list:
    """Generate 3 custom interview questions based on a pasted JD."""
    if not job_description:
        return []
        
    llm_pipe = load_llm_pipeline()
    system_prompt = "You are an expert recruiter."
    user_message = f"Based on this job description, generate 3 challenging behavioral interview questions. Output them as a numbered list.\n\nJob Description:\n{job_description}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    try:
        output = llm_pipe(messages, max_new_tokens=150)
        if isinstance(output, list) and "generated_text" in output[0]:
            gen_text = output[0]["generated_text"]
            if isinstance(gen_text, list):
                response = gen_text[-1]["content"]
            else:
                response = gen_text.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
                
        # Parse numbered list
        questions = [q.strip().lstrip("1234567890. ") for q in response.split("\n") if q.strip()]
        return questions[:3]
    except Exception as e:
        logger.error(f"LLM question generation failed: {e}")
        return []
