# MockIQ: Real-Time Interview Evaluator

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://real-time-interview-evaluator-p8tcs6jnoupwkqmssgul4r.streamlit.app/)

> **🚀 Live Web App:** Try the live deployed application directly on your browser: **[real-time-interview-evaluator-p8tcs6jnoupwkqmssgul4r.streamlit.app](https://real-time-interview-evaluator-p8tcs6jnoupwkqmssgul4r.streamlit.app/)**

MockIQ is a fully local, privacy-first AI interview evaluator designed to help you practice and perfect your interview skills. It records your video and audio, processes it using state-of-the-art machine learning models, and provides comprehensive coaching on your content, fluency, body language, and grammar.

## 🚀 Features (V2.0)

*   **Dynamic JD Questions:** Paste a Job Description and MockIQ will use a local Generative LLM to dynamically generate 3 tailored behavioral interview questions for you to practice.
*   **Content Evaluation:** After you record an answer, the LLM evaluates your transcript to determine if you actually answered the prompt using the STAR method, and provides actionable feedback.
*   **Body Language Tracker:** Upload a `.mp4` video alongside your audio. MockIQ uses `DeepFace` to track your "Smile Time" and flag "Nervous Expressions".
*   **Grammar & Vocabulary Profiler:** Built with `spaCy`, MockIQ calculates your Lexical Richness and flags instances of passive voice to help you use stronger, active phrasing.
*   **Speech Fluency & Tone:** Uses `Whisper` for blazing-fast local transcription, calculating your Words Per Minute (WPM), latency, and pauses. It also analyzes your tone and sentiment.
*   **PDF Report Export:** Download a beautifully formatted PDF report of your entire session to save or share.
*   **100% Local & Private:** All models (Whisper, Qwen LLM, DeepFace, spaCy) run entirely on your local machine. No data is sent to external APIs.

## 🛠️ Technology Stack

*   **Frontend:** Streamlit, Vanilla CSS, Plotly
*   **Transcription:** OpenAI Whisper (`whisper-tiny`)
*   **Content & JD Generation:** Qwen (`Qwen/Qwen1.5-0.5B-Chat` via HuggingFace `transformers`)
*   **Emotion & Tone:** Superb Emotion Recognition (`superb/wav2vec2-base-superb-er`)
*   **Body Language:** DeepFace & OpenCV
*   **Grammar:** spaCy (`en_core_web_sm`)
*   **Reporting:** ReportLab

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ShreyasVavley/Real-Time-Interview-Evaluator.git
    cd Real-Time-Interview-Evaluator
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv mockiq_env
    mockiq_env\Scripts\activate  # Windows
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download required models:**
    ```bash
    python -m spacy download en_core_web_sm
    ```
    *(Note: The other HuggingFace models will automatically download on the first run.)*

## 🏃‍♂️ Usage

You can launch the application by simply double-clicking the `launch_mockiq.bat` file (on Windows), or by running Streamlit from the command line:

```bash
streamlit run app.py
```

1. Open your browser to `http://localhost:8501`.
2. (Optional) Paste a Job Description to generate custom questions.
3. Select an interview question and record your audio response.
4. (Optional) Upload a video of yourself answering the question for body language analysis.
5. Click **Process Session** and review your comprehensive analytics dashboard!

## 📄 License
This project is open-source and available under the MIT License.
