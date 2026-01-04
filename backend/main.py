"""
================================================================================
SINHALA SPEECH THERAPY PLATFORM - BACKEND API
================================================================================

AI-Powered Interactive Learning Platform for Hearing-Impaired Children

Author: Data Science Undergraduate
Research Project: Final Year - Computational Audiology
Last Updated: January 3, 2026

================================================================================
SYSTEM OVERVIEW
================================================================================

This FastAPI backend powers an adaptive speech therapy game for Sinhala-speaking
hearing-impaired children (ages 4-12). The system combines:

- Story Generation: Custom SinLlama model + Google Gemini fallback
- Adaptive Learning: Thompson Sampling adjusts difficulty in real-time
- Engagement Tracking: Multimodal signals (emotion, timing, gestures)
- Dropout Prevention: Predictive interventions (30-60s early warning)
- Phoneme Analysis: First Sinhala phoneme confusion dataset
- Hearing Loss Estimation: Non-invasive WHO severity classification

================================================================================
FEATURES IMPLEMENTED (6/10)
================================================================================

✅ Feature 1: Adaptive Difficulty Engine (Thompson Sampling)
✅ Feature 2: Phoneme Confusion Matrix (Apriori Association Rules)
✅ Feature 3: Multimodal Engagement Scorer (LSTM + Weighted Ensemble)
✅ Feature 4: Visual Attention Heatmap (Gaze Tracking + Fixation Detection)
✅ Feature 5: Real-Time Dropout Prediction (13-feature behavioral model)
✅ Feature 6: Hearing Loss Severity Estimator (16-feature audiometric model)

📋 Planned Features 7-10: Peer Benchmarking, Automated Reports, 
   Temporal Pattern Mining, Transfer Learning

================================================================================
TECHNOLOGY STACK
================================================================================

Framework: FastAPI (Python 3.11+)
Database: Supabase (PostgreSQL)
ML Models: TensorFlow, XGBoost, Scikit-learn
LLM: SinLlama (Hugging Face), Google Gemini 2.5 Flash
Frontend: React 19.2 with MediaPipe (face/hand tracking)

================================================================================
API ENDPOINTS
================================================================================

Core Story Generation:
- POST /generate-story - Generate Sinhala story with target words
- POST /submit-answer - Check answer correctness and update score
- POST /text-to-speech - Convert Sinhala text to audio (Gemini TTS)

Feature 1 - Adaptive Difficulty:
- POST /update-performance - Submit performance data
- POST /get-difficulty-recommendation - Get next difficulty level
- GET /user-progress/{user_id} - View learning analytics

Feature 2 - Phoneme Analysis:
- POST /track-phoneme-confusion - Log phoneme errors
- GET /phoneme-confusion-matrix/{user_id} - View confusion heatmap
- GET /phoneme-therapy-recommendations/{user_id} - Get therapy plan
- POST /phoneme-analysis-report - Generate comprehensive report

Feature 3 - Engagement Tracking:
- POST /track-engagement - Submit multimodal engagement signal
- GET /engagement-dashboard/{user_id} - Real-time engagement metrics
- POST /engagement-report - Generate therapist/parent reports
- GET /predict-next-engagement/{user_id} - LSTM-based prediction

Feature 4 - Attention Tracking:
- POST /track-gaze - Log gaze coordinates
- GET /attention-heatmap/{user_id} - Generate attention heatmap
- GET /attention-recommendations/{user_id} - UI optimization tips
- POST /attention-report - Generate attention analysis report

Feature 5 - Dropout Prevention:
- POST /predict-dropout - Calculate real-time dropout risk
- GET /dropout-analysis/{user_id} - Historical dropout patterns

Feature 6 - Hearing Loss Estimation:
- POST /estimate-severity - Estimate hearing loss severity
- GET /severity-history/{user_id} - Track severity over time

System Health:
- GET / - System status and info
- GET /health - Service health check (Gemini, SinLlama, Database)
- GET /test-story-generation - Test story generation pipeline

================================================================================
RUNNING THE APPLICATION
================================================================================

1. Install dependencies: pip install -r requirements.txt
2. Create .env file with API keys (see .env.example)
3. Run database migrations: python database_migration.py
4. Start server: uvicorn main:app --reload
5. Access API docs: http://localhost:8000/docs

================================================================================
"""

import os
import json
import time
import requests
import base64
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import google.generativeai as genai
from dotenv import load_dotenv
from supabase import create_client, Client

# ======================================
# ML MODEL IMPORTS (Features 1-6)
# ======================================

# FEATURE 1: Adaptive Difficulty Engine
# Dynamically adjusts question difficulty based on child's performance using Thompson Sampling
from ml_model.adaptive_difficulty import AdaptiveDifficultyEngine
from ml_model.utils.sinhala_phonetics import classify_word_difficulty

# FEATURE 2: Phoneme Confusion Analyzer
# Tracks which Sinhala sounds (phonemes) the child confuses (e.g., ප vs බ, ත vs ද)
from ml_model.phoneme_analyzer import PhonemeConfusionAnalyzer
from ml_model.phoneme_visualization import PhonemeVisualizationGenerator, TherapyReportGenerator

# FEATURE 3: Multimodal Engagement Scorer
# Combines emotion, gesture quality, response time, and attention into single engagement score (0-100)
from ml_model.engagement_scorer import EngagementScorer, EngagementInterventionSystem
from ml_model.lstm_temporal import SimpleLSTM, DropoutPredictor, TemporalFeatures
from ml_model.engagement_visualization import EngagementVisualizationGenerator, EngagementReportGenerator

# FEATURE 4: Visual Attention Heatmap Tracker
# Uses eye-gaze tracking to detect where child looks on screen (attention zones, fixations, drift)
from ml_model.attention_tracker import AttentionHeatmapTracker, GazePoint, AttentionZone, AttentionDrift
from ml_model.attention_visualization import AttentionVisualizationGenerator, AttentionReportGenerator

# FEATURE 5: Real-Time Dropout Predictor
# Predicts if child is about to quit session based on frustration signals (errors, low engagement)
from ml_model.dropout_predictor import RealTimeDropoutPredictor, SessionFeatures, SessionDropoutAnalyzer

# FEATURE 6: Hearing Loss Severity Estimator
# Estimates severity of hearing loss (mild/moderate/severe) from behavioral patterns
from ml_model.hearing_loss_estimator import HearingLossSeverityEstimator, AudiometricFeatures, BehavioralFeatureExtractor

# ======================================
# 1. CONFIGURATION & SETUP
# ======================================
"""
This section initializes all external services and loads configuration.
- Environment variables (.env file) for API keys and database credentials
- Google Gemini AI for story generation fallback and text-to-speech
- Supabase PostgreSQL database for persistent storage
- Hugging Face API for custom SinLlama model inference
"""

# Load environment variables from .env file
# Required: GOOGLE_API_KEY, SUPABASE_URL, SUPABASE_KEY
load_dotenv()

# Hugging Face Settings
# Your custom fine-tuned SinLlama model hosted on HF Spaces
HF_TOKEN = "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"  # Hugging Face API token
HF_MODEL_ID = "thulasika-n/SinLlama-Story-Teller"  # Your model ID

# Initialize External APIs
# Configure Google Gemini AI for story formatting and fallback
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Configure Supabase (PostgreSQL database for user data, stories, performance logs)
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Initialize FastAPI application
app = FastAPI()

# ======================================
# IN-MEMORY STORAGE FOR ML MODELS
# ======================================
# Note: In production, these should be stored in Redis or database
# Currently using dictionaries for development/testing

# FEATURE 1: Adaptive Difficulty Engines (one per user)
# Stores Thompson Sampling state for each child
user_adaptive_engines = {}

# FEATURE 2: Phoneme Confusion Analyzers (one per user)
# Stores phoneme confusion matrices for each child
user_phoneme_analyzers = {}

# FEATURE 3: Engagement Scorers (one per user)
# Stores engagement history and LSTM state for each child
user_engagement_scorers = {}
# Global intervention system (shared across all users)
engagement_intervention_system = EngagementInterventionSystem()

# FEATURE 4: Visual Attention Trackers (one per user)
# Stores gaze history and heatmap data for each child
user_attention_trackers = {}

# FEATURE 5: Dropout Predictors (shared models)
# Real-time dropout risk calculator
dropout_predictor = RealTimeDropoutPredictor()
# Session-level dropout analyzer
dropout_analyzer = SessionDropoutAnalyzer()

# FEATURE 6: Hearing Loss Severity Estimator (shared model)
# Estimates hearing loss severity from behavioral data
hearing_loss_estimator = HearingLossSeverityEstimator()
# Extracts behavioral features from session logs
feature_extractor = BehavioralFeatureExtractor()

# ======================================
# 2. MIDDLEWARE (CORS)
# ======================================
# Enable Cross-Origin Resource Sharing for frontend access
# WARNING: "*" allows ALL origins - restrict this in production!
origins = ["*"]  # Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Domains that can access API
    allow_credentials=True,  # Allow cookies/auth headers
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# ======================================
# 3. EXCEPTION HANDLERS
# ======================================
# Custom error handler for request validation failures
# Returns user-friendly error messages when request data is invalid
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Handle Pydantic validation errors (missing fields, wrong types, etc.)"""
    print(f"Validation Error: {exc}")
    return PlainTextResponse(str(exc), status_code=400)

# ======================================
# 4. REQUEST/RESPONSE DATA MODELS
# ======================================
# Pydantic models for API request/response validation

# Story Generation Request
class StoryRequest(BaseModel):
    """Request model for generating a new Sinhala story"""
    user_id: str  # Unique user identifier
    topic: str  # Story topic (currently using difficult_words from profile)
    model_type: str = "sinllama"  # Which model to use (default: custom SinLlama)

# Answer Submission Request
class AnswerRequest(BaseModel):
    """Request model for submitting an answer to a story question"""
    user_id: str  # Who answered
    story_id: int  # Which story question
    answer: str  # Child's selected answer

# Text-to-Speech Request
class TTSRequest(BaseModel):
    """Request model for generating audio from Sinhala text"""
    text: str  # Sinhala text to convert to speech

# Performance Update Request (Feature 1)
class PerformanceUpdate(BaseModel):
    """Update adaptive difficulty engine with child's performance"""
    user_id: str  # Child's ID
    story_id: int  # Question ID
    is_correct: bool  # Did they get it right?
    response_time: float  # How long did it take (seconds)
    engagement_score: Optional[float] = None  # Optional engagement signal

# Difficulty Recommendation Request (Feature 1)
class DifficultyRequest(BaseModel):
    """Request recommended difficulty level for next question"""
    user_id: str  # Child's ID

# Phoneme Confusion Tracking Request (Feature 2)
class PhonemeAnswerRequest(BaseModel):
    """Track phoneme confusion when child selects wrong word"""
    user_id: str  # Child's ID
    target_word: str  # Correct word (e.g., "පාන්")
    selected_word: str  # What they chose (e.g., "බාන්")
    is_correct: bool  # Whether they got it right

# Phoneme Analysis Report Request (Feature 2)
class PhonemeAnalysisRequest(BaseModel):
    """Generate phoneme confusion report for therapist"""
    user_id: str  # Child's ID
    report_type: Optional[str] = "json"  # Output format: 'json', 'text_english', 'text_sinhala'
    days: Optional[int] = 30  # Analysis period (last N days)

# Engagement Signal Request (Feature 3)
class EngagementSignalRequest(BaseModel):
    """Record multimodal engagement signals (emotion, gesture, attention)"""
    user_id: str  # Child's ID
    emotion: Optional[str] = "neutral"  # Detected emotion: "happy", "neutral", "sad", "angry", "afraid"
    gesture_accuracy: float  # Hand gesture quality (0-1, from MediaPipe)
    response_time_seconds: float  # How long to answer (seconds)
    has_eye_contact: Optional[bool] = True  # Looking at screen? (from face mesh)

# Engagement Report Request (Feature 3)
class EngagementReportRequest(BaseModel):
    """Generate engagement report for therapist or parent"""
    user_id: str  # Child's ID
    child_name: str  # Child's name for report
    report_type: str = "therapist"  # Report audience: 'therapist' (detailed) or 'parent' (simplified)
    language: str = "english"  # Report language: 'english' or 'sinhala'

# Gaze Point Tracking Request (Feature 4)
class GazePointRequest(BaseModel):
    """Record single gaze point for attention heatmap"""
    user_id: str  # Child's ID
    x: float  # Horizontal gaze position (0=left, 1=right)
    y: float  # Vertical gaze position (0=top, 1=bottom)
    confidence: Optional[float] = 1.0  # Gaze detection confidence (0-1)

# Attention Analysis Request (Feature 4)
class AttentionAnalysisRequest(BaseModel):
    """Analyze attention patterns over time window"""
    user_id: str  # Child's ID
    time_window_minutes: Optional[int] = 30  # Analysis period

# Attention Report Request (Feature 4)
class AttentionReportRequest(BaseModel):
    """Generate attention analysis report"""
    user_id: str  # Child's ID
    child_name: str  # Child's name for report
    report_type: str = "therapist"  # Report audience: 'therapist' or 'parent'
    language: str = "english"  # Report language

# Dropout Prediction Request (Feature 5)
class DropoutPredictionRequest(BaseModel):
    """Predict if child is about to quit session (real-time)"""
    user_id: str  # Child's ID
    
    # Real-time frustration signals
    accuracy_decline_rate: float  # How fast is accuracy dropping? (% per minute)
    consecutive_errors: int  # How many wrong answers in a row?
    avg_response_time_increase: float  # Getting slower? (% increase)
    
    # Engagement signals
    engagement_score_ma: float  # Current engagement score (0-100, moving average)
    low_engagement_duration: float  # How long below 40? (seconds)
    
    # Session context
    session_duration_minutes: float  # How long in current session?
    time_since_last_reward: float  # Time since last success (minutes)
    current_difficulty_level: int  # Current question difficulty (1-5)
    questions_remaining: int  # How many questions left?
    
    # Optional behavioral signals
    gesture_accuracy_decline: Optional[float] = 0.0  # Hand gesture quality drop
    audio_replay_frequency: Optional[float] = 0.0  # How often replaying audio?
    pause_count: Optional[int] = 0  # How many times paused?

# Hearing Loss Severity Estimation Request (Feature 6)
class SeverityEstimationRequest(BaseModel):
    """Estimate hearing loss severity from behavioral patterns"""
    user_id: str  # Child's ID
    child_age_months: Optional[int] = 60  # Child's age (default: 5 years)
    
    # Optional: Provide audiometric features directly
    # If not provided, system extracts from session logs automatically
    avg_volume_preference: Optional[float] = None  # Preferred volume (0-100)
    audio_replay_rate: Optional[float] = None
    optimal_speech_rate: Optional[float] = None
    background_noise_tolerance: Optional[float] = None
    high_freq_confusion_rate: Optional[float] = None
    low_freq_confusion_rate: Optional[float] = None
    consonant_vowel_ratio: Optional[float] = None
    voicing_confusion_rate: Optional[float] = None
    avg_response_time_visual: Optional[float] = None
    avg_response_time_audio_only: Optional[float] = None
    response_time_ratio: Optional[float] = None
    accuracy_quiet: Optional[float] = None
    accuracy_noisy: Optional[float] = None
    accuracy_degradation: Optional[float] = None
    frustration_episodes: Optional[int] = None
    help_request_frequency: Optional[float] = None

# --- 5. HELPER FUNCTIONS ---

# ======================================
# 5. HELPER FUNCTIONS
# ======================================

def generate_story_with_gemini_fallback(keywords: str) -> str:
    """
    Generates a Sinhala story using Gemini 2.5 Flash as fallback.
    Used when SinLlama model is unavailable or fails.
    
    Args:
        keywords: Comma-separated Sinhala words to include (e.g., "හාවා, ඉබ්බා, කුකුළා")
    
    Returns:
        str: Generated Sinhala story text (3-5 sentences)
        None: If generation fails after all retries
    
    Flow:
        1. Build prompt with target words and requirements
        2. Call Gemini 2.5 Flash (max 3 retries)
        3. Validate output (must be >10 chars, actual Sinhala)
        4. Return story or None
    """
    # Log that we're using fallback model
    print(f"🔄 Using Gemini 2.5 Flash fallback for story generation...")
    
    # Build prompt for Gemini with strict requirements
    prompt = f"""
    You are a Sinhala children's story writer for hearing-impaired children aged 4-12.
    
    Write a SHORT, SIMPLE Sinhala story (3-5 sentences) that includes these target words:
    {keywords}
    
    Requirements:
    - Use SIMPLE vocabulary suitable for young children
    - Include ALL the target words naturally in the story
    - Keep sentences SHORT (5-8 words each)
    - Make the story engaging and fun
    - Write ONLY in Sinhala script (no English)
    - Do NOT add titles, headings, or explanations
    - Return ONLY the story text
    
    Example format:
    කුකුළා උදේ හඬනවා. හාවා වතුරේ යනවා. ඔවුන් හොඳ මිතුරන්.
    """
    
    # Retry loop (up to 3 attempts)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Initialize Gemini model
            model = genai.GenerativeModel('gemini-2.5-flash')
            # Generate story with specific parameters
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=300
                )
            )
            story = response.text.strip()
            
            # Validate that we got actual Sinhala content
            if story and len(story) > 10:
                print("✅ Story generated successfully using Gemini fallback!")
                return story
            else:
                print(f"⚠️ Gemini returned insufficient content on attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                    
        except Exception as e:
            print(f"❌ Gemini fallback error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
    
    print("❌ All Gemini fallback attempts failed")
    return None

def generate_story_with_huggingface(keywords: str):
    """
    Connects to your SinLlama model deployed on Hugging Face Spaces.
    Your custom-trained model runs on HF's servers (no local space needed).
    Falls back to Gemini 2.5 Flash if SinLlama fails.
    
    Args:
        keywords: Comma-separated Sinhala words to include in story
    
    Returns:
        str: Generated Sinhala story (from SinLlama or Gemini fallback)
        None: If both SinLlama and Gemini fail
    
    Flow:
        1. Call SinLlama Space API (up to 3 retries)
        2. Handle cold start (Space sleeping - first request takes ~30s)
        3. Validate story output
        4. If SinLlama fails → Fall back to Gemini
    
    Note: First request after Space sleep may timeout - this is normal!
    """
    # Your Hugging Face Space API endpoint
    # This is your deployed SinLlama model's prediction endpoint
    api_url = "https://thulasika-n-sinllama-story-api.hf.space/api/predict"
    
    # Build request payload (format expected by Gradio API)
    payload = {
        "data": [
            keywords,  # keywords
            250,       # max_tokens
            0.7        # temperature
        ]
    }
    
    print(f"🚀 Calling your SinLlama Space: thulasika-n/sinllama-story-api...")
    print(f"📝 Keywords: {keywords}")
    
    # Retry loop (first request after sleep may take ~30s to load model)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Longer timeout for first request (model loading)
            timeout = 120 if attempt == 0 else 60
            print(f"   Attempt {attempt + 1}/{max_retries} (timeout: {timeout}s)...")
            
            response = requests.post(api_url, json=payload, timeout=timeout)
            
            if response.status_code == 200:
                result = response.json()
                story = result.get("data", [None])[0]
                
                # Validate story content
                if story and len(story.strip()) > 10:
                    print("✅ Story generated successfully from SinLlama Space!")
                    return story
                else:
                    print(f"⚠️ SinLlama returned empty/invalid story on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        time.sleep(10)
                        continue
            else:
                error_msg = response.text[:200] if response.text else "Unknown error"
                print(f"❌ Space Error (Status {response.status_code}): {error_msg}")
                if attempt < max_retries - 1:
                    print(f"   Retrying in 10s... (attempt {attempt+2}/{max_retries})")
                    time.sleep(10)
                    continue
                
        except requests.Timeout:
            print(f"⏰ Timeout on attempt {attempt+1}/{max_retries}...")
            if attempt == 0:
                print("   (Space may be sleeping - loading model, please wait...)")
            if attempt < max_retries - 1:
                print(f"   Retrying in 10s...")
                time.sleep(10)
                continue
        except requests.ConnectionError as e:
            print(f"❌ Connection Error on attempt {attempt+1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                print(f"   Retrying in 10s...")
                time.sleep(10)
                continue
        except Exception as e:
            print(f"❌ Unexpected Error on attempt {attempt+1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                print(f"   Retrying in 10s...")
                time.sleep(10)
                continue
    
    # SinLlama failed after all retries - try Gemini fallback
    print("\n⚠️ SinLlama unavailable after all retries, switching to Gemini 2.5 Flash fallback...")
    return generate_story_with_gemini_fallback(keywords)

def format_story_for_game(raw_story_text: str, difficult_words: list):
    """
    Uses Gemini to format the raw text into the game's JSON structure.
    
    Args:
        raw_story_text: Raw Sinhala story from SinLlama/Gemini
        difficult_words: List of target words child is practicing
    
    Returns:
        str: JSON string with structured story data
    
    Output Format:
        {
          "story_sentences": [
            {
              "text": "Full sentence",
              "has_target_word": true,
              "target_word": "word",
              "options": ["word", "distractor1", "distractor2", "distractor3"]
            }
          ]
        }
    
    Flow:
        1. Break story into sentences
        2. Identify target words in each sentence
        3. Generate phonetically similar distractors
        4. Return structured JSON for game frontend
    """
    words_list = ", ".join(difficult_words)
    
    prompt = f"""
    I have a raw Sinhala story. Please format it for a hearing therapy game.
    
    RAW STORY: "{raw_story_text}"
    TARGET WORDS: {words_list}
    
    INSTRUCTIONS:
    1. Break the story into sentences.
    2. Identify sentences that contain the Target Words.
    3. If a target word is missing from a sentence, select another simple noun as the target.
    4. Generate 3 similar-sounding Sinhala distractor words for each target.
    5. RETURN ONLY VALID JSON. No markdown formatting.
    
    JSON Format:
    {{
      "story_sentences": [
        {{
          "text": "Full sentence text",
          "has_target_word": true,
          "target_word": "word",
          "options": ["word", "dist1", "dist2", "dist3"]
        }}
      ]
    }}
    """
    
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(prompt)
    clean_json = response.text.strip().replace("```json", "").replace("```", "")
    return clean_json

def generate_audio_with_gemini(text: str) -> bytes:
    """
    Generates audio using Gemini TTS (Text-to-Speech).
    
    Args:
        text: Sinhala text to convert to speech
    
    Returns:
        bytes: WAV audio data
    
    Raises:
        Exception: If no audio is generated
    
    Note: Uses "Puck" voice - adjust voice_name for different accents
    """
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(
        text,
        generation_config=genai.GenerationConfig(
            response_modalities=["AUDIO"],
            speech_config=genai.SpeechConfig(
                voice_config=genai.VoiceConfig(
                    prebuilt_voice_config=genai.PrebuiltVoiceConfig(
                        voice_name="Puck"
                    )
                )
            )
        )
    )
    if response.candidates and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'inline_data') and part.inline_data:
                return part.inline_data.data
    raise Exception("No audio generated")

# ======================================
# 6. API ENDPOINTS
# ======================================

# --------------------------------------
# CORE STORY GENERATION
# --------------------------------------

@app.post("/generate-story")
async def get_story(request: StoryRequest):
    """
    Main story generation endpoint.
    
    Flow:
        1. Fetch user profile (learning level, difficult words)
        2. Generate raw story using SinLlama → Gemini fallback
        3. Format story into game JSON structure
        4. Save to database
        5. Return structured story with sentences + options
    
    Request Body:
        - user_id: Child's unique ID
        - topic: Story topic (optional - uses difficult_words from profile)
        - model_type: "sinllama" (default)
    
    Response:
        {
          "story": {
            "id": 123,
            "story_text": "Full story...",
            "story_sentences": [
              {"text": "...", "target_word": "...", "options": [...]}
            ]
          }
        }
    
    Error Handling:
        - Returns 503 if both SinLlama and Gemini fail
        - Returns 500 for database errors
    """
    try:
        # ===== STEP 1: Fetch User Profile =====
        # Get child's learning level and target words from database
        profile_res = supabase.table('profiles').select('learning_level, difficult_words').eq('id', request.user_id).execute()
        
        if not profile_res.data:
            # Fallback defaults if user not found in database
            difficult_words = ["හාවා", "ඉබ්බා", "කෑම"]  # Simple animal/food words
        else:
            # Use child's personalized target words
            difficult_words = profile_res.data[0].get('difficult_words', [])
            if not difficult_words:
                # Default harder words if profile exists but no words set
                difficult_words = ["පුස්තකාලය", "ආහාර", "විශ්වාසය"]

        # Convert word list to comma-separated string
        keywords = ", ".join(difficult_words)
        print(f"📝 Generating story for: {keywords}")

        # ===== STEP 2: Generate Raw Story =====
        # Try SinLlama first, then Gemini if it fails
        print("\n" + "="*60)
        print("STORY GENERATION PIPELINE")
        print("="*60)
        
        raw_story = generate_story_with_huggingface(keywords)
        
        if not raw_story:
            error_detail = (
                "Story generation failed. Both SinLlama and Gemini fallback are unavailable. "
                "Please ensure: \n"
                "1. Your Hugging Face Space is running (https://huggingface.co/spaces/thulasika-n/sinllama-story-api)\n"
                "2. Your Google API key is configured in .env file\n"
                "3. You have internet connectivity"
            )
            print(f"\n❌ CRITICAL ERROR: {error_detail}\n")
            raise HTTPException(status_code=503, detail=error_detail)
        
        # Log successful generation
        print(f"\n✅ Raw story generated ({len(raw_story)} characters)")
        print(f"Story preview: {raw_story[:100]}...\n")

        # ===== STEP 3: Format Story for Game =====
        # Convert plain text to structured JSON with options
        print("✨ Formatting story JSON...")
        # Use Gemini to structure the story
        game_json = format_story_for_game(raw_story, difficult_words)
        
        # Parse JSON response
        try:
            story_data = json.loads(game_json)
        except json.JSONDecodeError:
            # Gemini returned invalid JSON
            raise HTTPException(status_code=500, detail="Failed to parse story JSON from AI")

        # ===== STEP 4: Save to Database =====
        # Combine sentences to create the full text block for storage
        full_story_text = " ".join([s['text'] for s in story_data.get('story_sentences', [])])
        
        story_insert_res = supabase.table('stories').insert({
            'story_text': full_story_text,
            'question': 'ඔබට ඇහුණු වචනය කුමක්ද?', 
            'options': [],    # Options are handled per sentence in the JSON response
            'correct_answer': '' 
        }).execute()

        if not story_insert_res.data:
            raise HTTPException(status_code=500, detail="Failed to save story to database.")

        # Attach the detailed JSON structure to the response
        new_story = story_insert_res.data[0]
        new_story['story_sentences'] = story_data['story_sentences']
        
        return {"story": new_story}

    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------------------
# SYSTEM HEALTH & TESTING
# --------------------------------------

@app.get("/health")
def health_check():
    """
    Health check endpoint - tests all critical services.
    
    Tests:
        - Gemini AI (story formatting, TTS)
        - SinLlama Space (story generation)
        - Supabase database (data storage)
    
    Returns:
        {
          "status": "healthy" | "degraded" | "unhealthy",
          "timestamp": "ISO timestamp",
          "services": {
            "gemini": {"status": "available", ...},
            "sinllama": {"status": "available", ...},
            "database": {"status": "available", ...}
          }
        }
    
    Use Cases:
        - Monitoring/alerting
        - Pre-deployment checks
        - Debugging service issues
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {}
    }
    
    # Test Gemini
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        test_response = model.generate_content(
            "Test",
            generation_config=genai.GenerationConfig(max_output_tokens=10)
        )
        health_status["services"]["gemini"] = {
            "status": "available",
            "message": "Gemini 2.5 Flash is responding"
        }
    except Exception as e:
        health_status["services"]["gemini"] = {
            "status": "unavailable",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Test SinLlama Space
    try:
        api_url = "https://thulasika-n-sinllama-story-api.hf.space/api/predict"
        response = requests.post(
            api_url,
            json={"data": ["test", 10, 0.5]},
            timeout=30
        )
        if response.status_code == 200:
            health_status["services"]["sinllama"] = {
                "status": "available",
                "message": "SinLlama Space is responding"
            }
        else:
            health_status["services"]["sinllama"] = {
                "status": "unavailable",
                "message": f"HTTP {response.status_code}"
            }
            health_status["status"] = "degraded"
    except requests.Timeout:
        health_status["services"]["sinllama"] = {
            "status": "sleeping",
            "message": "Space is sleeping (will wake on first request)"
        }
        health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["sinllama"] = {
            "status": "unavailable",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Test Database
    try:
        result = supabase.table('profiles').select('id').limit(1).execute()
        health_status["services"]["database"] = {
            "status": "available",
            "message": "Supabase connection OK"
        }
    except Exception as e:
        health_status["services"]["database"] = {
            "status": "unavailable",
            "error": str(e)
        }
        health_status["status"] = "unhealthy"
    
    return health_status

@app.get("/test-story-generation")
def test_story_generation(keywords: str = "හාවා, ඉබ්බා, කුකුළා"):
    """
    Test endpoint to manually verify story generation with fallback.
    
    Usage: GET /test-story-generation?keywords=හාවා,ඉබ්බා,කුකුළා
    """
    try:
        print(f"\n{'='*60}")
        print(f"TESTING STORY GENERATION")
        print(f"Keywords: {keywords}")
        print(f"{'='*60}\n")
        
        # Test SinLlama
        raw_story = generate_story_with_huggingface(keywords)
        
        if not raw_story:
            return {
                "success": False,
                "error": "Both SinLlama and Gemini fallback failed",
                "keywords": keywords
            }
        
        # Format story
        difficult_words = [w.strip() for w in keywords.split(',')]
        game_json = format_story_for_game(raw_story, difficult_words)
        
        try:
            story_data = json.loads(game_json)
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"JSON parsing failed: {e}",
                "raw_story": raw_story,
                "raw_json": game_json
            }
        
        return {
            "success": True,
            "raw_story": raw_story,
            "formatted_story": story_data,
            "keywords": difficult_words,
            "sentence_count": len(story_data.get('story_sentences', []))
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "keywords": keywords
        }

@app.post("/submit-answer")
async def submit_answer(request: AnswerRequest):
    """
    Checks if child's answer is correct and updates their score.
    
    Args:
        request: AnswerRequest with user_id, story_id, answer
    
    Returns:
        {"correct": true/false, "feedback": "නිවැරදියි!" or "නැවත උත්සාහ කරන්න!"}
    
    Flow:
        1. Fetch correct answer from database
        2. Compare with child's answer
        3. If correct → add 10 points to score
        4. Return feedback in Sinhala
    """
    try:
        # Fetch the correct answer from database
        story_res = supabase.table('stories').select('correct_answer').eq('id', request.story_id).execute()
        if not story_res.data:
            raise HTTPException(status_code=404, detail="Story not found.")
        
        # Compare child's answer with correct answer
        correct_answer = story_res.data[0]['correct_answer']
        is_correct = request.answer == correct_answer

        # Generate feedback message in Sinhala
        feedback_text = "නිවැරදියි! (+10 ලකුණු!)" if is_correct else "නැවත උත්සාහ කරන්න!"
        
        if is_correct:
            # Update Score in Database
            profile_res = supabase.table('profiles').select('score').eq('id', request.user_id).execute()
            if profile_res.data:
                current_score = profile_res.data[0].get('score', 0)
                new_score = current_score + 10
                supabase.table('profiles').update({'score': new_score}).eq('id', request.user_id).execute()

        return {"correct": is_correct, "feedback": feedback_text}
    except Exception as e:
        print(f"Error in submit-answer: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/text-to-speech")
async def generate_speech(request: TTSRequest):
    """
    Converts Sinhala text to speech audio.
    
    Args:
        request: TTSRequest with text to convert
    
    Returns:
        {"audio": "base64_encoded_wav_data"}
    
    Usage:
        Frontend decodes base64 and plays audio for child
    """
    try:
        audio_content = generate_audio_with_gemini(request.text)
        audio_base64 = base64.b64encode(audio_content).decode('utf-8')
        return {"audio": audio_base64}
    except Exception as e:
        print(f"Error in text-to-speech: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    """
    Root endpoint - API information.
    
    Returns system status and available features.
    Useful for quick availability check.
    """
    return {
        "status": "Online", 
        "model": "SinLlama-Story-Teller (Hugging Face)",
        "features": [
            "Adaptive Difficulty (Thompson Sampling)",
            "Hand Gesture Recognition",
            "Sinhala Story Generation"
        ]
    }

# --------------------------------------
# FEATURE 1: ADAPTIVE DIFFICULTY ENGINE
# --------------------------------------

@app.post("/update-performance")
async def update_performance(request: PerformanceUpdate):
    """
    Update adaptive difficulty engine with user's performance.
    
    **FEATURE 1 - Adaptive Difficulty**
    Uses Thompson Sampling to automatically adjust question difficulty
    based on child's accuracy, speed, and engagement.
    
    Call this after EVERY question is answered!
    
    Args:
        request: PerformanceUpdate with is_correct, response_time, engagement_score
    
    Returns:
        {
          "success": true,
          "recommendations": {...},  // Next difficulty level
          "current_state": {...}     // Learning statistics
        }
    
    Flow:
        1. Load/create adaptive engine for user
        2. Record performance (correct/wrong, time, engagement)
        3. Update Thompson Sampling algorithm
        4. Save state to database
        5. Log performance for analytics
    """
    try:
        # Get or create adaptive engine for user
        if request.user_id not in user_adaptive_engines:
            # Try to load from database
            profile_res = supabase.table('profiles').select('adaptive_state, learning_level').eq('id', request.user_id).execute()
            
            if profile_res.data and profile_res.data[0].get('adaptive_state'):
                # Load saved state
                engine = AdaptiveDifficultyEngine.load_state(
                    request.user_id, 
                    profile_res.data[0]['adaptive_state']
                )
            else:
                # Create new engine with initial level based on learning_level
                initial_level = 2  # Default medium-easy
                if profile_res.data:
                    level_map = {'beginner': 1, 'intermediate': 2, 'advanced': 3}
                    initial_level = level_map.get(profile_res.data[0].get('learning_level', 'intermediate'), 2)
                
                engine = AdaptiveDifficultyEngine(request.user_id, initial_level)
            
            user_adaptive_engines[request.user_id] = engine
        
        engine = user_adaptive_engines[request.user_id]
        
        # Update performance
        recommendations = engine.update_performance(
            is_correct=request.is_correct,
            response_time=request.response_time,
            engagement_score=request.engagement_score
        )
        
        # Save state to database
        saved_state = engine.save_state()
        supabase.table('profiles').update({
            'adaptive_state': saved_state
        }).eq('id', request.user_id).execute()
        
        # Log performance to database for analytics
        supabase.table('performance_logs').insert({
            'user_id': request.user_id,
            'story_id': request.story_id,
            'is_correct': request.is_correct,
            'response_time': request.response_time,
            'engagement_score': request.engagement_score,
            'difficulty_level': engine.current_level_id
        }).execute()
        
        return {
            "success": True,
            "recommendations": recommendations,
            "current_state": engine.get_state_summary()
        }
        
    except Exception as e:
        print(f"Error in update-performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get-difficulty-recommendation")
async def get_difficulty_recommendation(request: DifficultyRequest):
    """
    Get recommended difficulty level for next story/question.
    
    **FEATURE 1 - Adaptive Difficulty**
    Uses Thompson Sampling to select optimal difficulty that maximizes learning.
    
    Args:
        request: DifficultyRequest with user_id
    
    Returns:
        {
          "difficulty_level": {  // Selected level (1-5)
            "level_id": 3,
            "description": "Medium",
            "word_length_range": [6, 10],
            ...
          },
          "current_state": {...}  // Thompson Sampling statistics
        }
    
    Call this BEFORE generating each new story/question!
    """
    try:
        # Get or create engine
        if request.user_id not in user_adaptive_engines:
            profile_res = supabase.table('profiles').select('adaptive_state, learning_level').eq('id', request.user_id).execute()
            
            if profile_res.data and profile_res.data[0].get('adaptive_state'):
                engine = AdaptiveDifficultyEngine.load_state(
                    request.user_id,
                    profile_res.data[0]['adaptive_state']
                )
            else:
                initial_level = 2
                if profile_res.data:
                    level_map = {'beginner': 1, 'intermediate': 2, 'advanced': 3}
                    initial_level = level_map.get(profile_res.data[0].get('learning_level', 'intermediate'), 2)
                engine = AdaptiveDifficultyEngine(request.user_id, initial_level)
            
            user_adaptive_engines[request.user_id] = engine
        
        engine = user_adaptive_engines[request.user_id]
        
        # Select next difficulty level
        next_level = engine.select_difficulty_level()
        
        return {
            "difficulty_level": next_level.to_dict(),
            "current_state": engine.get_state_summary()
        }
        
    except Exception as e:
        print(f"Error in get-difficulty-recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user-progress/{user_id}")
async def get_user_progress(user_id: str):
    """
    Get comprehensive progress analytics for a user.
    
    **FEATURE 1 - Adaptive Difficulty**
    
    Returns:
        {
          "adaptive_state": {  // Thompson Sampling state
            "current_level": 3,
            "total_trials": 50,
            "success_rate_by_level": {...},
            ...
          },
          "recent_performance": [...]  // Last 20 questions
        }
    
    Use Cases:
        - Parent dashboard
        - Therapist progress review
        - Analytics/reporting
    """
    try:
        # Get adaptive engine state
        state_summary = None
        if user_id in user_adaptive_engines:
            engine = user_adaptive_engines[user_id]
            state_summary = engine.get_state_summary()
        else:
            # Try to load from database
            profile_res = supabase.table('profiles').select('adaptive_state').eq('id', user_id).execute()
            if profile_res.data and profile_res.data[0].get('adaptive_state'):
                engine = AdaptiveDifficultyEngine.load_state(user_id, profile_res.data[0]['adaptive_state'])
                state_summary = engine.get_state_summary()
        
        # Get recent performance logs
        logs_res = supabase.table('performance_logs').select('*').eq('user_id', user_id).order('created_at', desc=True).limit(20).execute()
        
        return {
            "adaptive_state": state_summary,
            "recent_performance": logs_res.data if logs_res.data else []
        }
        
    except Exception as e:
        print(f"Error in user-progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------------------
# FEATURE 2: PHONEME CONFUSION ANALYZER
# --------------------------------------

@app.post("/track-phoneme-confusion")
async def track_phoneme_confusion(request: PhonemeAnswerRequest):
    """
    Track phoneme confusions when user answers a question.
    
    **FEATURE 2 - Phoneme Confusion Analyzer**
    Builds confusion matrix of which Sinhala sounds child mixes up.
    
    Example:
        Child hears "පාන්" (pān - bread)
        Child selects "බාන්" (bān - wrong)
        → System records "ප/බ confusion" (p/b voicing confusion)
    
    Call this endpoint after EVERY answer (correct or wrong)!
    
    Args:
        request: PhonemeAnswerRequest with target_word, selected_word, is_correct
    
    Returns:
        {
          "success": true,
          "message": "Phoneme confusion tracked",
          "statistics": {  // Updated stats
            "total_confusions": 15,
            "most_confused_pairs": [...],
            ...
          }
        }
    
    Database Updates:
        - Saves confusion matrix to user's phoneme_state
        - Logs errors to phoneme_errors table
    """
    try:
        # Get or create phoneme analyzer
        if request.user_id not in user_phoneme_analyzers:
            # Try to load from database
            profile_res = supabase.table('profiles').select('phoneme_state').eq('id', request.user_id).execute()
            
            if profile_res.data and profile_res.data[0].get('phoneme_state'):
                analyzer = PhonemeConfusionAnalyzer.load_state(
                    request.user_id,
                    profile_res.data[0]['phoneme_state']
                )
            else:
                analyzer = PhonemeConfusionAnalyzer(request.user_id)
            
            user_phoneme_analyzers[request.user_id] = analyzer
        
        analyzer = user_phoneme_analyzers[request.user_id]
        
        # Record the answer
        analyzer.record_answer(
            target_word=request.target_word,
            selected_word=request.selected_word,
            is_correct=request.is_correct
        )
        
        # Save state to database
        saved_state = analyzer.save_state()
        supabase.table('profiles').update({
            'phoneme_state': saved_state
        }).eq('id', request.user_id).execute()
        
        # Also log to phoneme_errors table if incorrect
        if not request.is_correct:
            confused_phonemes = analyzer._identify_confused_phonemes(
                request.target_word,
                request.selected_word
            )
            
            for p1, p2 in confused_phonemes:
                pair_key = "-".join(sorted([p1, p2]))
                
                # Insert or update phoneme error
                supabase.table('phoneme_errors').upsert({
                    'user_id': request.user_id,
                    'phoneme_pair': pair_key,
                    'error_count': 1
                }, on_conflict='user_id,phoneme_pair').execute()
        
        # Get updated statistics
        stats = analyzer.get_summary_statistics()
        
        return {
            "success": True,
            "message": "Phoneme confusion tracked",
            "statistics": stats
        }
        
    except Exception as e:
        print(f"Error in track-phoneme-confusion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/phoneme-confusion-matrix/{user_id}")
async def get_phoneme_confusion_matrix(user_id: str):
    """
    Get confusion matrix heatmap data for visualization.
    """
    try:
        # Get or create analyzer
        if user_id not in user_phoneme_analyzers:
            profile_res = supabase.table('profiles').select('phoneme_state').eq('id', user_id).execute()
            
            if profile_res.data and profile_res.data[0].get('phoneme_state'):
                analyzer = PhonemeConfusionAnalyzer.load_state(
                    user_id,
                    profile_res.data[0]['phoneme_state']
                )
            else:
                analyzer = PhonemeConfusionAnalyzer(user_id)
            
            user_phoneme_analyzers[user_id] = analyzer
        
        analyzer = user_phoneme_analyzers[user_id]
        
        # Generate heatmap data
        viz_gen = PhonemeVisualizationGenerator()
        heatmap_data = viz_gen.generate_heatmap_data(analyzer)
        
        return heatmap_data
        
    except Exception as e:
        print(f"Error in get-phoneme-confusion-matrix: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/phoneme-therapy-recommendations/{user_id}")
async def get_phoneme_therapy_recommendations(user_id: str):
    """
    Get personalized therapy recommendations based on phoneme confusions.
    """
    try:
        # Get or create analyzer
        if user_id not in user_phoneme_analyzers:
            profile_res = supabase.table('profiles').select('phoneme_state').eq('id', user_id).execute()
            
            if profile_res.data and profile_res.data[0].get('phoneme_state'):
                analyzer = PhonemeConfusionAnalyzer.load_state(
                    user_id,
                    profile_res.data[0]['phoneme_state']
                )
            else:
                analyzer = PhonemeConfusionAnalyzer(user_id)
            
            user_phoneme_analyzers[user_id] = analyzer
        
        analyzer = user_phoneme_analyzers[user_id]
        
        # Generate recommendations
        recommendations = analyzer.get_therapy_recommendations()
        
        return {
            "user_id": user_id,
            "total_recommendations": len(recommendations),
            "recommendations": [rec.to_dict() for rec in recommendations],
            "statistics": analyzer.get_summary_statistics()
        }
        
    except Exception as e:
        print(f"Error in get-phoneme-therapy-recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/phoneme-analysis-report")
async def generate_phoneme_analysis_report(request: PhonemeAnalysisRequest):
    """
    Generate comprehensive phoneme analysis report.
    Supports multiple formats: JSON, text (English/Sinhala).
    """
    try:
        # Get or create analyzer
        if request.user_id not in user_phoneme_analyzers:
            profile_res = supabase.table('profiles').select('phoneme_state').eq('id', request.user_id).execute()
            
            if profile_res.data and profile_res.data[0].get('phoneme_state'):
                analyzer = PhonemeConfusionAnalyzer.load_state(
                    request.user_id,
                    profile_res.data[0]['phoneme_state']
                )
            else:
                analyzer = PhonemeConfusionAnalyzer(user_id=request.user_id)
            
            user_phoneme_analyzers[request.user_id] = analyzer
        
        analyzer = user_phoneme_analyzers[request.user_id]
        report_gen = TherapyReportGenerator()
        
        if request.report_type == "json":
            report = report_gen.generate_json_report(analyzer)
            return report
        
        elif request.report_type == "text_english":
            text_report = report_gen.generate_text_report(analyzer, language='english')
            return {
                "report_type": "text",
                "language": "english",
                "content": text_report
            }
        
        elif request.report_type == "text_sinhala":
            text_report = report_gen.generate_text_report(analyzer, language='sinhala')
            return {
                "report_type": "text",
                "language": "sinhala",
                "content": text_report
            }
        
        else:
            raise HTTPException(status_code=400, detail="Invalid report_type. Use 'json', 'text_english', or 'text_sinhala'")
        
    except Exception as e:
        print(f"Error in generate-phoneme-analysis-report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ======================================
# FEATURE 3: MULTIMODAL ENGAGEMENT SCORER
# ======================================

@app.post("/track-engagement")
async def track_engagement(request: EngagementSignalRequest):
    """
    Record a multimodal engagement signal and get real-time prediction.
    
    This endpoint:
    1. Records emotion, gesture quality, response time, attention
    2. Computes unified engagement score (0-100)
    3. Detects trends and risk levels
    4. Triggers interventions if needed
    
    Called after each child interaction (answer, gesture, etc.)
    """
    try:
        # Get or create engagement scorer
        if request.user_id not in user_engagement_scorers:
            # Try to load from database
            profile_res = supabase.table('profiles').select('engagement_state').eq('id', request.user_id).execute()
            
            if profile_res.data and profile_res.data[0].get('engagement_state'):
                scorer = EngagementScorer.load_state(profile_res.data[0]['engagement_state'])
            else:
                scorer = EngagementScorer(user_id=request.user_id)
            
            user_engagement_scorers[request.user_id] = scorer
        
        scorer = user_engagement_scorers[request.user_id]
        
        # Record signal and get prediction
        prediction = scorer.record_signal(
            emotion=request.emotion,
            gesture_accuracy=request.gesture_accuracy,
            response_time_seconds=request.response_time_seconds,
            has_eye_contact=request.has_eye_contact
        )
        
        # Save state to database (async in production)
        state = scorer.save_state()
        supabase.table('profiles').update({
            'engagement_state': state
        }).eq('id', request.user_id).execute()
        
        # Trigger intervention if needed
        intervention_action = None
        if prediction.intervention_needed:
            intervention_action = engagement_intervention_system.trigger_intervention(
                intervention_type=prediction.intervention_type,
                language='sinhala'
            )
            
            # Log intervention
            engagement_intervention_system.log_intervention(
                user_id=request.user_id,
                intervention_type=prediction.intervention_type,
                engagement_score=prediction.engagement_score,
                context=prediction.to_dict()
            )
        
        return {
            "success": True,
            "engagement_prediction": prediction.to_dict(),
            "intervention_action": intervention_action
        }
    
    except Exception as e:
        print(f"Engagement tracking error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/engagement-dashboard/{user_id}")
async def get_engagement_dashboard(user_id: str):
    """
    Get real-time engagement dashboard data.
    
    Returns:
    - Current engagement score (gauge)
    - Engagement timeline chart
    - Component breakdown (radar chart)
    - Risk dashboard
    - Intervention history
    """
    try:
        # Get scorer
        if user_id not in user_engagement_scorers:
            # Try to load from database
            profile_res = supabase.table('profiles').select('engagement_state').eq('id', user_id).execute()
            
            if not profile_res.data or not profile_res.data[0].get('engagement_state'):
                return {
                    "error": "No engagement data found",
                    "user_id": user_id
                }
            
            scorer = EngagementScorer.load_state(profile_res.data[0]['engagement_state'])
            user_engagement_scorers[user_id] = scorer
        
        scorer = user_engagement_scorers[user_id]
        stats = scorer.get_statistics()
        
        # Generate visualizations
        viz_gen = EngagementVisualizationGenerator(user_id=user_id)
        
        # Current score gauge
        current_score = scorer.engagement_history[-1] if scorer.engagement_history else 50.0
        current_trend = stats['current_trend']
        current_risk = stats.get('current_risk', 'unknown')
        
        gauge_data = viz_gen.generate_realtime_gauge(
            current_score=current_score,
            trend=current_trend,
            risk_level=current_risk
        )
        
        # Timeline chart
        timeline_data = viz_gen.generate_timeline_chart(
            engagement_history=scorer.engagement_history
        )
        
        # Component breakdown (latest signal)
        component_scores = {}
        if scorer.signal_history:
            latest_signal = list(scorer.signal_history)[-1]
            component_scores = {
                'emotion': latest_signal.emotion_score * 100,
                'gesture': latest_signal.gesture_quality * 100,
                'response_time': (10.0 - min(latest_signal.response_time, 10.0)) * 10,
                'attention': latest_signal.attention_score * 100
            }
        
        component_data = viz_gen.generate_component_breakdown(component_scores)
        
        # Dropout risk prediction
        lstm = SimpleLSTM()
        dropout_predictor = DropoutPredictor()
        
        dropout_risk, risk_level, risk_factors = dropout_predictor.predict_dropout_risk(
            engagement_history=scorer.engagement_history
        )
        
        # Count consecutive low sessions
        consecutive_low = 0
        for score in reversed(scorer.engagement_history):
            if score < 40:
                consecutive_low += 1
            else:
                break
        
        risk_data = viz_gen.generate_risk_dashboard(
            dropout_risk=dropout_risk,
            risk_factors=risk_factors,
            consecutive_low_sessions=consecutive_low
        )
        
        # Intervention history
        interventions = scorer.get_intervention_history()
        intervention_data = viz_gen.generate_intervention_timeline(interventions)
        
        return {
            "user_id": user_id,
            "statistics": stats,
            "visualizations": {
                "gauge": gauge_data,
                "timeline": timeline_data,
                "components": component_data,
                "risk": risk_data,
                "interventions": intervention_data
            },
            "dropout_prediction": {
                "risk_score": dropout_risk,
                "risk_level": risk_level,
                "risk_factors": risk_factors
            }
        }
    
    except Exception as e:
        print(f"Dashboard error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/engagement-report")
async def generate_engagement_report(request: EngagementReportRequest):
    """
    Generate comprehensive engagement report for therapists or parents.
    
    Report types:
    - therapist: Detailed clinical analysis (English/Sinhala)
    - parent: Simplified progress summary (English/Sinhala)
    """
    try:
        # Get scorer
        if request.user_id not in user_engagement_scorers:
            profile_res = supabase.table('profiles').select('engagement_state').eq('id', request.user_id).execute()
            
            if not profile_res.data or not profile_res.data[0].get('engagement_state'):
                raise HTTPException(status_code=404, detail="No engagement data found")
            
            scorer = EngagementScorer.load_state(profile_res.data[0]['engagement_state'])
            user_engagement_scorers[request.user_id] = scorer
        
        scorer = user_engagement_scorers[request.user_id]
        stats = scorer.get_statistics()
        
        # Get dropout prediction
        dropout_predictor = DropoutPredictor()
        dropout_risk, risk_level, risk_factors = dropout_predictor.predict_dropout_risk(
            engagement_history=scorer.engagement_history
        )
        
        risk_analysis = {
            'dropout_risk': dropout_risk,
            'alert_level': risk_level,
            'risk_factors': risk_factors,
            'total_interventions': len(scorer.get_intervention_history())
        }
        
        # Generate visualizations for report
        viz_gen = EngagementVisualizationGenerator(user_id=request.user_id)
        
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        progress_summary = viz_gen.generate_progress_summary(
            engagement_history=scorer.engagement_history,
            start_date=start_date,
            end_date=end_date
        )
        
        # Generate report
        report_gen = EngagementReportGenerator(
            user_id=request.user_id,
            child_name=request.child_name
        )
        
        if request.report_type == "therapist":
            recommendations = viz_gen._generate_risk_recommendations(risk_factors)
            
            text_report = report_gen.generate_therapist_report(
                engagement_stats=progress_summary,
                risk_analysis=risk_analysis,
                recommendations=recommendations,
                language=request.language
            )
            
            return {
                "report_type": "therapist",
                "language": request.language,
                "content": text_report
            }
        
        else:  # parent
            text_report = report_gen.generate_parent_report(
                engagement_stats=progress_summary,
                language=request.language
            )
            
            return {
                "report_type": "parent",
                "language": request.language,
                "content": text_report
            }
    
    except Exception as e:
        print(f"Report generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict-next-engagement/{user_id}")
async def predict_next_engagement(user_id: str):
    """
    Use LSTM to predict next session's engagement score.
    
    Returns:
    - Predicted engagement score
    - Detected temporal pattern
    - Confidence level
    """
    try:
        from datetime import datetime
        
        # Get scorer
        if user_id not in user_engagement_scorers:
            profile_res = supabase.table('profiles').select('engagement_state').eq('id', user_id).execute()
            
            if not profile_res.data or not profile_res.data[0].get('engagement_state'):
                raise HTTPException(status_code=404, detail="No engagement data found")
            
            scorer = EngagementScorer.load_state(profile_res.data[0]['engagement_state'])
            user_engagement_scorers[user_id] = scorer
        
        scorer = user_engagement_scorers[user_id]
        
        # Use LSTM for prediction
        lstm = SimpleLSTM()
        
        # Extract temporal features (current context)
        now = datetime.now()
        temporal_features = TemporalFeatures(
            hour_of_day=now.hour,
            day_of_week=now.weekday(),
            session_duration=30.0,  # Default 30 min
            time_since_last_session=24.0,  # Assume 24h gap
            consecutive_sessions=len(scorer.engagement_history)
        )
        
        predicted_score, pattern, confidence = lstm.predict_next(
            engagement_sequence=scorer.engagement_history,
            temporal_features=temporal_features
        )
        
        return {
            "user_id": user_id,
            "predicted_engagement": predicted_score,
            "detected_pattern": pattern,
            "confidence": confidence,
            "recommendation": "Schedule session in morning for optimal engagement" if pattern == "morning_peak" else
                            "Consider shorter session - afternoon slump detected" if pattern == "afternoon_slump" else
                            "Continue with current schedule"
        }
    
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ======================================
# FEATURE 4: VISUAL ATTENTION HEATMAP TRACKER
# ======================================

@app.post("/track-gaze")
async def track_gaze(request: GazePointRequest):
    """
    Record a gaze point and update attention heatmap.
    
    This endpoint:
    1. Records gaze coordinates (x, y) with timestamp
    2. Detects fixations (sustained gaze in area)
    3. Updates attention zones and heatmap
    4. Detects attention drift events
    
    Called continuously during session (e.g., every 100ms from face mesh tracking)
    """
    try:
        # Get or create attention tracker
        if request.user_id not in user_attention_trackers:
            # Try to load from database
            profile_res = supabase.table('profiles').select('attention_state').eq('id', request.user_id).execute()
            
            if profile_res.data and profile_res.data[0].get('attention_state'):
                tracker = AttentionHeatmapTracker.load_state(profile_res.data[0]['attention_state'])
            else:
                tracker = AttentionHeatmapTracker(user_id=request.user_id)
            
            user_attention_trackers[request.user_id] = tracker
        
        tracker = user_attention_trackers[request.user_id]
        
        # Record gaze point
        result = tracker.record_gaze(
            x=request.x,
            y=request.y,
            confidence=request.confidence
        )
        
        # Save state to database (async in production)
        state = tracker.save_state()
        supabase.table('profiles').update({
            'attention_state': state
        }).eq('id', request.user_id).execute()
        
        return {
            "success": True,
            "gaze_result": result
        }
    
    except Exception as e:
        print(f"Gaze tracking error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/attention-heatmap/{user_id}")
async def get_attention_heatmap(user_id: str):
    """
    Get attention heatmap data for visualization.
    
    Returns:
    - Heatmap grid data
    - Hotspots (high-attention zones)
    - Gaze path visualization
    - Attention statistics
    """
    try:
        # Get tracker
        if user_id not in user_attention_trackers:
            # Try to load from database
            profile_res = supabase.table('profiles').select('attention_state').eq('id', user_id).execute()
            
            if not profile_res.data or not profile_res.data[0].get('attention_state'):
                return {
                    "error": "No attention data found",
                    "user_id": user_id
                }
            
            tracker = AttentionHeatmapTracker.load_state(profile_res.data[0]['attention_state'])
            user_attention_trackers[user_id] = tracker
        
        tracker = user_attention_trackers[user_id]
        
        # Generate heatmap data
        heatmap_data = tracker.generate_heatmap_data()
        
        # Generate visualizations
        viz_gen = AttentionVisualizationGenerator(user_id=user_id)
        
        # Heatmap overlay
        heatmap_overlay = viz_gen.generate_heatmap_overlay(
            heatmap_data=heatmap_data['heatmap'],
            grid_size=heatmap_data['grid_size'],
            color_scheme='hot'
        )
        
        # Gaze path
        gaze_history = [gaze.to_dict() for gaze in list(tracker.gaze_history)[-100:]]
        gaze_path = viz_gen.generate_gaze_path_visualization(
            gaze_history=gaze_history,
            max_points=100
        )
        
        # Get statistics
        stats = tracker.get_attention_statistics()
        
        # Focus quality dashboard
        focus_dashboard = viz_gen.generate_focus_quality_dashboard(
            focus_quality=stats['focus_quality'],
            drift_count=stats['drift_events'],
            avg_fixation_duration=stats['average_fixation_duration'],
            attention_distribution=stats['attention_distribution']
        )
        
        return {
            "user_id": user_id,
            "heatmap_data": heatmap_data,
            "visualizations": {
                "heatmap_overlay": heatmap_overlay,
                "gaze_path": gaze_path,
                "focus_dashboard": focus_dashboard
            },
            "statistics": stats
        }
    
    except Exception as e:
        print(f"Heatmap retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/attention-recommendations/{user_id}")
async def get_attention_recommendations(user_id: str):
    """
    Get UI optimization recommendations based on attention patterns.
    
    Returns:
    - UI placement recommendations
    - Content optimization suggestions
    - Attention intervention strategies
    """
    try:
        # Get tracker
        if user_id not in user_attention_trackers:
            profile_res = supabase.table('profiles').select('attention_state').eq('id', user_id).execute()
            
            if not profile_res.data or not profile_res.data[0].get('attention_state'):
                raise HTTPException(status_code=404, detail="No attention data found")
            
            tracker = AttentionHeatmapTracker.load_state(profile_res.data[0]['attention_state'])
            user_attention_trackers[user_id] = tracker
        
        tracker = user_attention_trackers[user_id]
        
        # Get recommendations
        recommendations = tracker.get_ui_recommendations()
        
        # Get statistics for context
        stats = tracker.get_attention_statistics()
        
        return {
            "user_id": user_id,
            "recommendations": recommendations,
            "context": {
                "focus_quality": stats['focus_quality'],
                "drift_events": stats['drift_events'],
                "total_gaze_points": stats['total_gaze_points']
            }
        }
    
    except Exception as e:
        print(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ======================================
# FEATURE 5: DROPOUT PREDICTION
# ======================================

@app.post("/predict-dropout")
async def predict_dropout_risk(request: DropoutPredictionRequest):
    """
    Real-time dropout prediction with intervention recommendations.
    
    Call this every 10 seconds during active session.
    
    Returns:
    - Dropout probability (0-1)
    - Risk level (low/medium/high/critical)
    - Intervention recommendation (if needed)
    - Contributing factors (top 3 reasons)
    
    Example Response:
    {
      "dropout_probability": 0.75,
      "risk_level": "critical",
      "intervention_needed": true,
      "intervention_type": "easier_content",
      "intervention_details": {
        "message": "මෙය උත්සාහ කරමු - මෙය පහසුයි!",
        "action": "Insert confidence booster question (easier word)"
      },
      "contributing_factors": [
        "4+ errors in a row (frustration)",
        "Very low engagement for 70s",
        "No reward for 3.0 minutes"
      ]
    }
    """
    try:
        # Build session features
        features = SessionFeatures(
            accuracy_decline_rate=request.accuracy_decline_rate,
            consecutive_errors=request.consecutive_errors,
            avg_response_time_increase=request.avg_response_time_increase,
            engagement_score_ma=request.engagement_score_ma,
            low_engagement_duration=request.low_engagement_duration,
            session_duration_minutes=request.session_duration_minutes,
            time_since_last_reward=request.time_since_last_reward,
            current_difficulty_level=request.current_difficulty_level,
            questions_remaining=request.questions_remaining,
            gesture_accuracy_decline=request.gesture_accuracy_decline,
            audio_replay_frequency=request.audio_replay_frequency,
            pause_count=request.pause_count
        )
        
        # Predict dropout risk
        prediction = dropout_predictor.predict_dropout_risk(features)
        
        # Get intervention details if needed
        intervention_details = None
        if prediction.intervention_needed and prediction.intervention_type:
            intervention_details = dropout_predictor.get_intervention_details(
                prediction.intervention_type,
                language='sinhala'
            )
        
        # Save prediction to database for research analysis
        try:
            supabase.table('dropout_predictions').insert({
                'user_id': request.user_id,
                'dropout_probability': prediction.dropout_probability,
                'risk_level': prediction.risk_level,
                'intervention_type': prediction.intervention_type,
                'contributing_factors': prediction.contributing_factors,
                'session_features': features.to_dict(),
                'timestamp': time.time()
            }).execute()
        except Exception as db_err:
            print(f"Warning: Could not save dropout prediction: {db_err}")
        
        return {
            "dropout_probability": prediction.dropout_probability,
            "risk_level": prediction.risk_level,
            "intervention_needed": prediction.intervention_needed,
            "intervention_type": prediction.intervention_type,
            "intervention_details": intervention_details,
            "contributing_factors": prediction.contributing_factors,
            "confidence": prediction.confidence
        }
    
    except Exception as e:
        print(f"Dropout prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dropout-analysis/{user_id}")
async def analyze_dropout_patterns(user_id: str, days: int = 30):
    """
    Analyze historical dropout patterns for a user.
    
    Returns:
    - Session completion rate
    - Average dropout time
    - Common dropout patterns
    - Risk trend over time
    """
    try:
        # Get dropout predictions from last N days
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        
        predictions = supabase.table('dropout_predictions')\
            .select('*')\
            .eq('user_id', user_id)\
            .gte('timestamp', cutoff_time)\
            .execute()
        
        if not predictions.data:
            return {
                "user_id": user_id,
                "message": "No dropout prediction data available",
                "sessions_analyzed": 0
            }
        
        # Aggregate statistics
        total_predictions = len(predictions.data)
        high_risk_count = sum(1 for p in predictions.data if p['risk_level'] in ['high', 'critical'])
        avg_dropout_prob = sum(p['dropout_probability'] for p in predictions.data) / total_predictions
        
        # Most common intervention types
        interventions = [p['intervention_type'] for p in predictions.data if p['intervention_type']]
        intervention_counts = {}
        for intervention in interventions:
            intervention_counts[intervention] = intervention_counts.get(intervention, 0) + 1
        
        # Most common risk factors
        all_factors = []
        for p in predictions.data:
            all_factors.extend(p.get('contributing_factors', []))
        
        factor_counts = {}
        for factor in all_factors:
            factor_counts[factor] = factor_counts.get(factor, 0) + 1
        
        top_factors = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "user_id": user_id,
            "analysis_period_days": days,
            "statistics": {
                "total_predictions": total_predictions,
                "high_risk_sessions": high_risk_count,
                "high_risk_rate": high_risk_count / total_predictions if total_predictions > 0 else 0,
                "average_dropout_probability": round(avg_dropout_prob, 3)
            },
            "common_interventions": intervention_counts,
            "top_risk_factors": [{"factor": f[0], "occurrences": f[1]} for f in top_factors]
        }
    
    except Exception as e:
        print(f"Dropout analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ======================================
# FEATURE 6: HEARING LOSS SEVERITY ESTIMATION
# ======================================

@app.post("/estimate-severity")
async def estimate_hearing_loss_severity(request: SeverityEstimationRequest):
    """
    Estimate hearing loss severity from behavioral patterns.
    
    Requires 30+ therapy sessions for reliable estimation.
    Returns WHO severity category (normal/mild/moderate/severe/profound).
    
    Example Response:
    {
      "severity_category": "moderate",
      "estimated_threshold_db": 52.3,
      "threshold_range": [42.3, 62.3],
      "confidence": 0.78,
      "clinical_description": "Difficulty with conversational speech without amplification",
      "intervention_recommendations": [
        "Bilateral hearing aids recommended",
        "FM system for classroom",
        "Intensive speech therapy"
      ]
    }
    """
    try:
        # Check if features provided directly or need extraction
        if request.avg_volume_preference is not None:
            # Use provided features
            features = AudiometricFeatures(
                avg_volume_preference=request.avg_volume_preference,
                audio_replay_rate=request.audio_replay_rate or 0,
                optimal_speech_rate=request.optimal_speech_rate or 1.0,
                background_noise_tolerance=request.background_noise_tolerance or 1.0,
                high_freq_confusion_rate=request.high_freq_confusion_rate or 0,
                low_freq_confusion_rate=request.low_freq_confusion_rate or 0,
                consonant_vowel_ratio=request.consonant_vowel_ratio or 1.0,
                voicing_confusion_rate=request.voicing_confusion_rate or 0,
                avg_response_time_visual=request.avg_response_time_visual or 5.0,
                avg_response_time_audio_only=request.avg_response_time_audio_only or 5.0,
                response_time_ratio=request.response_time_ratio or 1.0,
                accuracy_quiet=request.accuracy_quiet or 0.8,
                accuracy_noisy=request.accuracy_noisy or 0.6,
                accuracy_degradation=request.accuracy_degradation or 0.2,
                frustration_episodes=request.frustration_episodes or 0,
                help_request_frequency=request.help_request_frequency or 0
            )
        else:
            # Extract features from session logs
            # Get all sessions for user (need 30+ for reliable estimation)
            session_res = supabase.table('sessions')\
                .select('*')\
                .eq('user_id', request.user_id)\
                .order('created_at', desc=False)\
                .execute()
            
            if not session_res.data or len(session_res.data) < 30:
                raise HTTPException(
                    status_code=400,
                    detail=f"Insufficient data: {len(session_res.data) if session_res.data else 0} sessions found. Need 30+ sessions for reliable estimation."
                )
            
            # Extract features
            features = feature_extractor.extract_features(session_res.data)
            
            if not features:
                raise HTTPException(
                    status_code=400,
                    detail="Could not extract features from session data"
                )
        
        # Estimate severity
        estimate = hearing_loss_estimator.estimate_severity(
            features=features,
            child_age_months=request.child_age_months
        )
        
        # Save estimate to database
        try:
            supabase.table('severity_estimates').insert({
                'user_id': request.user_id,
                'severity_category': estimate.severity_category,
                'estimated_threshold_db': estimate.estimated_threshold_db,
                'threshold_range_lower': estimate.threshold_range[0],
                'threshold_range_upper': estimate.threshold_range[1],
                'confidence': estimate.confidence,
                'features': features.to_dict(),
                'timestamp': time.time()
            }).execute()
        except Exception as db_err:
            print(f"Warning: Could not save severity estimate: {db_err}")
        
        return estimate.to_dict()
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Severity estimation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/severity-history/{user_id}")
async def get_severity_history(user_id: str, limit: int = 10):
    """
    Get historical severity estimates for longitudinal analysis.
    
    Returns:
    - Severity trend over time
    - Average estimated threshold
    - Confidence evolution
    """
    try:
        estimates = supabase.table('severity_estimates')\
            .select('*')\
            .eq('user_id', user_id)\
            .order('timestamp', desc=True)\
            .limit(limit)\
            .execute()
        
        if not estimates.data:
            return {
                "user_id": user_id,
                "message": "No severity estimates found",
                "estimates": []
            }
        
        # Calculate trend statistics
        categories = [e['severity_category'] for e in estimates.data]
        thresholds = [e['estimated_threshold_db'] for e in estimates.data]
        confidences = [e['confidence'] for e in estimates.data]
        
        # Most recent vs oldest (check for progression/improvement)
        if len(estimates.data) >= 2:
            recent_threshold = estimates.data[0]['estimated_threshold_db']
            oldest_threshold = estimates.data[-1]['estimated_threshold_db']
            threshold_change = recent_threshold - oldest_threshold
            
            trend = 'stable'
            if threshold_change > 5:
                trend = 'worsening'
            elif threshold_change < -5:
                trend = 'improving'
        else:
            trend = 'insufficient_data'
        
        return {
            "user_id": user_id,
            "total_estimates": len(estimates.data),
            "current_severity": categories[0],
            "trend": trend,
            "statistics": {
                "avg_threshold_db": round(sum(thresholds) / len(thresholds), 1),
                "avg_confidence": round(sum(confidences) / len(confidences), 3),
                "min_threshold": min(thresholds),
                "max_threshold": max(thresholds)
            },
            "estimates": estimates.data
        }
    
    except Exception as e:
        print(f"Severity history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/attention-report")
async def generate_attention_report(request: AttentionReportRequest):
    """
    Generate comprehensive attention analysis report.
    
    Report types:
    - therapist: Detailed clinical analysis (English/Sinhala)
    - parent: Simplified progress summary (English/Sinhala)
    """
    try:
        # Get tracker
        if request.user_id not in user_attention_trackers:
            profile_res = supabase.table('profiles').select('attention_state').eq('id', request.user_id).execute()
            
            if not profile_res.data or not profile_res.data[0].get('attention_state'):
                raise HTTPException(status_code=404, detail="No attention data found")
            
            tracker = AttentionHeatmapTracker.load_state(profile_res.data[0]['attention_state'])
            user_attention_trackers[request.user_id] = tracker
        
        tracker = user_attention_trackers[request.user_id]
        
        # Get data for report
        stats = tracker.get_attention_statistics()
        heatmap_data = tracker.generate_heatmap_data()
        recommendations = tracker.get_ui_recommendations()
        
        # Generate report
        report_gen = AttentionReportGenerator(
            user_id=request.user_id,
            child_name=request.child_name
        )
        
        text_report = report_gen.generate_therapist_report(
            attention_stats=stats,
            heatmap_data=heatmap_data,
            recommendations=recommendations,
            language=request.language
        )
        
        return {
            "report_type": request.report_type,
            "language": request.language,
            "content": text_report
        }
    
    except Exception as e:
        print(f"Report generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ================================================================================
# END OF API ENDPOINTS
# ================================================================================
# For complete documentation, see README.md in project root
# API documentation available at: http://localhost:8000/docs (Swagger UI)
# ================================================================================

