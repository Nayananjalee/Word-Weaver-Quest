"""
================================================================================
WORD WEAVER QUEST — BACKEND API
================================================================================

Interactive Sinhala word-learning game for hearing-impaired children (ages 4-12).

Core features:
  - Story generation (Gemini + SinLlama fallback)
  - Text-to-speech (Gemini TTS, Aoede voice)
  - Word management (therapist adds/removes target words)
  - Spaced Repetition (SM-2 + Leitner boxes)
  - Score tracking

Tech stack:
  - FastAPI (Python 3.11+)
  - SQLite (local) / Supabase PostgreSQL (production)
  - Google Gemini AI (text + TTS)
  - SinLlama on Hugging Face Spaces (story generation)

Run:
  uvicorn main:app --reload
  API docs: http://localhost:8000/docs
================================================================================
"""

import os
import json
import time
import random
import hashlib
import asyncio
from concurrent.futures import ThreadPoolExecutor
import requests
import base64
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from google import genai
from google.genai import types
from gtts import gTTS
import wave
import io
from dotenv import load_dotenv

# Local SQLite database (replaces Supabase in local dev)
from database import db as supabase

# Spaced Repetition Engine — SM-2 algorithm + Leitner boxes for word scheduling
from ml_model.spaced_repetition import SpacedRepetitionEngine, WordReviewCard, ReviewSession

# ================================================================================
# CONFIGURATION
# ================================================================================

load_dotenv()

# Hugging Face SinLlama model endpoint
HF_TOKEN = "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"
HF_MODEL_ID = "thulasika-n/SinLlama-Story-Teller"

# Google Gemini client (text generation)
_google_api_key = os.getenv("GOOGLE_API_KEY")
if not _google_api_key:
    print("⚠️ WARNING: GOOGLE_API_KEY is not set! Story generation will fail.")
    print("   Set it in .env (local) or Render Dashboard → Environment (production).")
else:
    print(f"✅ GOOGLE_API_KEY loaded (starts with {_google_api_key[:8]}...)")
client = genai.Client(api_key=_google_api_key)

# FastAPI app
app = FastAPI()

# Per-user SRS engines (in-memory; persisted to DB on each update)
user_srs_engines = {}

# TTS audio cache — avoids regenerating audio for the same text
tts_cache = {}  # key: hash(text) → value: base64 WAV string
TTS_CACHE_MAX = 200  # evict oldest if cache grows beyond this

# Thread pool for parallel TTS generation (2 workers to respect Gemini rate limits)
tts_executor = ThreadPoolExecutor(max_workers=2)

# ================================================================================
# MIDDLEWARE — CORS
# ================================================================================

origins = [
    "https://word-weaver-quest-ys15.vercel.app",  # Production (Vercel)
    "http://localhost:3000",                        # Local dev
    "http://localhost:5173",                        # Vite local dev
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================================================================
# EXCEPTION HANDLERS
# ================================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Return readable error when request validation fails."""
    print(f"Validation Error: {exc}")
    return PlainTextResponse(str(exc), status_code=400)

# ================================================================================
# REQUEST MODELS (Pydantic)
# ================================================================================

class StoryRequest(BaseModel):
    """Generate a new story for the child."""
    user_id: str
    topic: str
    model_type: str = "sinllama"

class AnswerRequest(BaseModel):
    """Submit an answer to a story question."""
    user_id: str
    story_id: int
    answer: str

class TTSRequest(BaseModel):
    """Convert Sinhala text to speech."""
    text: str

class UpdateScoreRequest(BaseModel):
    """Award stars → update score."""
    user_id: str
    stars: int

class WordManageRequest(BaseModel):
    """Add or remove a target word."""
    user_id: str
    word: str

class SRSAddWordRequest(BaseModel):
    """Add a word to the SRS deck."""
    user_id: str
    word: str
    phonemes: Optional[List[str]] = None
    difficulty_class: Optional[str] = "medium"

class SRSReviewRequest(BaseModel):
    """Submit a word review result (SM-2 quality 0-5)."""
    user_id: str
    word: str
    quality: int          # 0=blackout … 5=perfect
    response_time: float
    confused_with: Optional[str] = None

class SRSSessionRequest(BaseModel):
    """Request a review session."""
    user_id: str
    max_words: Optional[int] = 8
    include_new: Optional[int] = 2

# ================================================================================
# HELPER — SRS engine per user
# ================================================================================

def _get_or_create_srs(user_id: str) -> SpacedRepetitionEngine:
    """Load SRS engine from DB cache or create a fresh one."""
    if user_id not in user_srs_engines:
        profile_res = supabase.table('profiles').select('srs_state').eq('id', user_id).execute()
        if profile_res.data and profile_res.data[0].get('srs_state'):
            engine = SpacedRepetitionEngine.load_state(profile_res.data[0]['srs_state'])
        else:
            engine = SpacedRepetitionEngine(user_id)
        user_srs_engines[user_id] = engine
    return user_srs_engines[user_id]

# ================================================================================
# HELPER — Story generation (Gemini)
# ================================================================================

def generate_story_with_gemini_fallback(keywords: str, topic: str = "") -> str:
    """
    Two-step Architecture:
    1. Generator: Uses Gemini 3.1 Pro for high-quality Sinhala story writing.
    2. Formatter: Uses Gemini 2.5 Flash for lightning-fast JSON structuring.
    """
    print(f"✍️ Step 1: Writing beautiful Sinhala story using Gemini 3.1 Pro...")

    topic_instruction = f"මාතෘකාව: {topic}" if topic else ""

    # --- STEP 1: STORY GENERATION (GEMINI 3.1 PRO) ---
    story_prompt = f"""ඔබ ශ්‍රව්‍යාබාධිත කුඩා දරුවන්ට කතා ලියන විශේෂඥයෙකි.
පහත වචන අනිවාර්යයෙන්ම භාවිතා කර, වාක්‍ය 4-6 කින් යුත් ඉතා සරල, ලස්සන සිංහල කතාවක් ලියන්න. 
වචන: {keywords}
{topic_instruction}
(කරුණාකර කතාව පමණක් ලබා දෙන්න. වෙනත් කිසිදු විස්තරයක් අවශ්‍ය නැත)"""

    try:
        story_response = client.models.generate_content(
            model='gemini-3.1-pro-preview',
            contents=story_prompt,
            config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=1000)
        )
        raw_story = story_response.text.strip()

        if not raw_story or len(raw_story) < 20:
            raise ValueError("Story generation failed or too short.")

        print(f"✅ Story written successfully:\n{raw_story}\n")

    except Exception as e:
        print(f"❌ Step 1 (Pro Writer) Failed: {e}")
        return None

    # --- STEP 2: JSON FORMATTING (GEMINI 2.5 FLASH) ---
    print(f"⚡ Step 2: Formatting to JSON using Gemini 2.5 Flash...")

    format_prompt = f"""Here is a Sinhala story:
"{raw_story}"

Convert this into a quiz JSON format.
RULES:
1. Break into sentences.
2. Each sentence needs a "target_word" (pick an important noun/verb from the sentence).
3. Generate 3 phonetically similar Sinhala distractor words.
4. "options" = [target_word + 3 distractors] in random order.

JSON SCHEMA:
{{
  "story_sentences": [
    {{
      "text": "sentence text",
      "has_target_word": true,
      "target_word": "word",
      "options": ["opt1", "opt2", "opt3", "opt4"]
    }}
  ]
}}"""

    try:
        json_response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=format_prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json"
            )
        )
        json_text = json_response.text.strip()
        print("✅ JSON formatted successfully!")
        return json_text

    except Exception as e:
        print(f"❌ Step 2 (Flash Formatter) Failed: {e}")
        return None

# ================================================================================
# HELPER — Story generation (SinLlama on HuggingFace)
# ================================================================================

def generate_story_with_huggingface(keywords: str):
    """
    Call custom SinLlama model on Hugging Face Spaces.
    Falls back to Gemini if SinLlama is unavailable.

    Note: First request after Space sleep may take ~30 s (model loading).
    """
    api_url = "https://thulasika-n-sinllama-story-api.hf.space/api/predict"
    payload = {"data": [keywords, 250, 0.7]}

    print(f"🚀 Calling SinLlama Space with keywords: {keywords}")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            timeout = 120 if attempt == 0 else 60
            print(f"   Attempt {attempt + 1}/{max_retries} (timeout: {timeout}s)...")

            response = requests.post(api_url, json=payload, timeout=timeout)
            if response.status_code == 200:
                result = response.json()
                story = result.get("data", [None])[0]
                if story and len(story.strip()) > 10:
                    print("✅ Story generated from SinLlama Space!")
                    return story
                else:
                    print(f"⚠️ SinLlama returned empty/invalid story (attempt {attempt + 1})")
            else:
                print(f"❌ Space error (HTTP {response.status_code}): {response.text[:200]}")

            if attempt < max_retries - 1:
                time.sleep(10)

        except requests.Timeout:
            print(f"⏰ Timeout (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(10)
        except requests.ConnectionError as e:
            print(f"❌ Connection error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(10)
        except Exception as e:
            print(f"❌ Unexpected error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(10)

    # Fallback to Gemini
    print("⚠️ SinLlama unavailable — switching to Gemini fallback...")
    return generate_story_with_gemini_fallback(keywords)

# ================================================================================
# HELPER — Format story into game JSON
# ================================================================================

def format_story_for_game(raw_story_text: str, difficult_words: list):
    """
    Use Gemini to convert raw story text into structured game JSON.

    Each sentence becomes a question with:
      - text: the sentence
      - target_word: the word the child must identify
      - options: [correct + 3 phonetically similar distractors]
    """
    words_list = ", ".join(difficult_words)

    prompt = f"""You are formatting a Sinhala story for a hearing therapy game for hearing-impaired children.

RAW STORY: "{raw_story_text}"
TARGET WORDS: {words_list}

TASK: Convert this story into a structured JSON quiz format. EVERY sentence becomes a question.

RULES:
1. Break the story into individual sentences.
2. EVERY sentence MUST have has_target_word=true and a target_word with 4 answer options.
3. For sentences containing a TARGET WORD from the list above, use that target word.
4. For sentences WITHOUT a target word, pick the most important/concrete NOUN or VERB from that sentence as the target_word.
5. For each target_word, generate exactly 3 phonetically similar Sinhala distractor words.
   - Distractors should differ by 1-2 phonemes (voicing: ප→බ, ත→ද, ක→ග; place: ත→ට, ද→ඩ; nasality: ම→බ, න→ද)
   - At least one distractor should be a minimal pair (differ by exactly 1 phoneme)
   - All distractors must be real Sinhala words, not nonsense
6. The options array must contain the correct target_word plus the 3 distractors in SHUFFLED order.
7. RETURN ONLY VALID JSON. No markdown, no code blocks, no explanation.

JSON FORMAT:
{{
  "story_sentences": [
    {{
      "text": "Full original sentence text",
      "has_target_word": true,
      "target_word": "the_correct_word",
      "options": ["distractor1", "the_correct_word", "distractor2", "distractor3"]
    }}
  ]
}}"""

    response = client.models.generate_content(
        model='gemini-3.1-pro-preview',
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=4096
        )
    )
    raw_text = response.text.strip()

    # Robust JSON extraction — try multiple strategies
    clean_json = _extract_json(raw_text)
    if clean_json:
        return clean_json

    # If first attempt failed, retry once with stricter prompt
    print("⚠️ First format attempt produced unparseable JSON, retrying...")
    retry_prompt = (
        "The previous response was not valid JSON. "
        "Return ONLY the JSON object below, with NO markdown, NO explanation, NO extra text.\n\n"
        + prompt
    )
    try:
        response2 = client.models.generate_content(
            model='gemini-3.1-pro-preview',
            contents=retry_prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=4096
            )
        )
        raw_text2 = response2.text.strip()
        clean_json2 = _extract_json(raw_text2)
        if clean_json2:
            return clean_json2
    except Exception as e:
        print(f"⚠️ Format retry failed: {e}")

    # Last resort: return whatever we extracted, even if invalid
    print(f"❌ Could not extract valid JSON. Raw response (first 500 chars): {raw_text[:500]}")
    return raw_text


def _extract_json(raw_text: str) -> Optional[str]:
    """
    Try multiple strategies to extract valid JSON from AI response text.
    Returns the JSON string if parseable, or None.
    """
    # Strategy 1: Strip markdown fences
    clean = raw_text.replace("```json", "").replace("```", "").strip()

    # Strategy 2: Find outermost { ... }
    start = clean.find('{')
    end = clean.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidate = clean[start:end + 1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    # Strategy 3: Find outermost [ ... ] (array response)
    start_arr = clean.find('[')
    end_arr = clean.rfind(']')
    if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
        candidate = clean[start_arr:end_arr + 1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    # Strategy 4: Try the whole cleaned text
    try:
        json.loads(clean)
        return clean
    except json.JSONDecodeError:
        pass

    # Strategy 5: Remove common trailing corruption (incomplete last entry)
    if start != -1 and end != -1:
        candidate = clean[start:end + 1]
        # Try removing the last incomplete object in an array
        last_complete = candidate.rfind('},')
        if last_complete != -1:
            trimmed = candidate[:last_complete + 1] + ']}'
            try:
                json.loads(trimmed)
                print("⚠️ Recovered JSON by trimming incomplete last entry")
                return trimmed
            except json.JSONDecodeError:
                pass

    return None

# ================================================================================
# HELPER — Text-to-Speech (gTTS for Sinhala, Gemini for English fallback)
# ================================================================================

def generate_audio(text: str) -> bytes:
    """
    Generate speech audio using Gemini 2.5 Pro Preview TTS.

    Matches the proven working approach:
      - Pass raw text directly as contents (no prompt wrapping)
      - Only response_modalities=["AUDIO"] (no speech_config / voice_config)
      - Works for both Sinhala and English
      - Raw PCM output wrapped in WAV header (mono, 16-bit, 24000 Hz)

    Falls back to gTTS if Gemini TTS fails.
    Uses in-memory cache to avoid re-generating identical phrases.
    """
    # Check cache first
    cache_key = hashlib.md5(text.encode('utf-8')).hexdigest()
    if cache_key in tts_cache:
        print(f"🔊 TTS cache hit for: {text[:30]}...")
        return tts_cache[cache_key]

    audio_bytes = None

    # Primary: Gemini 2.5 Pro Preview TTS (works for Sinhala + English)
    try:
        response = client.models.generate_content(
            model='gemini-2.5-pro-preview-tts',
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"]
            )
        )
        if (response.candidates and response.candidates[0].content
                and response.candidates[0].content.parts):
            raw_pcm = response.candidates[0].content.parts[0].inline_data.data
            # Wrap raw PCM in WAV header (mono, 16-bit, 24000 Hz)
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(raw_pcm)
            wav_buffer.seek(0)
            audio_bytes = wav_buffer.read()
            print(f"🔊 Gemini TTS OK ({len(audio_bytes)} bytes): {text[:40]}...")
        else:
            raise ValueError("Gemini TTS returned no audio parts")
    except Exception as e:
        print(f"⚠️ Gemini TTS failed ({e}), trying gTTS fallback...")
        try:
            has_sinhala = any('\u0D80' <= ch <= '\u0DFF' for ch in text)
            tts = gTTS(text=text, lang='si' if has_sinhala else 'en', slow=False)
            mp3_buffer = io.BytesIO()
            tts.write_to_fp(mp3_buffer)
            mp3_buffer.seek(0)
            audio_bytes = mp3_buffer.read()
            print(f"🔊 gTTS fallback OK ({len(audio_bytes)} bytes): {text[:40]}...")
        except Exception as e2:
            print(f"❌ gTTS also failed: {e2}")
            raise

    # Store in cache (evict oldest if full)
    if len(tts_cache) >= TTS_CACHE_MAX:
        oldest_key = next(iter(tts_cache))
        del tts_cache[oldest_key]
    tts_cache[cache_key] = audio_bytes

    return audio_bytes


async def pregenerate_tts_for_story(sentences: list) -> dict:
    """
    Pre-generate TTS audio for all sentences using ThreadPoolExecutor.
    Returns dict mapping sentence text → base64 WAV string.
    """
    loop = asyncio.get_running_loop()
    results = {}

    async def gen_one(text):
        try:
            wav_bytes = await loop.run_in_executor(tts_executor, generate_audio, text)
            return text, base64.b64encode(wav_bytes).decode('utf-8')
        except Exception as e:
            print(f"⚠️ TTS pre-gen failed for '{text[:30]}...': {e}")
            return text, None

    tasks = [gen_one(s['text']) for s in sentences if s.get('text')]
    completed = await asyncio.gather(*tasks, return_exceptions=True)

    for item in completed:
        if isinstance(item, Exception):
            print(f"⚠️ TTS pre-gen task exception: {item}")
            continue
        text, audio_b64 = item
        if audio_b64:
            results[text] = audio_b64

    print(f"🔊 Pre-generated TTS for {len(results)}/{len(sentences)} sentences")
    return results

# ================================================================================
# HELPER — Smart word selection for story generation
# ================================================================================

def select_words_for_story(user_id: str, all_words: list, max_words: int = 5) -> list:
    """
    Pick an optimal subset of words for the next story, balancing:
      1. SRS due-words first (overdue interval → highest priority)
      2. Leitner box weight (Box 1 > Box 5)
      3. Low-accuracy words favoured
      4. Max 2 unseen words per story
      5. Random jitter to avoid repetitive selections
    """
    if len(all_words) <= max_words:
        return all_words

    try:
        engine = _get_or_create_srs(user_id)
    except Exception:
        engine = None

    scored_words = []
    unseen_words = []

    for word in all_words:
        if engine and word in engine.cards:
            card = engine.cards[word]
            accuracy = (card.correct_attempts / card.total_attempts) if card.total_attempts > 0 else 0.5

            # Days overdue (capped at 10)
            days_overdue = 0
            if card.next_review_at and card.next_review_at > 0:
                days_overdue = max(0, (time.time() - card.next_review_at) / 86400)

            box_weight = {1: 5.0, 2: 3.0, 3: 2.0, 4: 1.0, 5: 0.5}.get(card.leitner_box, 3.0)
            accuracy_weight = (1.0 - accuracy) * 4.0
            overdue_weight = min(days_overdue * 2.0, 10.0)
            jitter = random.uniform(0, 1.0)

            priority = box_weight + accuracy_weight + overdue_weight + jitter
            scored_words.append((word, priority))
        else:
            unseen_words.append(word)

    scored_words.sort(key=lambda x: x[1], reverse=True)

    max_new = min(2, max_words // 3, len(unseen_words))
    max_practiced = max_words - max_new

    selected = [w for w, _ in scored_words[:max_practiced]]

    if unseen_words and max_new > 0:
        new_picks = random.sample(unseen_words, min(max_new, len(unseen_words)))
        selected.extend(new_picks)

    if len(selected) < max_words and len(scored_words) > max_practiced:
        extras = [w for w, _ in scored_words[max_practiced:max_words]]
        selected.extend(extras)

    return selected[:max_words]

# ================================================================================
# ENDPOINT — Generate Story  (POST /generate-story)
# ================================================================================

@app.post("/generate-story")
async def get_story(request: StoryRequest):
    """
    Main story generation pipeline.

    Flow:
      1. Fetch child's target words from profile
      2. Smart-select a subset via SRS priority
      3. Generate raw story (Gemini → SinLlama fallback)
      4. Format into game JSON (sentences + options)
      5. Validate & save to DB
      6. Return structured story
    """
    try:
        # --- 1. Fetch profile words ---
        profile_res = supabase.table('profiles').select('learning_level, difficult_words').eq('id', request.user_id).execute()

        if not profile_res.data:
            difficult_words = ["හාවා", "ඉබ්බා", "කෑම"]
        else:
            raw_words = profile_res.data[0].get('difficult_words', [])
            if isinstance(raw_words, str):
                try:
                    raw_words = json.loads(raw_words) if raw_words else []
                except (json.JSONDecodeError, TypeError):
                    raw_words = []
            difficult_words = raw_words if raw_words else []
            if not difficult_words:
                difficult_words = ["පුස්තකාලය", "ආහාර", "විශ්වාසය"]

        # --- 2. Smart word selection ---
        difficult_words = select_words_for_story(request.user_id, difficult_words)
        keywords = ", ".join(difficult_words)
        print(f"📝 Generating story for: {keywords}")

        # --- 3. Generate story + quiz JSON in ONE call ---
        print("\n" + "=" * 60)
        print("STORY GENERATION PIPELINE (Gemini 3.1 Pro — single call)")
        print("=" * 60)

        game_json = generate_story_with_gemini_fallback(keywords, request.topic)

        if not game_json:
            key_status = "SET" if os.getenv("GOOGLE_API_KEY") else "MISSING"
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Story generation failed after all retries. "
                    f"GOOGLE_API_KEY is {key_status}. "
                    f"Check Render logs for detailed error messages."
                )
            )

        try:
            story_data = json.loads(game_json)
        except json.JSONDecodeError as parse_err:
            print(f"❌ JSON parse error: {parse_err}")
            print(f"❌ Raw game_json (first 500 chars): {game_json[:500] if game_json else 'EMPTY'}")
            raise HTTPException(
                status_code=500,
                detail="Failed to parse story JSON from AI. Please try again."
            )

        # Normalize shape
        if isinstance(story_data, list):
            story_data = {"story_sentences": story_data}
        elif isinstance(story_data, dict) and "story_sentences" not in story_data:
            story_data = {"story_sentences": []}

        # --- 5. Validate sentences ---
        validated = []
        for s in story_data.get('story_sentences', []):
            if not isinstance(s, dict) or not s.get('text'):
                continue
            s['has_target_word'] = True
            if not s.get('target_word'):
                continue
            opts = s.get('options', [])
            if not isinstance(opts, list) or len(opts) < 2:
                continue
            if s['target_word'] not in opts:
                opts[0] = s['target_word']
            s['options'] = opts
            validated.append(s)

        story_data['story_sentences'] = validated
        if not validated:
            raise HTTPException(status_code=500, detail="Story formatting produced no valid sentences")

        # --- 6. Pre-generate TTS for all sentences in parallel ---
        print("🔊 Pre-generating TTS audio for all sentences...")
        try:
            audio_map = await pregenerate_tts_for_story(validated)
        except Exception as e:
            print(f"⚠️ TTS pre-generation failed, frontend will fetch on-demand: {e}")
            audio_map = {}

        # --- 7. Save to DB & return ---
        full_text = " ".join([s['text'] for s in validated])

        story_insert_res = supabase.table('stories').insert({
            'user_id': request.user_id,
            'story_text': full_text,
            'question': 'ඔබට ඇහුණු වචනය කුමක්ද?',
            'options': [],
            'correct_answer': '',
            'target_words': difficult_words
        }).execute()

        inserted = story_insert_res.data
        if isinstance(inserted, list):
            new_story = inserted[0] if inserted else {}
        elif isinstance(inserted, dict):
            new_story = inserted
        else:
            new_story = {'id': 0, 'story_text': full_text}

        new_story['story_sentences'] = validated
        return {"story": new_story, "audio_map": audio_map}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ================================================================================
# ENDPOINT — Submit Answer  (POST /submit-answer)
# ================================================================================

@app.post("/submit-answer")
async def submit_answer(request: AnswerRequest):
    """Check child's answer; award 10 pts if correct."""
    try:
        story_res = supabase.table('stories').select('correct_answer').eq('id', request.story_id).execute()
        if not story_res.data:
            raise HTTPException(status_code=404, detail="Story not found.")

        correct_answer = story_res.data[0]['correct_answer']
        is_correct = request.answer == correct_answer
        feedback = "නිවැරදියි! (+10 ලකුණු!)" if is_correct else "නැවත උත්සාහ කරන්න!"

        if is_correct:
            profile_res = supabase.table('profiles').select('score').eq('id', request.user_id).execute()
            if profile_res.data:
                new_score = (profile_res.data[0].get('score', 0) or 0) + 10
                supabase.table('profiles').update({'score': new_score}).eq('id', request.user_id).execute()

        return {"correct": is_correct, "feedback": feedback}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in submit-answer: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ================================================================================
# ENDPOINT — Text-to-Speech  (POST /text-to-speech)
# ================================================================================

@app.post("/text-to-speech")
async def generate_speech(request: TTSRequest):
    """Convert Sinhala text to audio via Gemini TTS. Returns base64."""
    try:
        audio_bytes = generate_audio(request.text)
        # Detect format: WAV starts with RIFF header, otherwise MP3
        fmt = "wav" if audio_bytes[:4] == b'RIFF' else "mp3"
        return {"audio": base64.b64encode(audio_bytes).decode('utf-8'), "format": fmt}
    except Exception as e:
        print(f"Error in text-to-speech: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ================================================================================
# ENDPOINT — User Profile  (GET /profile/{user_id})
# ================================================================================

@app.get("/profile/{user_id}")
def get_profile(user_id: str, username: str = None):
    """Get or auto-create user profile. Accepts optional ?username= query param."""
    display_name = username or 'child'
    try:
        result = supabase.table('profiles').select('*').eq('id', user_id).single().execute()
        if result.data:
            # Update username if still default
            if username and result.data.get('username') in ('child', 'default_child', '', None):
                supabase.table('profiles').update({'username': display_name}).eq('id', user_id).execute()
                result.data['username'] = display_name
            return result.data
        # Create new profile
        supabase.table('profiles').insert({
            'id': user_id,
            'username': display_name,
            'score': 0,
            'learning_level': 1,
            'difficult_words': []
        }).execute()
        return {"id": user_id, "username": display_name, "score": 0, "learning_level": 1}
    except Exception:
        try:
            supabase.table('profiles').insert({
                'id': user_id,
                'username': display_name,
                'score': 0,
                'learning_level': 1,
                'difficult_words': []
            }).execute()
        except Exception:
            pass
        return {"id": user_id, "username": display_name, "score": 0, "learning_level": 1}

# ================================================================================
# ENDPOINT — Update Score  (POST /update-score)
# ================================================================================

@app.post("/update-score")
def update_score(request: UpdateScoreRequest):
    """Add stars × 10 to user's score."""
    try:
        profile_res = supabase.table('profiles').select('score').eq('id', request.user_id).single().execute()
        current = profile_res.data.get('score', 0) or 0 if profile_res.data else 0
        new_score = current + (request.stars * 10)
        supabase.table('profiles').update({'score': new_score}).eq('id', request.user_id).execute()
        return {"success": True, "new_score": new_score}
    except Exception as e:
        print(f"Score update error: {e}")
        return {"success": False, "error": str(e), "new_score": 0}

# ================================================================================
# ENDPOINTS — Word Management
# ================================================================================

@app.get("/words/{user_id}")
def get_words(user_id: str):
    """
    Get child's target words + per-word SRS stats.
    Returns both the therapist's word list and all practiced words.
    """
    try:
        profile_res = supabase.table('profiles').select('difficult_words').eq('id', user_id).single().execute()
        words = []
        if profile_res.data:
            raw = profile_res.data.get('difficult_words', [])
            if isinstance(raw, str):
                words = json.loads(raw) if raw else []
            else:
                words = raw or []

        # Gather SRS stats for all known words
        word_stats = {}
        all_practiced_words = []
        try:
            engine = _get_or_create_srs(user_id)
            all_practiced_words = list(engine.cards.keys())
            all_words_set = set(words) | set(all_practiced_words)

            for w in all_words_set:
                if w in engine.cards:
                    card = engine.cards[w]
                    word_stats[w] = {
                        "attempts": card.total_attempts,
                        "correct": card.correct_attempts,
                        "accuracy": round(card.correct_attempts / card.total_attempts * 100, 1) if card.total_attempts > 0 else 0,
                        "easiness": round(card.easiness_factor, 2),
                        "interval_days": card.interval_days,
                        "leitner_box": card.leitner_box,
                        "in_difficult_list": w in words,
                    }
                else:
                    word_stats[w] = {
                        "attempts": 0, "correct": 0, "accuracy": 0,
                        "easiness": 2.5, "interval_days": 0, "leitner_box": 1,
                        "in_difficult_list": w in words,
                    }
        except Exception:
            pass

        return {"words": words, "word_stats": word_stats, "practiced_words": all_practiced_words}
    except Exception as e:
        print(f"Get words error: {e}")
        return {"words": [], "word_stats": {}, "practiced_words": []}


@app.post("/words/add")
def add_word(request: WordManageRequest):
    """Add a target word to the child's list (therapist action)."""
    try:
        profile_res = supabase.table('profiles').select('difficult_words').eq('id', request.user_id).single().execute()
        words = []
        if profile_res.data:
            raw = profile_res.data.get('difficult_words', [])
            if isinstance(raw, str):
                words = json.loads(raw) if raw else []
            else:
                words = raw or []

        word = request.word.strip()
        if not word:
            raise HTTPException(status_code=400, detail="Word cannot be empty")
        if word in words:
            return {"success": True, "words": words, "message": "Word already exists"}

        words.append(word)
        supabase.table('profiles').update({'difficult_words': words}).eq('id', request.user_id).execute()
        return {"success": True, "words": words}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Add word error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/words/remove")
def remove_word(request: WordManageRequest):
    """Remove a target word from the child's list."""
    try:
        profile_res = supabase.table('profiles').select('difficult_words').eq('id', request.user_id).single().execute()
        words = []
        if profile_res.data:
            raw = profile_res.data.get('difficult_words', [])
            if isinstance(raw, str):
                words = json.loads(raw) if raw else []
            else:
                words = raw or []

        word = request.word.strip()
        if word in words:
            words.remove(word)

        supabase.table('profiles').update({'difficult_words': words}).eq('id', request.user_id).execute()
        return {"success": True, "words": words}
    except Exception as e:
        print(f"Remove word error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/words/progress/{user_id}")
def get_word_progress(user_id: str):
    """
    Detailed per-word progress for therapist/parent.

    For each word returns:
      mastery_level, accuracy, trend, recommendation, leitner_box, etc.

    Mastery levels (Leitner-based):
      Box 1 → struggling, Box 2-3 → learning, Box 4 → familiar, Box 5 → mastered

    Recommendations:
      - ready_to_remove: ≥80% accuracy + 5 encounters + Box 4+
      - needs_attention:  <50% accuracy + 3 encounters
      - keep_practicing:  everything else
    """
    try:
        profile_res = supabase.table('profiles').select('difficult_words, srs_state').eq('id', user_id).single().execute()
        therapist_words = []
        if profile_res.data:
            raw = profile_res.data.get('difficult_words', [])
            if isinstance(raw, str):
                therapist_words = json.loads(raw) if raw else []
            else:
                therapist_words = raw or []

        engine = _get_or_create_srs(user_id)

        # Recent per-word history from performance_logs
        perf_logs = supabase.table('performance_logs').select('word, is_correct, created_at').eq('user_id', user_id).order('created_at', desc=True).limit(500).execute()
        word_history = {}
        for log in (perf_logs.data or []):
            w = log.get('word')
            if not w:
                continue
            word_history.setdefault(w, []).append(log.get('is_correct', False))

        # word_mastery table
        mastery_data = supabase.table('word_mastery').select('*').eq('user_id', user_id).execute()
        mastery_map = {m['word']: m for m in (mastery_data.data or [])}

        all_words = set(therapist_words) | set(engine.cards.keys())
        word_progress = {}

        for w in all_words:
            card = engine.cards.get(w)
            mastery = mastery_map.get(w, {})
            history = word_history.get(w, [])

            if card:
                attempts = card.total_attempts
                correct = card.correct_attempts
                accuracy = round(correct / attempts * 100, 1) if attempts > 0 else 0
                leitner_box = card.leitner_box
                easiness = round(card.easiness_factor, 2)
                interval = card.interval_days
                mastery_level = {1: 'struggling', 2: 'learning', 3: 'learning', 4: 'familiar', 5: 'mastered'}.get(leitner_box, 'new')
                if attempts == 0:
                    mastery_level = 'new'

                recent = history[:5]
                recent_accuracy = round(sum(1 for r in recent if r) / len(recent) * 100, 1) if recent else accuracy

                # Trend: compare last-5 vs prior-5
                if len(history) >= 5:
                    r5 = sum(1 for r in history[:5] if r) / 5
                    o5 = sum(1 for r in history[5:10] if r) / max(1, min(5, len(history[5:10])))
                    trend = 'improving' if r5 - o5 > 0.15 else ('declining' if o5 - r5 > 0.15 else 'stable')
                else:
                    trend = 'stable'

                if accuracy >= 80 and attempts >= 5 and leitner_box >= 4:
                    recommendation = 'ready_to_remove'
                elif accuracy < 50 and attempts >= 3:
                    recommendation = 'needs_attention'
                else:
                    recommendation = 'keep_practicing'
            else:
                attempts = mastery.get('attempts', 0)
                correct = mastery.get('correct', 0)
                accuracy = round(correct / attempts * 100, 1) if attempts > 0 else 0
                leitner_box = 1
                easiness = 2.5
                interval = 0
                mastery_level = 'new'
                recent_accuracy = 0
                trend = 'stable'
                recommendation = 'keep_practicing'

            word_progress[w] = {
                'word': w,
                'in_therapist_list': w in therapist_words,
                'mastery_level': mastery_level,
                'accuracy': accuracy,
                'recent_accuracy': recent_accuracy,
                'attempts': attempts,
                'correct': correct,
                'leitner_box': leitner_box,
                'easiness': easiness,
                'interval_days': interval,
                'trend': trend,
                'recommendation': recommendation,
                'last_seen': mastery.get('last_seen_at'),
            }

        # Summary
        total = len(word_progress)
        mastered_count = sum(1 for p in word_progress.values() if p['mastery_level'] == 'mastered')
        struggling_count = sum(1 for p in word_progress.values() if p['mastery_level'] == 'struggling')

        return {
            'word_progress': word_progress,
            'summary': {
                'total_words': total,
                'mastered': mastered_count,
                'struggling': struggling_count,
                'ready_to_remove': [w for w, p in word_progress.items() if p['recommendation'] == 'ready_to_remove'],
                'needs_attention': [w for w, p in word_progress.items() if p['recommendation'] == 'needs_attention'],
                'overall_accuracy': round(sum(p['accuracy'] for p in word_progress.values()) / max(1, total), 1),
            }
        }
    except Exception as e:
        print(f"Word progress error: {e}")
        return {'word_progress': {}, 'summary': {'total_words': 0, 'mastered': 0, 'struggling': 0, 'ready_to_remove': [], 'needs_attention': [], 'overall_accuracy': 0}}

# ================================================================================
# ENDPOINTS — Spaced Repetition (SRS)
# ================================================================================

@app.post("/srs/add-word")
async def srs_add_word(request: SRSAddWordRequest):
    """Add a word to the spaced repetition deck."""
    try:
        engine = _get_or_create_srs(request.user_id)
        card = engine.add_word(
            word=request.word,
            phonemes=request.phonemes,
            difficulty_class=request.difficulty_class
        )
        # Persist SRS state
        state = engine.save_state()
        supabase.table('profiles').update({'srs_state': state}).eq('id', request.user_id).execute()
        return {"success": True, "card": card.to_dict()}
    except Exception as e:
        print(f"SRS add word error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/srs/review")
async def srs_review_word(request: SRSReviewRequest):
    """
    Submit a word review result.
    Quality scale: 0=blackout, 1=wrong-recalled, 2=wrong-easy,
                   3=correct-hard, 4=correct-hesitation, 5=perfect
    """
    try:
        engine = _get_or_create_srs(request.user_id)
        result = engine.review_word(
            word=request.word,
            quality=request.quality,
            response_time=request.response_time,
            confused_with=request.confused_with
        )
        # Persist SRS state
        state = engine.save_state()
        supabase.table('profiles').update({'srs_state': state}).eq('id', request.user_id).execute()

        # Sync to word_mastery table
        try:
            if request.word in engine.cards:
                card = engine.cards[request.word]
                supabase.table('word_mastery').upsert({
                    'user_id': request.user_id,
                    'word': request.word,
                    'mastery_score': round(card.easiness_factor * 100, 1),
                    'attempts': card.total_attempts,
                    'correct': card.correct_attempts,
                    'last_seen_at': datetime.now().isoformat()
                }, on_conflict='user_id,word').execute()
        except Exception as wm_err:
            print(f"Warning: Could not sync word_mastery: {wm_err}")

        return {"success": True, "review_result": result}
    except Exception as e:
        print(f"SRS review error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/srs/due-words/{user_id}")
async def srs_get_due_words(user_id: str, max_count: int = 10):
    """Get words due for review, sorted by priority."""
    try:
        engine = _get_or_create_srs(user_id)
        due_words = engine.get_due_words(max_count=max_count)
        return {
            "user_id": user_id,
            "due_count": len(due_words),
            "words": [card.to_dict() for card in due_words]
        }
    except Exception as e:
        print(f"SRS due words error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/srs/generate-session")
async def srs_generate_session(request: SRSSessionRequest):
    """Generate an optimal review session."""
    try:
        engine = _get_or_create_srs(request.user_id)
        session = engine.generate_review_session(
            max_words=request.max_words,
            include_new=request.include_new
        )
        return {
            "success": True,
            "session": session.to_dict(),
            "word_details": {
                word: engine.cards[word].to_dict()
                for word in session.words_to_review
                if word in engine.cards
            }
        }
    except Exception as e:
        print(f"SRS session error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/srs/statistics/{user_id}")
async def srs_get_statistics(user_id: str):
    """Get SRS statistics, forgetting curves, and phoneme mastery map."""
    try:
        engine = _get_or_create_srs(user_id)
        return {
            "user_id": user_id,
            "statistics": engine.get_statistics(),
            "forgetting_curves": engine.get_forgetting_curve_data(),
            "phoneme_mastery": engine.get_phoneme_mastery_map()
        }
    except Exception as e:
        print(f"SRS statistics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ================================================================================
# ENDPOINTS — Health & Testing
# ================================================================================

@app.get("/health")
def health_check():
    """
    Test all critical services: Gemini, SinLlama Space, Database.
    Returns { status: "healthy" | "degraded" | "unhealthy", services: {...} }.
    """
    health = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {}
    }

    # Gemini
    try:
        client.models.generate_content(model='gemini-3.1-pro-preview', contents='Test')
        health["services"]["gemini"] = {"status": "available", "message": "Gemini 3.1 Pro Preview responding"}
    except Exception as e:
        health["services"]["gemini"] = {"status": "unavailable", "error": str(e)}
        health["status"] = "degraded"

    # SinLlama
    try:
        r = requests.post(
            "https://thulasika-n-sinllama-story-api.hf.space/api/predict",
            json={"data": ["test", 10, 0.5]},
            timeout=30
        )
        if r.status_code == 200:
            health["services"]["sinllama"] = {"status": "available", "message": "SinLlama Space responding"}
        else:
            health["services"]["sinllama"] = {"status": "unavailable", "message": f"HTTP {r.status_code}"}
            health["status"] = "degraded"
    except requests.Timeout:
        health["services"]["sinllama"] = {"status": "sleeping", "message": "Space is sleeping (wakes on first request)"}
        health["status"] = "degraded"
    except Exception as e:
        health["services"]["sinllama"] = {"status": "unavailable", "error": str(e)}
        health["status"] = "degraded"

    # Database
    try:
        supabase.table('profiles').select('id').limit(1).execute()
        health["services"]["database"] = {"status": "available", "message": "Database OK"}
    except Exception as e:
        health["services"]["database"] = {"status": "unavailable", "error": str(e)}
        health["status"] = "unhealthy"

    return health


@app.get("/test-story-generation")
def test_story_generation(keywords: str = "හාවා, ඉබ්බා, කුකුළා"):
    """Quick test endpoint: GET /test-story-generation?keywords=හාවා,ඉබ්බා"""
    try:
        game_json = generate_story_with_gemini_fallback(keywords)
        if not game_json:
            return {"success": False, "error": "Gemini story generation failed", "keywords": keywords}

        difficult_words = [w.strip() for w in keywords.split(',')]

        try:
            story_data = json.loads(game_json)
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"JSON parsing failed: {e}", "raw_json": game_json}

        if isinstance(story_data, list):
            story_data = {"story_sentences": story_data}
        elif isinstance(story_data, dict) and "story_sentences" not in story_data:
            story_data = {"story_sentences": []}

        return {
            "success": True,
            "formatted_story": story_data,
            "keywords": difficult_words,
            "sentence_count": len(story_data.get('story_sentences', []))
        }
    except Exception as e:
        return {"success": False, "error": str(e), "keywords": keywords}


@app.get("/")
def read_root():
    """Root — basic API status."""
    return {
        "status": "Online",
        "model": "SinLlama-Story-Teller (Hugging Face)",
        "features": [
            "Sinhala Story Generation",
            "Gemini TTS",
            "Spaced Repetition (SM-2 + Leitner)",
            "Hand Gesture Recognition"
        ]
    }


@app.get("/system-info")
def get_system_info():
    """System overview."""
    return {
        "system": "Word-Weaver-Quest",
        "version": "2.0.0",
        "target": "Hearing-impaired Sinhala-speaking children (ages 4-12)",
        "features": [
            "Sinhala story generation (Gemini + SinLlama)",
            "Text-to-speech (Gemini TTS, Aoede voice)",
            "Spaced Repetition (SM-2 + Leitner boxes)",
            "Word management & progress tracking",
            "Hand gesture recognition (frontend)"
        ],
        "technology_stack": {
            "backend": "FastAPI (Python 3.11+)",
            "frontend": "React 19 + TailwindCSS",
            "database": "SQLite (local) / Supabase (production)",
            "llm": "Google Gemini 3.1 Pro Preview + Gemini 2.5 Pro TTS + SinLlama",
        }
    }

# ================================================================================
# END OF API
# ================================================================================

