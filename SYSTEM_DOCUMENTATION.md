# Word-Weaver-Quest: AI-Powered Word Memorization Game
## Comprehensive System Documentation

---

## 1. SYSTEM OVERVIEW

**Word-Weaver-Quest** is an AI-powered, interactive word memorization game designed for **Sinhala-speaking hearing-impaired children aged 4–12**. The system uses AI-generated stories as vehicles for vocabulary acquisition, embedding target difficult words into engaging narrative contexts. The child listens to a story sentence-by-sentence, then identifies a missing word from phonetically similar distractors — training both auditory discrimination and word memory.

> **Note:** This module focuses solely on **word memorization through stories**. Dashboards, clinical reports, and analytics are handled by other team members' modules within the SilentSpark ecosystem.

### 1.1 Core Therapeutic Principle

> **"Words are learned through repeated encounters in varied meaningful contexts"**
> — Nation (2001), *Learning Vocabulary in Another Language*

The system operationalizes this principle through:
- **AI story generation** that embeds target words in 5–8 sentence narratives
- **Repetitive exposure** — each target word appears in 2–3 different sentences within a story
- **Phoneme discrimination training** — distractors differ by 1–2 phonemes (voicing, place, nasality)
- **Spaced repetition** — the SM-2 algorithm schedules word review across sessions
- **Adaptive difficulty** — Thompson Sampling adjusts challenge level to the child's Zone of Proximal Development

### 1.2 Target Users

| Role | Usage |
|------|-------|
| **Child (4–12 yrs)** | Plays the story game, selects answers via click or hand gestures |
| **Therapist / Parent** | Adds difficult words the child needs to memorize, monitors per-word mastery |

---

## 2. ARCHITECTURE

```
┌────────────────────────────────────────────────────────┐
│                    FRONTEND (React 19)                  │
│  ┌──────────────────┐  ┌──────────────────────┐        │
│  │    Game Tab       │  │    Words Tab          │        │
│  │  (Story Play)     │  │  (Add/Remove Words,   │        │
│  │                   │  │   Per-word Mastery)    │        │
│  └────────┬──────────┘  └──────────┬────────────┘        │
│           │                        │                     │
│  ┌────────┴────────────────────────┴────────────────┐   │
│  │         API Layer (fetch → API_BASE_URL)        │   │
│  └────────────────────┬───────────────────────────┘   │
│                       │                                │
│  ┌────────────────────┴───────────────────────────┐   │
│  │  Computer Vision Layer (optional/background)    │   │
│  │  • MediaPipe Hand Gesture (finger counting)     │   │
│  │  • TF.js Face Mesh (gaze tracking)              │   │
│  │  • Emotion Detection                            │   │
│  └────────────────────────────────────────────────┘   │
└────────────────────────┬───────────────────────────────┘
                         │ HTTPS
┌────────────────────────┴───────────────────────────────┐
│                  BACKEND (FastAPI / Python)              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Story Engine │  │ Word Manager │  │ ML Features  │  │
│  │ • Gemini 3.1 │  │ • Add/Remove │  │ • Adaptive   │  │
│  │   Pro story  │  │ • Progress   │  │   Difficulty  │  │
│  │ • SinLlama   │  │ • SRS Engine │  │ • Engagement │  │
│  │   fallback   │  │              │  │ • Dropout    │  │
│  │ • Gemini TTS │  │              │  │ • Cognitive  │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         └────────────┬────┘─────────────────┘           │
│                ┌─────┴─────┐                            │
│                │  Database  │                            │
│                │  (SQLite / │                            │
│                │   Supabase)│                            │
│                └────────────┘                            │
└────────────────────────────────────────────────────────┘
```

### 2.1 Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | React 19.2 + TailwindCSS | UI, animations, game flow |
| **Backend** | FastAPI (Python 3.11+) | API, ML models, story generation |
| **LLM (Stories)** | Google Gemini 3.1 Pro Preview | Sinhala story generation & formatting |
| **LLM (TTS)** | Gemini 2.5 Pro Preview TTS | Expressive Sinhala text-to-speech (Aoede voice) |
| **LLM (Fallback)** | SinLlama (HuggingFace Spaces) | Custom fine-tuned Sinhala story model |
| **Database** | SQLite (dev) / Supabase PostgreSQL (prod) | Profiles, stories, analytics |
| **Computer Vision** | MediaPipe + TensorFlow.js | Hand gesture detection, eye gaze tracking |
| **ML Models** | Scikit-learn, XGBoost | Adaptive difficulty, dropout prediction, etc. |
| **Deployment** | Render (backend) + Vercel (frontend) | Cloud hosting |

---

## 3. USER JOURNEY (End-to-End Flow)

### 3.1 Session Flow

```
Child opens app → Profile loaded → Home Screen
                                        │
            ┌───────────────────────────┤
            │                           │
      [Pick Topic]              [Words Tab]
      (Animals, School,          Add/Remove
       Food, Nature,             therapy words
       Family, Festival)
            │
      [Start Story]
            │
      ┌─────┴─────┐
      │ Loading    │  ← Gemini generates 5-8 sentence story
      │ Animation  │  ← Gemini formats into quiz JSON
      └─────┬─────┘
            │
      ┌─────┴───────────────────────────┐
      │    STORY GAME LOOP              │
      │  ┌────────────────────────┐     │
      │  │ Phase 1: LISTEN        │     │
      │  │ • Angel reads sentence │     │
      │  │ • Gemini TTS audio     │     │
      │  │ • Child listens        │     │
      │  └──────────┬─────────────┘     │
      │             │                    │
      │  ┌──────────┴─────────────┐     │
      │  │ Phase 2: ANSWER        │     │
      │  │ • Sentence shown with  │     │
      │  │   blank (____) for     │     │
      │  │   target word          │     │
      │  │ • 4 options shown      │     │
      │  │   (1 correct +         │     │
      │  │    3 phonetic          │     │
      │  │    distractors)        │     │
      │  │ • Click or hand        │     │
      │  │   gesture selection    │     │
      │  └──────────┬─────────────┘     │
      │             │                    │
      │  ┌──────────┴─────────────┐     │
      │  │ Phase 3: FEEDBACK      │     │
      │  │ • ✅ Correct → star +  │     │
      │  │   celebration          │     │
      │  │ • ❌ Wrong → show      │     │
      │  │   correct answer       │     │
      │  │ • Angel speaks         │     │
      │  │   feedback in Sinhala  │     │
      │  └──────────┬─────────────┘     │
      │             │                    │
      │     [Next Sentence or Finish]    │
      └─────────────┬───────────────────┘
                    │
      ┌─────────────┴──────────────┐
      │ REWARD SCREEN              │
      │ • Stars earned / total     │
      │ • Percentage accuracy      │
      │ • Session saved to DB      │
      │ • Learning trajectory      │
      │   updated                  │
      └────────────────────────────┘
```

### 3.2 Data Flow Per Answer

When the child selects an answer, **7 parallel API calls** fire to record comprehensive data:

| # | API Endpoint | Data Recorded | ML Feature |
|---|-------------|---------------|-----------|
| 1 | `/session/record-answer` | word, correctness, response time | Session Analytics |
| 2 | `/update-performance` | user performance for adaptive difficulty | Adaptive Difficulty |
| 3 | `/track-engagement` | emotion, gesture accuracy, eye contact | Engagement Scoring |
| 4 | `/cognitive-load/record` | response time, hesitation, difficulty | Cognitive Load |
| 5 | `/srs/review` | SM-2 quality grade, word spacing | Spaced Repetition |
| 6 | `/track-gaze` | x,y gaze coordinates | Attention Heatmap |
| 7 | `/track-phoneme-confusion` | target vs selected word phonemes | Phoneme Analysis |

---

## 4. FRONTEND TABS (2 Tabs)

### 4.1 🎮 Game Tab (Main)

The core therapy experience. Children:
1. Choose a story topic (Animals, School, Food, Nature, Family, Festival)
2. Click "Start Story" → AI generates a personalized story using their target words
3. Listen → Answer → Get feedback for each sentence
4. Receive stars based on accuracy

**Key Components:**
- `SentenceBySentenceStory.js` — Core game loop (listen → answer → feedback)
- `StorytellingScene.js` — Immersive visual garden scene with Angel storyteller + animated children
- `HandGestureDetector.js` — MediaPipe finger counting for gesture-based answer selection
- `GazeTracker.js` — TF.js eye tracking for attention analytics
- `EngagementTracker.js` — Multimodal engagement scoring
- `CelebrationEffects.js` — Star/heart burst animations on correct answers

**Interaction Modes:**
- **Click Mode** (🖱️) — Child clicks answer option directly
- **Gesture Mode** (🖐️) — Child holds up 1–4 fingers for 2 seconds to select an answer

### 4.2 📝 Words Tab

Therapists and parents manage the child's target word list here.

**Features:**
- **Add words** — Type Sinhala words the child needs to practice
- **Remove words** — Remove mastered words from therapy list
- **Per-word progress** — Accuracy %, attempt count, correct count, Leitner box level
- **Mastery levels** — 🏆 Mastered (≥80%) → 📗 Familiar (60-80%) → 📙 Learning (40-60%) → 📕 Struggling (<40%) → 🆕 New
- **Trend indicators** — ↗ Improving, → Stable, ↘ Declining
- **Smart recommendations** — "✅ Ready to Remove" / "⚠️ Needs Attention" / "🔄 Keep Practicing"
- **View modes** — Therapist words, All practiced words, Words from stories
- **Sorting** — By priority, accuracy, or practice count

**Research basis:**
- Nation (2001): Word known after >80% accuracy + 5+ encounters
- Leitner (1972): Box system for graduated mastery

---

## 5. BACKEND ML FEATURES (10 Features)

### Feature 1: Adaptive Difficulty Engine
**Algorithm:** Thompson Sampling (Bayesian MAB)
**Purpose:** Automatically adjusts question difficulty to keep child in optimal learning zone.
**Research:** Settles & Meeder (2016), Kang (2020) — adaptive SRS for learning disabilities.
- Tracks per-difficulty-level success rates (Beta distributions)
- Selects difficulty maximizing information gain while avoiding frustration
- Adjusts in real-time as child answers questions

### Feature 2: Phoneme Confusion Matrix
**Algorithm:** Apriori Association Rules + frequency counting
**Purpose:** Maps which Sinhala phoneme pairs the child confuses (e.g., ප/බ, ත/ද, ක/ග).
**Research:** First Sinhala phoneme confusion dataset for hearing-impaired children.
- Tracks every incorrect answer's target vs. selected phonemes
- Builds per-user confusion matrix
- Generates therapy recommendations (which phoneme contrasts to drill)

### Feature 3: Multimodal Engagement Scorer
**Algorithm:** Weighted ensemble (LSTM temporal + real-time signals)
**Purpose:** Combines emotion detection, gesture accuracy, response time, and eye contact into a single engagement score (0–100).
**Research:** Multimodal learning analytics, Csikszentmihalyi flow theory.
- Triggers interventions: breaks for frustration, encouragement for low engagement, rewards for flow state
- Trend analysis: increasing, stable, declining

### Feature 4: Visual Attention Heatmap
**Algorithm:** Gaze point aggregation + fixation detection
**Purpose:** Maps where on screen the child looks, detecting attention drift.
**Research:** Eye-tracking in hearing-impaired education.
- Tracks gaze coordinates from TF.js Face Mesh
- Generates attention zone heatmaps
- Recommends UI optimizations (which screen areas to emphasize)

### Feature 5: Real-Time Dropout Prediction
**Algorithm:** 13-feature behavioral model (XGBoost)
**Purpose:** Predicts if child is about to quit session (30–60 second early warning).
**Signals:** Accuracy decline, consecutive errors, engagement drop, frustration episodes.
- Risk levels: low, medium, high, critical
- Auto-triggers interventions before dropout

### Feature 6: Hearing Loss Severity Estimator
**Algorithm:** 16-feature audiometric model
**Purpose:** Non-invasive WHO severity classification from behavioral patterns.
**Features:** Volume preference, audio replay rate, high/low frequency confusion rates.
- Output: Normal / Mild / Moderate / Severe / Profound
- WHO grade (A–E) with dB threshold range

### Feature 7: Session Analytics & Learning Trajectory
**Metrics:** Learning Efficiency Index (LEI), Flow State Ratio, ZPD Alignment, Resilience Score.
**Purpose:** Cross-session learning curve analysis and therapist recommendations.
- Tracks accuracy trend, engagement trend, learning rate
- ZPD recommendations: "increase difficulty" / "stay" / "decrease"

### Feature 8: Phoneme-Aware Spaced Repetition
**Algorithm:** SM-2 (Wozniak, 1990) + Leitner boxes + phoneme confusion integration.
**Purpose:** Schedules word review at optimal intervals for long-term retention.
**Research:** Nakata & Elgort (2021) — optimal SRS for vocabulary acquisition.
- Quality grades: 0 (blackout) to 5 (perfect recall)
- Leitner boxes 1–5 (promotion/demotion based on quality)
- Phoneme-aware: words confused with similar-sounding words reviewed together

### Feature 9: Real-Time Cognitive Load Monitor
**Algorithm:** Sweller's Cognitive Load Theory (CLT) differentiated estimation.
**Purpose:** Estimates intrinsic, extraneous, and germane cognitive load in real-time.
- Intrinsic: task difficulty (word complexity, phoneme count)
- Extraneous: interface friction (hesitation, help requests, audio replays)
- Germane: learning-relevant processing (correct responses, speed improvement)
- Alerts when cognitive load exceeds optimal zone

### Feature 10: AI-Powered Clinical Report Generator (Backend API Only)
**Purpose:** Generates comprehensive therapy reports for IEP documentation.
- Integrates all 9 other features into clinical narratives
- Three report types: Therapist, Parent, Research
- Bilingual output (Sinhala + English)
- IEP goal tracking with status (on_track, needs_attention, behind)
- **Note:** This feature is exposed as a backend API endpoint (`/generate-clinical-report`). The dashboard UI is handled by other team members' modules.

---

## 6. AI STORY GENERATION PIPELINE

### 6.1 Story Generation (Gemini 3.1 Pro Preview)

```
User's target words (from Words tab)
        │
        ▼
┌── Smart Word Selection ──┐
│ • SRS due words first    │
│ • Leitner boxes 1-2 get  │
│   priority              │
│ • Low-accuracy words     │
│   prioritized           │
│ • Max 2 new words/story  │
│ • Max 5 words per story  │
└───────────┬──────────────┘
            │
            ▼
┌── Story Generation Prompt ────────────────────────┐
│ "Write {5-8} sentences using these words: ..."     │
│ • Each word appears 2-3 times in different contexts│
│ • Even 1 word → full 5-8 sentence story            │
│ • Begin → Middle → End narrative structure          │
│ • Simple Sinhala, concrete nouns, action verbs      │
└───────────┬───────────────────────────────────────┘
            │
            ▼
┌── Story Formatting Prompt ────────────────────────┐
│ "Format this story into quiz JSON..."              │
│ • EVERY sentence becomes a question                │
│ • Each sentence: target word + 3 phonetic          │
│   distractors (minimal pairs, voicing changes)     │
│ • has_target_word = true for ALL sentences          │
│ • Options shuffled so correct isn't always first    │
└───────────┬───────────────────────────────────────┘
            │
            ▼
┌── Validation ─────────────────────────────────────┐
│ • Ensure every sentence has text, target_word,     │
│   options array with correct answer included       │
│ • Remove malformed sentences                        │
│ • Verify at least 1 valid sentence exists           │
└───────────────────────────────────────────────────┘
```

### 6.2 Text-to-Speech (Gemini 2.5 Pro Preview TTS)

- **Model:** `gemini-2.5-pro-preview-tts`
- **Voice:** Aoede (expressive, warm)
- **Prompt:** "Friendly storyteller reading to hearing-impaired children. Read slowly, clearly, with expressive, warm emotion."
- **Output:** 16-bit PCM WAV audio at 24000 Hz
- **Fallback:** Browser Web Speech API (si-LK locale)

### 6.3 SinLlama Fallback

Custom fine-tuned Sinhala story generation model deployed on HuggingFace Spaces (`thulasika-n/sinllama-story-api`). Used as primary generator with Gemini as fallback when SinLlama is sleeping or unavailable.

---

## 7. API ENDPOINTS

### Core Story & Game
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/generate-story` | Generate AI story with target words |
| POST | `/submit-answer` | Check answer correctness |
| POST | `/text-to-speech` | Gemini TTS (returns WAV base64) |
| POST | `/update-score` | Add earned stars to score |

### Word Management
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/words/{user_id}` | Get word list + SRS stats |
| GET | `/words/progress/{user_id}` | Per-word mastery, trend, recommendation |
| POST | `/words/add` | Add therapy word |
| POST | `/words/remove` | Remove therapy word |

### User Profile
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/profile/{user_id}` | Get/create user profile |

### Session Analytics (Feature 7)
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/session/record-answer` | Record answer for session tracking |
| POST | `/session/complete` | Close session, save analytics |
| GET | `/session/metrics/{user_id}` | Real-time or last session metrics |

### Adaptive Difficulty (Feature 1)
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/update-performance` | Submit performance data |
| POST | `/get-difficulty-recommendation` | Get next difficulty level |
| GET | `/user-progress/{user_id}` | View learning analytics |

### Phoneme Analysis (Feature 2)
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/track-phoneme-confusion` | Log phoneme errors |
| GET | `/phoneme-confusion-matrix/{user_id}` | View confusion heatmap |
| GET | `/phoneme-therapy-recommendations/{user_id}` | Get therapy plan |

### Engagement (Feature 3)
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/track-engagement` | Submit multimodal signals |
| GET | `/engagement-dashboard/{user_id}` | Real-time engagement |
| GET | `/predict-next-engagement/{user_id}` | LSTM prediction |

### Attention (Feature 4)
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/track-gaze` | Log gaze coordinates |
| GET | `/attention-heatmap/{user_id}` | Generate heatmap |

### Dropout Prevention (Feature 5)
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/predict-dropout` | Real-time dropout risk |

### Hearing Loss (Feature 6)
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/estimate-severity` | Estimate hearing loss |
| GET | `/severity-history/{user_id}` | Track over time |

### Spaced Repetition (Feature 8)
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/srs/statistics/{user_id}` | SRS stats |
| GET | `/srs/due-words/{user_id}` | Words due for review |
| POST | `/srs/generate-session` | Create review session |
| POST | `/srs/review` | Submit SM-2 review |
| POST | `/srs/add-word` | Add word to SRS deck |

### Cognitive Load (Feature 9)
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/cognitive-load/record` | Record cognitive load signal |
| GET | `/cognitive-load/status/{user_id}` | Current load estimate |

### Clinical Reports (Feature 10 — API for other team modules)
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/generate-clinical-report` | Generate therapist/parent/research report (consumed by other dashboards) |

### System
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | API status |
| GET | `/health` | Service health check |
| GET | `/system-info` | Full system info |
| GET | `/test-story-generation` | Test story pipeline |
| GET | `/learning-trajectory/{user_id}` | Cross-session trajectory |

---

## 8. RESEARCH FOUNDATIONS

### Theoretical Framework

| Theory | Author(s) | Application in System |
|--------|----------|----------------------|
| **Incidental Vocabulary Acquisition** | Nation (2001) | Words learned through repeated story encounters |
| **Spaced Repetition** | Wozniak (1990), Ebbinghaus (1885) | SM-2 algorithm for optimal word review intervals |
| **Leitner System** | Leitner (1972) | Box-based mastery classification (1–5) |
| **Zone of Proximal Development** | Vygotsky (1978) | Adaptive difficulty keeps child in optimal challenge zone |
| **Cognitive Load Theory** | Sweller (2020) | Monitor intrinsic/extraneous/germane load in real-time |
| **Thompson Sampling** | Thompson (1933) | Bayesian bandit for adaptive difficulty selection |
| **Flow Theory** | Csikszentmihalyi (1990) | Engagement optimization via challenge-skill balance |
| **Productive Failure** | Kapur (2020) | Resilience scoring — errors as learning opportunities |
| **Multimodal Learning Analytics** | Ochoa (2017) | Combine gesture, gaze, emotion for holistic engagement |
| **WHO Hearing Classification** | WHO (2021) | Severity estimation from behavioral patterns |
| **Phoneme Confusion in HI** | Knoors & Marschark (2020) | Phonetically similar distractors for discrimination training |
| **Vocabulary Acquisition via SRS** | Nakata & Elgort (2021) | Spaced repetition effectiveness for HI word learning |
| **SRS for Rehabilitation** | Kang (2020) | SRS adaptations for learning disabilities |
| **Repetition in Context** | Webb (2007) | Multiple encounters in varied contexts deepens word knowledge |
| **Trainable SRS** | Settles & Meeder (2016) | Machine learning to optimize review scheduling |

### Word Mastery Criteria (Nation, 2001)

A word is considered **mastered** when:
- ≥80% accuracy across all encounters
- ≥5 total encounters (attempts)
- Leitner box ≥4 (promoted through successful reviews)

The system's "Ready to Remove" recommendation triggers when all three criteria are met.

---

## 9. DATABASE SCHEMA

### profiles
| Column | Type | Description |
|--------|------|-------------|
| id | UUID | User ID |
| username | text | Display name |
| score | integer | Total stars earned |
| difficult_words | jsonb | Therapist's target word list |
| learning_level | integer | Current difficulty level |
| srs_state | jsonb | Serialized SM-2 engine state |
| trajectory_state | jsonb | Cross-session learning trajectory |

### stories  
| Column | Type | Description |
|--------|------|-------------|
| id | serial | Story ID |
| user_id | UUID | Who this story was for |
| story_text | text | Full story text |
| target_words | jsonb | Words used in this story |
| question | text | Question template |
| options | jsonb | Answer options |
| correct_answer | text | Correct word |

### session_analytics
| Column | Type | Description |
|--------|------|-------------|
| user_id | UUID | Child ID |
| session_id | text | Session identifier |
| summary | jsonb | Full session metrics |
| research_metrics | jsonb | LEI, flow, ZPD, resilience |
| timestamp | float | Unix timestamp |

### therapy_sessions
| Column | Type | Description |
|--------|------|-------------|
| id | text | Session ID |
| user_id | UUID | Child ID |
| ended_at | timestamp | Session end time |
| total_questions | integer | Questions answered |
| correct_answers | integer | Correct count |
| average_response_time | float | Avg response seconds |
| average_engagement_score | float | Avg engagement (0–100) |
| dropout | boolean | Did child quit early? |

### word_mastery
| Column | Type | Description |
|--------|------|-------------|
| user_id | UUID | Child ID |
| word | text | Sinhala word |
| mastery_score | float | SM-2 easiness × 100 |
| attempts | integer | Total encounters |
| correct | integer | Correct count |
| last_seen_at | timestamp | Last review time |

---

## 10. DEPLOYMENT

### Backend (Render)
- **Runtime:** Python 3.11
- **Entry:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Build:** `pip install -r requirements.txt`
- **Environment Variables:** `GOOGLE_API_KEY`

### Frontend (Vercel)
- **Framework:** Create React App
- **Build:** `npm run build`
- **Environment Variables:** `REACT_APP_API_URL`

### Required API Keys
| Key | Service | Purpose |
|-----|---------|---------|
| `GOOGLE_API_KEY` | Google AI Studio | Gemini 3.1 Pro (stories) + Gemini 2.5 Pro TTS |
| `HF_TOKEN` | Hugging Face | SinLlama model access |

---

## 11. KEY DEPENDENCIES

### Backend (requirements.txt)
```
fastapi                  # API framework
uvicorn[standard]       # ASGI server
google-genai            # Google Gemini SDK (new unified SDK)
python-dotenv           # Environment variables
requests                # HTTP client (SinLlama calls)
scikit-learn            # ML models
xgboost                 # Dropout prediction, severity estimation
numpy, scipy, pandas    # Data processing
matplotlib, seaborn     # Visualization (reports)
reportlab               # PDF report generation
mlxtend                 # Association rules (phoneme analysis)
imbalanced-learn        # Handling imbalanced datasets
```

### Frontend (package.json)
```
react                   # UI framework
tailwindcss             # Utility-first CSS
@mediapipe/tasks-vision # Hand gesture detection
@tensorflow/tfjs        # Eye gaze tracking
```
