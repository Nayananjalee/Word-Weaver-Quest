
-- ============================================================================
-- PHASE 1: Adaptive Difficulty Engine Tables
-- ============================================================================

-- Add adaptive_state and engagement_state columns to profiles table
ALTER TABLE profiles 
ADD COLUMN IF NOT EXISTS adaptive_state JSONB;

ALTER TABLE profiles 
ADD COLUMN IF NOT EXISTS engagement_state JSONB;

-- Performance logs table (tracks every question attempt)
DROP TABLE IF EXISTS performance_logs CASCADE;

CREATE TABLE performance_logs (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    story_id INTEGER,
    is_correct BOOLEAN NOT NULL,
    response_time FLOAT NOT NULL,
    engagement_score FLOAT,
    difficulty_level INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for faster queries
CREATE INDEX idx_performance_logs_user_id ON performance_logs(user_id);
CREATE INDEX idx_performance_logs_created_at ON performance_logs(created_at);

-- ============================================================================
-- PHASE 2: Phoneme Confusion Matrix Tables
-- ============================================================================

-- Phoneme error tracking (for confusion matrix)
DROP TABLE IF EXISTS phoneme_errors CASCADE;

CREATE TABLE phoneme_errors (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    phoneme_pair VARCHAR(10) NOT NULL,  -- e.g., "ප-බ"
    error_count INTEGER DEFAULT 1,
    last_error_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, phoneme_pair)
);

CREATE INDEX idx_phoneme_errors_user_id ON phoneme_errors(user_id);

-- ============================================================================
-- PHASE 3: Engagement & Session Tracking
-- ============================================================================

-- Engagement logs (for multimodal engagement tracking)
DROP TABLE IF EXISTS engagement_logs CASCADE;

CREATE TABLE engagement_logs (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    session_id UUID NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    engagement_score FLOAT,
    response_time FLOAT,
    gesture_quality FLOAT
);

CREATE INDEX idx_engagement_logs_session_id ON engagement_logs(session_id);
CREATE INDEX idx_engagement_logs_user_id ON engagement_logs(user_id);

-- ============================================================================
-- PHASE 4: Dropout Prevention
-- ============================================================================

-- Dropout predictions (for ML-based intervention)
DROP TABLE IF EXISTS dropout_predictions CASCADE;

CREATE TABLE dropout_predictions (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    predicted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    dropout_probability FLOAT,
    intervention_triggered BOOLEAN DEFAULT FALSE,
    actual_dropout BOOLEAN  -- Filled after session ends
);

CREATE INDEX idx_dropout_predictions_session_id ON dropout_predictions(session_id);

-- ============================================================================
-- PHASE 5: Session Management
-- ============================================================================

-- Sessions table (tracks complete therapy sessions)
-- Drop and recreate to ensure schema is correct
DROP TABLE IF EXISTS therapy_sessions CASCADE;

CREATE TABLE therapy_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ended_at TIMESTAMP WITH TIME ZONE,
    total_questions INTEGER DEFAULT 0,
    correct_answers INTEGER DEFAULT 0,
    average_response_time FLOAT,
    average_engagement_score FLOAT,
    difficulty_level_start INTEGER,
    difficulty_level_end INTEGER,
    dropout BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_therapy_sessions_user_id ON therapy_sessions(user_id);
CREATE INDEX idx_therapy_sessions_started_at ON therapy_sessions(started_at);

-- ============================================================================
-- PHASE 6: Progress Reports & Analytics
-- ============================================================================

-- Weekly progress summaries (for automated reports)
DROP TABLE IF EXISTS weekly_summaries CASCADE;

CREATE TABLE weekly_summaries (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    week_start_date DATE NOT NULL,
    week_end_date DATE NOT NULL,
    total_sessions INTEGER,
    total_questions INTEGER,
    accuracy_rate FLOAT,
    average_engagement FLOAT,
    most_confused_phonemes JSONB,
    improvement_areas JSONB,
    report_generated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, week_start_date)
);

CREATE INDEX idx_weekly_summaries_user_id ON weekly_summaries(user_id);

-- ============================================================================
-- PHASE 7: Peer Benchmarking
-- ============================================================================

-- Anonymized peer statistics (for benchmarking)
DROP TABLE IF EXISTS peer_benchmarks CASCADE;

CREATE TABLE peer_benchmarks (
    id BIGSERIAL PRIMARY KEY,
    age_group VARCHAR(10),  -- e.g., "6-7", "8-9"
    hearing_loss_severity VARCHAR(20),  -- mild, moderate, severe
    average_accuracy FLOAT,
    average_sessions_per_week FLOAT,
    median_difficulty_level INTEGER,
    percentile_25_accuracy FLOAT,
    percentile_50_accuracy FLOAT,
    percentile_75_accuracy FLOAT,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- SAMPLE DATA & VIEWS
-- ============================================================================

-- View: User Performance Dashboard
CREATE OR REPLACE VIEW user_performance_dashboard AS
SELECT 
    p.user_id,
    COUNT(p.id) as total_attempts,
    SUM(CASE WHEN p.is_correct THEN 1 ELSE 0 END)::FLOAT / COUNT(p.id) as accuracy,
    AVG(p.response_time) as avg_response_time,
    AVG(p.engagement_score) as avg_engagement,
    MAX(p.difficulty_level) as highest_difficulty_reached,
    MAX(p.created_at) as last_activity
FROM performance_logs p
GROUP BY p.user_id;

-- View: Recent Session Stats
CREATE OR REPLACE VIEW recent_session_stats AS
SELECT 
    ts.user_id,
    ts.id as session_id,
    ts.started_at,
    ts.ended_at,
    ts.total_questions,
    ts.correct_answers,
    CASE 
        WHEN ts.total_questions > 0 THEN ts.correct_answers::FLOAT / ts.total_questions 
        ELSE 0 
    END as session_accuracy,
    ts.average_engagement_score,
    ts.dropout
FROM therapy_sessions ts
ORDER BY ts.started_at DESC;

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to calculate user's percentile rank
CREATE OR REPLACE FUNCTION get_user_percentile_rank(target_user_id UUID)
RETURNS TABLE(accuracy_percentile FLOAT, engagement_percentile FLOAT) AS $$
BEGIN
    RETURN QUERY
    WITH user_stats AS (
        SELECT 
            user_id,
            SUM(CASE WHEN is_correct THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as accuracy,
            AVG(engagement_score) as engagement
        FROM performance_logs
        WHERE created_at >= NOW() - INTERVAL '30 days'
        GROUP BY user_id
    )
    SELECT 
        PERCENT_RANK() OVER (ORDER BY accuracy) as accuracy_percentile,
        PERCENT_RANK() OVER (ORDER BY engagement) as engagement_percentile
    FROM user_stats
    WHERE user_id = target_user_id;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- CLEANUP (Run only if resetting)
-- ============================================================================

-- Uncomment below to drop all new tables (CAUTION: deletes all data!)
/*
DROP TABLE IF EXISTS performance_logs CASCADE;
DROP TABLE IF EXISTS phoneme_errors CASCADE;
DROP TABLE IF EXISTS engagement_logs CASCADE;
DROP TABLE IF EXISTS dropout_predictions CASCADE;
DROP TABLE IF EXISTS therapy_sessions CASCADE;
DROP TABLE IF EXISTS weekly_summaries CASCADE;
DROP TABLE IF EXISTS peer_benchmarks CASCADE;
DROP VIEW IF EXISTS user_performance_dashboard;
DROP VIEW IF EXISTS recent_session_stats;
DROP FUNCTION IF EXISTS get_user_percentile_rank(UUID);
*/
