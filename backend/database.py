"""
PostgreSQL Database Layer for Word-Weaver-Quest (Neon)
======================================================
Persistent PostgreSQL database hosted on Neon.
Provides a simple interface compatible with the existing codebase patterns.
"""

import json
import os
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
from psycopg2.extras import RealDictCursor, Json
from psycopg2.pool import ThreadedConnectionPool

# ---------------------------------------------------------------------------
# Connection URL — set DATABASE_URL env-var on Render / local .env
# ---------------------------------------------------------------------------
DATABASE_URL = os.environ["DATABASE_URL"]

# ---------------------------------------------------------------------------
# Connection pool (lazy-initialised)
# ---------------------------------------------------------------------------
_pool = None


def _get_pool():
    global _pool
    if _pool is None:
        _pool = ThreadedConnectionPool(1, 10, DATABASE_URL)
    return _pool


@contextmanager
def get_connection():
    """Thread-safe pooled connection context manager."""
    pool = _get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def init_db():
    """Create all tables if they don't exist. Called once at startup."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
            -- Profiles table
            CREATE TABLE IF NOT EXISTS profiles (
                id TEXT PRIMARY KEY,
                username TEXT DEFAULT '',
                score INTEGER DEFAULT 0,
                learning_level INTEGER DEFAULT 1,
                difficult_words JSONB DEFAULT '[]'::jsonb,
                adaptive_state JSONB,
                engagement_state JSONB,
                phoneme_state JSONB,
                attention_state JSONB,
                trajectory_state JSONB,
                srs_state JSONB,
                cognitive_load_state JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );

            -- Stories table
            CREATE TABLE IF NOT EXISTS stories (
                id SERIAL PRIMARY KEY,
                user_id TEXT REFERENCES profiles(id) ON DELETE CASCADE,
                story_text TEXT,
                question TEXT DEFAULT '',
                correct_answer TEXT,
                options JSONB,
                difficulty_level INTEGER DEFAULT 1,
                topic TEXT DEFAULT '',
                created_at TIMESTAMPTZ DEFAULT NOW()
            );

            -- Performance logs
            CREATE TABLE IF NOT EXISTS performance_logs (
                id SERIAL PRIMARY KEY,
                user_id TEXT REFERENCES profiles(id) ON DELETE CASCADE,
                story_id INTEGER,
                is_correct BOOLEAN NOT NULL DEFAULT FALSE,
                response_time DOUBLE PRECISION NOT NULL DEFAULT 0,
                engagement_score DOUBLE PRECISION,
                difficulty_level INTEGER,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );

            -- Phoneme errors
            CREATE TABLE IF NOT EXISTS phoneme_errors (
                id SERIAL PRIMARY KEY,
                user_id TEXT REFERENCES profiles(id) ON DELETE CASCADE,
                phoneme_pair TEXT NOT NULL,
                error_count INTEGER DEFAULT 1,
                last_error_at TIMESTAMPTZ DEFAULT NOW(),
                created_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(user_id, phoneme_pair)
            );

            -- Engagement logs
            CREATE TABLE IF NOT EXISTS engagement_logs (
                id SERIAL PRIMARY KEY,
                user_id TEXT REFERENCES profiles(id) ON DELETE CASCADE,
                session_id TEXT NOT NULL,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                engagement_score DOUBLE PRECISION,
                response_time DOUBLE PRECISION,
                gesture_quality DOUBLE PRECISION
            );

            -- Dropout predictions
            CREATE TABLE IF NOT EXISTS dropout_predictions (
                id SERIAL PRIMARY KEY,
                session_id TEXT,
                user_id TEXT REFERENCES profiles(id) ON DELETE CASCADE,
                predicted_at TIMESTAMPTZ DEFAULT NOW(),
                dropout_probability DOUBLE PRECISION,
                risk_level TEXT,
                intervention_type TEXT,
                contributing_factors JSONB,
                session_features JSONB,
                intervention_triggered BOOLEAN DEFAULT FALSE,
                actual_dropout BOOLEAN,
                timestamp DOUBLE PRECISION
            );

            -- Therapy sessions
            CREATE TABLE IF NOT EXISTS therapy_sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT REFERENCES profiles(id) ON DELETE CASCADE,
                started_at TIMESTAMPTZ DEFAULT NOW(),
                ended_at TIMESTAMPTZ,
                total_questions INTEGER DEFAULT 0,
                correct_answers INTEGER DEFAULT 0,
                average_response_time DOUBLE PRECISION,
                average_engagement_score DOUBLE PRECISION,
                difficulty_level_start INTEGER,
                difficulty_level_end INTEGER,
                dropout BOOLEAN DEFAULT FALSE
            );

            -- Severity estimates
            CREATE TABLE IF NOT EXISTS severity_estimates (
                id SERIAL PRIMARY KEY,
                user_id TEXT REFERENCES profiles(id) ON DELETE CASCADE,
                estimated_severity TEXT,
                severity_category TEXT,
                estimated_threshold_db DOUBLE PRECISION,
                threshold_range_lower DOUBLE PRECISION,
                threshold_range_upper DOUBLE PRECISION,
                confidence DOUBLE PRECISION,
                features JSONB,
                details JSONB,
                timestamp DOUBLE PRECISION,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );

            -- Session analytics (Feature 7)
            CREATE TABLE IF NOT EXISTS session_analytics (
                id SERIAL PRIMARY KEY,
                user_id TEXT REFERENCES profiles(id) ON DELETE CASCADE,
                session_id TEXT,
                total_questions INTEGER DEFAULT 0,
                correct_answers INTEGER DEFAULT 0,
                accuracy DOUBLE PRECISION,
                avg_response_time DOUBLE PRECISION,
                learning_efficiency_index DOUBLE PRECISION,
                flow_ratio DOUBLE PRECISION,
                zpd_alignment DOUBLE PRECISION,
                resilience_score DOUBLE PRECISION,
                engagement_consistency DOUBLE PRECISION,
                attention_quality_index DOUBLE PRECISION,
                streak_max INTEGER DEFAULT 0,
                frustration_events INTEGER DEFAULT 0,
                boredom_events INTEGER DEFAULT 0,
                topic TEXT,
                difficulty_start INTEGER,
                difficulty_end INTEGER,
                summary JSONB,
                research_metrics JSONB,
                timestamp DOUBLE PRECISION,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );

            -- Word mastery
            CREATE TABLE IF NOT EXISTS word_mastery (
                id SERIAL PRIMARY KEY,
                user_id TEXT REFERENCES profiles(id) ON DELETE CASCADE,
                word TEXT NOT NULL,
                mastery_score DOUBLE PRECISION DEFAULT 500.0,
                attempts INTEGER DEFAULT 0,
                correct INTEGER DEFAULT 0,
                last_seen_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(user_id, word)
            );

            -- Weekly summaries
            CREATE TABLE IF NOT EXISTS weekly_summaries (
                id SERIAL PRIMARY KEY,
                user_id TEXT REFERENCES profiles(id) ON DELETE CASCADE,
                week_start_date TEXT,
                week_end_date TEXT,
                total_sessions INTEGER,
                total_questions INTEGER,
                accuracy_rate DOUBLE PRECISION,
                average_engagement DOUBLE PRECISION,
                most_confused_phonemes JSONB,
                improvement_areas JSONB,
                report_generated_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(user_id, week_start_date)
            );

            -- Insert default user profile if it doesn't exist
            INSERT INTO profiles (id, username, score, learning_level)
            VALUES ('123e4567-e89b-12d3-a456-426614174000', 'default_child', 0, 1)
            ON CONFLICT (id) DO NOTHING;
            """)
    print("[DB] Neon PostgreSQL database initialized")


# =====================================================
# HELPER CLASS: Drop-in replacement for Supabase client
# =====================================================

# Map table → unique/primary key columns for ON CONFLICT in upsert
_UPSERT_CONFLICT_TARGETS = {
    'profiles': '(id)',
    'phoneme_errors': '(user_id, phoneme_pair)',
    'word_mastery': '(user_id, word)',
    'weekly_summaries': '(user_id, week_start_date)',
    'therapy_sessions': '(id)',
}

# Columns that hold JSONB data (for automatic Json wrapping)
_JSONB_COLUMNS = frozenset([
    'difficult_words', 'adaptive_state', 'engagement_state',
    'phoneme_state', 'attention_state', 'trajectory_state',
    'srs_state', 'cognitive_load_state', 'options', 'details',
    'most_confused_phonemes', 'improvement_areas',
    'contributing_factors', 'session_features', 'features',
    'research_metrics', 'summary',
])


class DBResult:
    """Mimics Supabase response object with .data attribute."""
    def __init__(self, data):
        self.data = data if data is not None else []
        self.count = len(self.data) if isinstance(self.data, list) else (1 if self.data else 0)


class QueryBuilder:
    """
    Fluent query builder that mimics Supabase's chained API:
        db.table('profiles').select('*').eq('id', uid).execute()
    """

    def __init__(self, table_name):
        self._table = table_name
        self._operation = None   # 'select', 'insert', 'update', 'upsert', 'delete'
        self._columns = '*'
        self._filters = []       # list of (column, op, value)
        self._order_col = None
        self._order_desc = False
        self._limit_val = None
        self._single = False
        self._data = None
        self._on_conflict = None

    # --- Operations ---
    def select(self, columns='*'):
        self._operation = 'select'
        self._columns = columns
        return self

    def insert(self, data):
        self._operation = 'insert'
        self._data = data
        return self

    def update(self, data):
        self._operation = 'update'
        self._data = data
        return self

    def upsert(self, data, on_conflict=None):
        self._operation = 'upsert'
        self._data = data
        self._on_conflict = on_conflict
        return self

    def delete(self):
        self._operation = 'delete'
        return self

    # --- Filters ---
    def eq(self, column, value):
        self._filters.append((column, '=', value))
        return self

    def neq(self, column, value):
        self._filters.append((column, '!=', value))
        return self

    def gt(self, column, value):
        self._filters.append((column, '>', value))
        return self

    def gte(self, column, value):
        self._filters.append((column, '>=', value))
        return self

    def lt(self, column, value):
        self._filters.append((column, '<', value))
        return self

    def lte(self, column, value):
        self._filters.append((column, '<=', value))
        return self

    # --- Modifiers ---
    def order(self, column, desc=False):
        self._order_col = column
        self._order_desc = desc
        return self

    def limit(self, n):
        self._limit_val = n
        return self

    def single(self):
        self._single = True
        self._limit_val = 1
        return self

    # --- Execute ---
    def execute(self):
        with get_connection() as conn:
            if self._operation == 'select':
                return self._exec_select(conn)
            elif self._operation == 'insert':
                return self._exec_insert(conn)
            elif self._operation == 'update':
                return self._exec_update(conn)
            elif self._operation == 'upsert':
                return self._exec_upsert(conn)
            elif self._operation == 'delete':
                return self._exec_delete(conn)

    # --- Internal helpers ---

    def _build_where(self):
        if not self._filters:
            return "", []
        clauses = []
        params = []
        for col, op, val in self._filters:
            clauses.append(f"{col} {op} %s")
            params.append(val)
        return " WHERE " + " AND ".join(clauses), params

    @staticmethod
    def _wrap_value(key, value):
        """Wrap dicts/lists as psycopg2 Json for JSONB columns."""
        if isinstance(value, (dict, list)):
            return Json(value)
        return value

    def _exec_select(self, conn):
        cols = self._columns.replace(' ', '')
        sql = f"SELECT {cols} FROM {self._table}"
        where, params = self._build_where()
        sql += where
        if self._order_col:
            direction = "DESC" if self._order_desc else "ASC"
            sql += f" ORDER BY {self._order_col} {direction}"
        if self._limit_val:
            sql += f" LIMIT {self._limit_val}"

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = [dict(r) for r in cur.fetchall()]

        if self._single:
            return DBResult(rows[0] if rows else None)
        return DBResult(rows)

    def _exec_insert(self, conn):
        data = self._data
        if isinstance(data, dict):
            data = [data]

        results = []
        for record in data:
            processed = {k: self._wrap_value(k, v) for k, v in record.items()}

            columns = ', '.join(processed.keys())
            placeholders = ', '.join(['%s'] * len(processed))
            values = list(processed.values())

            sql = f"INSERT INTO {self._table} ({columns}) VALUES ({placeholders}) RETURNING *"
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, values)
                row = cur.fetchone()
                if row:
                    results.append(dict(row))

        return DBResult(results[0] if len(results) == 1 else results)

    def _exec_update(self, conn):
        processed = {k: self._wrap_value(k, v) for k, v in self._data.items()}

        set_clause = ', '.join([f"{k} = %s" for k in processed.keys()])
        set_values = list(processed.values())

        where, where_params = self._build_where()
        sql = f"UPDATE {self._table} SET {set_clause}{where} RETURNING *"
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, set_values + where_params)
            rows = [dict(r) for r in cur.fetchall()]
        return DBResult(rows)

    def _exec_upsert(self, conn):
        data = self._data
        if isinstance(data, dict):
            data = [data]

        # Determine conflict target
        if self._on_conflict:
            conflict_cols = f"({self._on_conflict})"
        else:
            conflict_cols = _UPSERT_CONFLICT_TARGETS.get(self._table, '(id)')

        results = []
        for record in data:
            processed = {k: self._wrap_value(k, v) for k, v in record.items()}

            columns = ', '.join(processed.keys())
            placeholders = ', '.join(['%s'] * len(processed))
            updates = ', '.join([f"{k} = EXCLUDED.{k}" for k in processed.keys()])
            values = list(processed.values())

            sql = (
                f"INSERT INTO {self._table} ({columns}) VALUES ({placeholders}) "
                f"ON CONFLICT {conflict_cols} DO UPDATE SET {updates} "
                f"RETURNING *"
            )
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, values)
                row = cur.fetchone()
                if row:
                    results.append(dict(row))
                else:
                    results.append(record)

        return DBResult(results[0] if len(results) == 1 else results)

    def _exec_delete(self, conn):
        where, params = self._build_where()
        sql = f"DELETE FROM {self._table}{where}"
        with conn.cursor() as cur:
            cur.execute(sql, params)
        return DBResult([])


class LocalDatabase:
    """
    Drop-in replacement for the Supabase client.
    Usage:  db.table('profiles').select('*').eq('id', uid).execute()
    """

    def __init__(self):
        init_db()

    def table(self, table_name: str) -> QueryBuilder:
        return QueryBuilder(table_name)


# Singleton instance
db = LocalDatabase()
