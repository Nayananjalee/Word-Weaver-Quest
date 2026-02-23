"""
SQLite Database Layer for Word-Weaver-Quest
=============================================
Replaces Supabase with a local SQLite database.
Provides a simple interface compatible with the existing codebase patterns.
"""

import sqlite3
import json
import uuid
import os
from datetime import datetime
from contextlib import contextmanager

# Database file lives next to main.py
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "word_weaver.db")


@contextmanager
def get_connection():
    """Thread-safe connection context manager."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Return dict-like rows
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create all tables if they don't exist. Called once at startup."""
    with get_connection() as conn:
        conn.executescript("""
        -- Profiles table (replaces Supabase auth.users + profiles)
        CREATE TABLE IF NOT EXISTS profiles (
            id TEXT PRIMARY KEY,
            username TEXT DEFAULT '',
            score INTEGER DEFAULT 0,
            learning_level INTEGER DEFAULT 1,
            difficult_words TEXT DEFAULT '[]',
            adaptive_state TEXT,
            engagement_state TEXT,
            phoneme_state TEXT,
            attention_state TEXT,
            trajectory_state TEXT,
            srs_state TEXT,
            cognitive_load_state TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );

        -- Stories table
        CREATE TABLE IF NOT EXISTS stories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT REFERENCES profiles(id),
            story_text TEXT,
            question TEXT DEFAULT '',
            correct_answer TEXT,
            options TEXT,
            difficulty_level INTEGER DEFAULT 1,
            topic TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now'))
        );

        -- Performance logs
        CREATE TABLE IF NOT EXISTS performance_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT REFERENCES profiles(id),
            story_id INTEGER,
            is_correct INTEGER NOT NULL DEFAULT 0,
            response_time REAL NOT NULL DEFAULT 0,
            engagement_score REAL,
            difficulty_level INTEGER,
            created_at TEXT DEFAULT (datetime('now'))
        );

        -- Phoneme errors
        CREATE TABLE IF NOT EXISTS phoneme_errors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT REFERENCES profiles(id),
            phoneme_pair TEXT NOT NULL,
            error_count INTEGER DEFAULT 1,
            last_error_at TEXT DEFAULT (datetime('now')),
            created_at TEXT DEFAULT (datetime('now')),
            UNIQUE(user_id, phoneme_pair)
        );

        -- Engagement logs
        CREATE TABLE IF NOT EXISTS engagement_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT REFERENCES profiles(id),
            session_id TEXT NOT NULL,
            timestamp TEXT DEFAULT (datetime('now')),
            engagement_score REAL,
            response_time REAL,
            gesture_quality REAL
        );

        -- Dropout predictions
        CREATE TABLE IF NOT EXISTS dropout_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_id TEXT REFERENCES profiles(id),
            predicted_at TEXT DEFAULT (datetime('now')),
            dropout_probability REAL,
            intervention_triggered INTEGER DEFAULT 0,
            actual_dropout INTEGER
        );

        -- Therapy sessions
        CREATE TABLE IF NOT EXISTS therapy_sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT REFERENCES profiles(id),
            started_at TEXT DEFAULT (datetime('now')),
            ended_at TEXT,
            total_questions INTEGER DEFAULT 0,
            correct_answers INTEGER DEFAULT 0,
            average_response_time REAL,
            average_engagement_score REAL,
            difficulty_level_start INTEGER,
            difficulty_level_end INTEGER,
            dropout INTEGER DEFAULT 0
        );

        -- Severity estimates
        CREATE TABLE IF NOT EXISTS severity_estimates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT REFERENCES profiles(id),
            estimated_severity TEXT,
            confidence REAL,
            details TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );

        -- Session analytics (Feature 7)
        CREATE TABLE IF NOT EXISTS session_analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT REFERENCES profiles(id),
            session_id TEXT,
            total_questions INTEGER DEFAULT 0,
            correct_answers INTEGER DEFAULT 0,
            accuracy REAL,
            avg_response_time REAL,
            learning_efficiency_index REAL,
            flow_ratio REAL,
            zpd_alignment REAL,
            resilience_score REAL,
            engagement_consistency REAL,
            attention_quality_index REAL,
            streak_max INTEGER DEFAULT 0,
            frustration_events INTEGER DEFAULT 0,
            boredom_events INTEGER DEFAULT 0,
            topic TEXT,
            difficulty_start INTEGER,
            difficulty_end INTEGER,
            summary TEXT,
            research_metrics TEXT,
            timestamp REAL,
            created_at TEXT DEFAULT (datetime('now'))
        );

        -- Word mastery
        CREATE TABLE IF NOT EXISTS word_mastery (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT REFERENCES profiles(id),
            word TEXT NOT NULL,
            mastery_score REAL DEFAULT 500.0,
            attempts INTEGER DEFAULT 0,
            correct INTEGER DEFAULT 0,
            last_seen_at TEXT DEFAULT (datetime('now')),
            UNIQUE(user_id, word)
        );

        -- Weekly summaries
        CREATE TABLE IF NOT EXISTS weekly_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT REFERENCES profiles(id),
            week_start_date TEXT,
            week_end_date TEXT,
            total_sessions INTEGER,
            total_questions INTEGER,
            accuracy_rate REAL,
            average_engagement REAL,
            most_confused_phonemes TEXT,
            improvement_areas TEXT,
            report_generated_at TEXT DEFAULT (datetime('now')),
            UNIQUE(user_id, week_start_date)
        );

        -- Insert default user profile if it doesn't exist
        INSERT OR IGNORE INTO profiles (id, username, score, learning_level)
        VALUES ('123e4567-e89b-12d3-a456-426614174000', 'default_child', 0, 1);
        """)
    print(f"[DB] SQLite database initialized at {DB_PATH}")

    # Migrate existing tables — add any columns that may have been added after initial creation
    _migrate_columns()


def _migrate_columns():
    """Add missing columns to existing tables without data loss."""
    migrations = [
        ("session_analytics", "summary", "TEXT"),
        ("session_analytics", "research_metrics", "TEXT"),
        ("session_analytics", "timestamp", "REAL"),
    ]
    with get_connection() as conn:
        for table, column, col_type in migrations:
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
            except Exception:
                pass  # Column already exists — that's fine


# =====================================================
# HELPER CLASS: Drop-in replacement for Supabase client
# =====================================================

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

    def upsert(self, data):
        self._operation = 'upsert'
        self._data = data
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

    def _build_where(self):
        if not self._filters:
            return "", []
        clauses = []
        params = []
        for col, op, val in self._filters:
            clauses.append(f"{col} {op} ?")
            params.append(val)
        return " WHERE " + " AND ".join(clauses), params

    def _exec_select(self, conn):
        cols = self._columns.replace(' ', '')  # "id, score" -> "id,score"
        sql = f"SELECT {cols} FROM {self._table}"
        where, params = self._build_where()
        sql += where
        if self._order_col:
            direction = "DESC" if self._order_desc else "ASC"
            sql += f" ORDER BY {self._order_col} {direction}"
        if self._limit_val:
            sql += f" LIMIT {self._limit_val}"
        
        cursor = conn.execute(sql, params)
        rows = [dict(r) for r in cursor.fetchall()]

        # Parse JSON columns — all state blobs are parsed to Python objects so
        # ML engines that expect dicts receive dicts. Engines that call json.loads()
        # on them are fixed to accept both str and dict.
        json_columns = ['difficult_words', 'adaptive_state', 'engagement_state',
                        'phoneme_state', 'attention_state', 'trajectory_state',
                        'srs_state', 'cognitive_load_state',
                        'most_confused_phonemes', 'improvement_areas', 'options', 'details']
        for row in rows:
            for jcol in json_columns:
                if jcol in row and isinstance(row[jcol], str):
                    try:
                        row[jcol] = json.loads(row[jcol])
                    except (json.JSONDecodeError, TypeError):
                        pass

        if self._single:
            return DBResult(rows[0] if rows else None)
        return DBResult(rows)

    def _exec_insert(self, conn):
        data = self._data
        if isinstance(data, dict):
            data = [data]
        
        results = []
        for record in data:
            # Serialize any dict/list values to JSON
            processed = {}
            for k, v in record.items():
                if isinstance(v, (dict, list)):
                    processed[k] = json.dumps(v)
                else:
                    processed[k] = v

            columns = ', '.join(processed.keys())
            placeholders = ', '.join(['?'] * len(processed))
            values = list(processed.values())
            
            cursor = conn.execute(
                f"INSERT INTO {self._table} ({columns}) VALUES ({placeholders})",
                values
            )
            # Return inserted row
            inserted_id = cursor.lastrowid
            row_cursor = conn.execute(f"SELECT * FROM {self._table} WHERE rowid = ?", [inserted_id])
            row = row_cursor.fetchone()
            if row:
                results.append(dict(row))
        
        return DBResult(results[0] if len(results) == 1 else results)

    def _exec_update(self, conn):
        processed = {}
        for k, v in self._data.items():
            if isinstance(v, (dict, list)):
                processed[k] = json.dumps(v)
            else:
                processed[k] = v

        set_clause = ', '.join([f"{k} = ?" for k in processed.keys()])
        set_values = list(processed.values())
        
        where, where_params = self._build_where()
        conn.execute(
            f"UPDATE {self._table} SET {set_clause}{where}",
            set_values + where_params
        )
        
        # Return updated rows
        select_cursor = conn.execute(f"SELECT * FROM {self._table}{where}", where_params)
        rows = [dict(r) for r in select_cursor.fetchall()]
        return DBResult(rows)

    def _exec_upsert(self, conn):
        data = self._data
        if isinstance(data, dict):
            data = [data]
        
        results = []
        for record in data:
            processed = {}
            for k, v in record.items():
                if isinstance(v, (dict, list)):
                    processed[k] = json.dumps(v)
                else:
                    processed[k] = v

            columns = ', '.join(processed.keys())
            placeholders = ', '.join(['?'] * len(processed))
            updates = ', '.join([f"{k} = excluded.{k}" for k in processed.keys()])
            values = list(processed.values())
            
            conn.execute(
                f"INSERT INTO {self._table} ({columns}) VALUES ({placeholders}) "
                f"ON CONFLICT DO UPDATE SET {updates}",
                values
            )
            results.append(record)
        
        return DBResult(results[0] if len(results) == 1 else results)

    def _exec_delete(self, conn):
        where, params = self._build_where()
        conn.execute(f"DELETE FROM {self._table}{where}", params)
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
