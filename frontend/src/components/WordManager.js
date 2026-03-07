import React, { useState, useEffect, useCallback } from 'react';
import API_BASE_URL from '../config';

/**
 * WordManager — Therapist tab to manage a child's difficult words.
 *
 * Features:
 *  • View current word list (therapist-added)
 *  • View practiced words (from gameplay)
 *  • Add / remove words
 *  • Per-word performance stats (accuracy, attempts, SRS box)
 */
export default function WordManager({ userId }) {
  const [words, setWords] = useState([]);
  const [practicedWords, setPracticedWords] = useState([]);
  const [wordStats, setWordStats] = useState({});
  const [newWord, setNewWord] = useState('');
  const [loading, setLoading] = useState(true);
  const [adding, setAdding] = useState(false);
  const [error, setError] = useState('');
  const [viewMode, setViewMode] = useState('all'); // 'all', 'therapist', 'practiced'

  // ----------------- Fetch words -----------------
  const fetchWords = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/words/${userId}`);
      const data = await res.json();
      setWords(data.words || []);
      setPracticedWords(data.practiced_words || []);
      setWordStats(data.word_stats || {});
    } catch (err) {
      console.error('Error fetching words:', err);
    } finally {
      setLoading(false);
    }
  }, [userId]);

  useEffect(() => { fetchWords(); }, [fetchWords]);

  // ----------------- Add word -----------------
  const handleAdd = async () => {
    const trimmed = newWord.trim();
    if (!trimmed) return;
    setAdding(true);
    setError('');
    try {
      const res = await fetch(`${API_BASE_URL}/words/add`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, word: trimmed }),
      });
      const data = await res.json();
      if (data.success) {
        setNewWord('');
        fetchWords();
      } else {
        setError(data.detail || 'Failed to add word');
      }
    } catch (err) {
      setError('Network error');
    } finally {
      setAdding(false);
    }
  };

  // ----------------- Remove word -----------------
  const handleRemove = async (word) => {
    try {
      const res = await fetch(`${API_BASE_URL}/words/remove`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, word }),
      });
      const data = await res.json();
      if (data.success) fetchWords();
    } catch (err) {
      console.error('Remove word error:', err);
    }
  };

  // ---- helpers ----
  const accuracyColor = (acc) => {
    if (acc >= 80) return '#22c55e';
    if (acc >= 50) return '#eab308';
    return '#ef4444';
  };

  const boxLabel = (box) => {
    const labels = { 1: 'New', 2: 'Learning', 3: 'Reviewing', 4: 'Familiar', 5: 'Mastered' };
    return labels[box] || `Box ${box}`;
  };

  const boxColor = (box) => {
    const colors = { 1: '#ef4444', 2: '#f97316', 3: '#eab308', 4: '#3b82f6', 5: '#22c55e' };
    return colors[box] || '#9ca3af';
  };

  // -------------------- UI --------------------

  if (loading) {
    return (
      <div style={styles.center}>
        <div style={styles.spinner}>⏳</div>
        <p style={{ color: '#fff', marginTop: 12 }}>Loading words…</p>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      {/* --------- Header --------- */}
      <div style={styles.header}>
        <span style={{ fontSize: 32 }}>📝</span>
        <div>
          <h2 style={styles.title}>Difficult Words Manager</h2>
          <p style={styles.subtitle}>දුෂ්කර වචන කළමනාකරු · Add or remove therapy target words</p>
        </div>
      </div>

      {/* --------- Add word input --------- */}
      <div style={styles.addRow}>
        <input
          style={styles.input}
          type="text"
          value={newWord}
          onChange={(e) => setNewWord(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleAdd()}
          placeholder="Type a Sinhala word… (e.g. පුස්තකාලය)"
        />
        <button style={styles.addBtn} onClick={handleAdd} disabled={adding || !newWord.trim()}>
          {adding ? '…' : '➕ Add'}
        </button>
      </div>
      {error && <p style={styles.error}>{error}</p>}

      {/* --------- View mode tabs --------- */}
      <div style={styles.viewTabs}>
        {[
          { key: 'all', label: '🔤 All Words', count: Object.keys(wordStats).length },
          { key: 'therapist', label: '📋 Therapist List', count: words.length },
          { key: 'practiced', label: '🎮 Practiced', count: practicedWords.length },
        ].map(tab => (
          <button
            key={tab.key}
            onClick={() => setViewMode(tab.key)}
            style={{
              ...styles.viewTab,
              ...(viewMode === tab.key ? styles.viewTabActive : {}),
            }}
          >
            {tab.label} ({tab.count})
          </button>
        ))}
      </div>

      {/* --------- Word count --------- */}
      {(() => {
        const displayWords = viewMode === 'therapist' ? words
          : viewMode === 'practiced' ? practicedWords
          : [...new Set([...words, ...practicedWords])];
        const count = displayWords.length;
        return <p style={styles.count}>{count} word{count !== 1 ? 's' : ''} {viewMode === 'all' ? 'total' : viewMode === 'therapist' ? 'in therapist list' : 'practiced in stories'}</p>;
      })()}

      {/* --------- Word list --------- */}
      {(() => {
        const displayWords = viewMode === 'therapist' ? words
          : viewMode === 'practiced' ? practicedWords
          : [...new Set([...words, ...practicedWords])];
        
        if (displayWords.length === 0) {
          return (
            <div style={styles.empty}>
              <span style={{ fontSize: 48 }}>📚</span>
              <p style={{ color: '#94a3b8', marginTop: 8 }}>
                {viewMode === 'practiced'
                  ? 'No words practiced yet. Play some stories!'
                  : 'No words added yet. Add difficult words above so they appear in stories.'}
              </p>
            </div>
          );
        }
        
        return (
          <div style={styles.list}>
            {displayWords.map((word) => {
              const s = wordStats[word] || {};
              const acc = s.accuracy ?? 0;
              const attempts = s.attempts ?? 0;
              const correct = s.correct ?? 0;
              const box = s.leitner_box ?? 1;
              const inTherapistList = words.includes(word);

              return (
                <div key={word} style={styles.card}>
                  {/* Left: word + box badge */}
                  <div style={styles.cardLeft}>
                    <span style={styles.word}>{word}</span>
                    <div style={{ display: 'flex', gap: 4 }}>
                      <span style={{ ...styles.boxBadge, background: boxColor(box) }}>{boxLabel(box)}</span>
                      {!inTherapistList && (
                        <span style={{ ...styles.boxBadge, background: '#6366f1' }}>Story</span>
                      )}
                    </div>
                  </div>

                  {/* Middle: stats */}
                  <div style={styles.statsRow}>
                    {/* Accuracy bar */}
                    <div style={styles.statBlock}>
                      <span style={styles.statLabel}>Accuracy</span>
                      <div style={styles.barBg}>
                        <div style={{ ...styles.barFill, width: `${acc}%`, background: accuracyColor(acc) }} />
                      </div>
                      <span style={{ ...styles.statValue, color: accuracyColor(acc) }}>{acc}%</span>
                    </div>

                    {/* Attempts */}
                    <div style={styles.statBlock}>
                      <span style={styles.statLabel}>Attempts</span>
                      <span style={styles.statValue}>{attempts}</span>
                    </div>

                    {/* Correct */}
                    <div style={styles.statBlock}>
                      <span style={styles.statLabel}>Correct</span>
                      <span style={{ ...styles.statValue, color: '#22c55e' }}>{correct}</span>
                    </div>
                  </div>

                  {/* Right: remove button (only for therapist words) */}
                  {inTherapistList ? (
                    <button style={styles.removeBtn} onClick={() => handleRemove(word)} title="Remove word">
                      ✕
                    </button>
                  ) : (
                    <button
                      style={{ ...styles.addSmallBtn }}
                      onClick={async () => {
                        await fetch(`${API_BASE_URL}/words/add`, {
                          method: 'POST',
                          headers: { 'Content-Type': 'application/json' },
                          body: JSON.stringify({ user_id: userId, word }),
                        });
                        fetchWords();
                      }}
                      title="Add to therapist list"
                    >
                      ➕
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        );
      })()}
    </div>
  );
}

// ======================= Inline Styles =======================

const styles = {
  container: {
    maxWidth: 800,
    margin: '0 auto',
    padding: 16,
  },
  center: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    height: '60vh',
  },
  spinner: { fontSize: 40, animation: 'spin 1s linear infinite' },
  header: {
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    marginBottom: 20,
  },
  title: {
    margin: 0,
    fontSize: 22,
    fontWeight: 800,
    color: '#fff',
  },
  subtitle: {
    margin: 0,
    fontSize: 13,
    color: '#cbd5e1',
  },
  addRow: {
    display: 'flex',
    gap: 8,
    marginBottom: 8,
  },
  input: {
    flex: 1,
    padding: '10px 14px',
    borderRadius: 12,
    border: '2px solid rgba(255,255,255,0.3)',
    background: 'rgba(255,255,255,0.15)',
    color: '#fff',
    fontSize: 16,
    outline: 'none',
    backdropFilter: 'blur(6px)',
  },
  addBtn: {
    padding: '10px 20px',
    borderRadius: 12,
    border: 'none',
    background: 'linear-gradient(135deg, #22c55e, #16a34a)',
    color: '#fff',
    fontWeight: 700,
    fontSize: 15,
    cursor: 'pointer',
    whiteSpace: 'nowrap',
  },
  error: { color: '#f87171', fontSize: 13, margin: '4px 0 0 4px' },
  count: { color: '#94a3b8', fontSize: 13, marginBottom: 12 },
  empty: {
    textAlign: 'center',
    padding: 40,
    borderRadius: 16,
    background: 'rgba(255,255,255,0.07)',
    border: '2px dashed rgba(255,255,255,0.2)',
  },
  list: {
    display: 'flex',
    flexDirection: 'column',
    gap: 10,
  },
  card: {
    display: 'flex',
    alignItems: 'center',
    gap: 16,
    padding: '14px 16px',
    borderRadius: 14,
    background: 'rgba(255,255,255,0.12)',
    backdropFilter: 'blur(8px)',
    border: '1px solid rgba(255,255,255,0.18)',
  },
  cardLeft: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'flex-start',
    gap: 4,
    minWidth: 110,
  },
  word: {
    fontSize: 18,
    fontWeight: 700,
    color: '#fff',
  },
  boxBadge: {
    padding: '2px 8px',
    borderRadius: 6,
    fontSize: 11,
    fontWeight: 700,
    color: '#fff',
  },
  statsRow: {
    flex: 1,
    display: 'flex',
    gap: 20,
    alignItems: 'center',
  },
  statBlock: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: 2,
    minWidth: 70,
  },
  statLabel: { fontSize: 11, color: '#94a3b8', fontWeight: 600 },
  statValue: { fontSize: 15, fontWeight: 700, color: '#fff' },
  barBg: {
    width: 70,
    height: 6,
    borderRadius: 3,
    background: 'rgba(255,255,255,0.15)',
    overflow: 'hidden',
  },
  barFill: {
    height: '100%',
    borderRadius: 3,
    transition: 'width 0.5s ease',
  },
  removeBtn: {
    width: 32,
    height: 32,
    borderRadius: '50%',
    border: 'none',
    background: 'rgba(239,68,68,0.25)',
    color: '#f87171',
    fontSize: 16,
    fontWeight: 700,
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    flexShrink: 0,
  },
  addSmallBtn: {
    width: 32,
    height: 32,
    borderRadius: '50%',
    border: 'none',
    background: 'rgba(34,197,94,0.25)',
    color: '#4ade80',
    fontSize: 16,
    fontWeight: 700,
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    flexShrink: 0,
  },
  viewTabs: {
    display: 'flex',
    gap: 6,
    marginBottom: 12,
    flexWrap: 'wrap',
  },
  viewTab: {
    padding: '6px 14px',
    borderRadius: 20,
    border: '1px solid rgba(255,255,255,0.2)',
    background: 'rgba(255,255,255,0.08)',
    color: '#cbd5e1',
    fontSize: 13,
    fontWeight: 600,
    cursor: 'pointer',
  },
  viewTabActive: {
    background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
    color: '#fff',
    border: '1px solid transparent',
  },
};
