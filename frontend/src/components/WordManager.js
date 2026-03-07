import React, { useState, useEffect, useCallback } from 'react';
import API_BASE_URL from '../config';

/**
 * WordManager — Therapist/Parent tab to manage a child's difficult words.
 *
 * Core flow:
 *  1. Therapist adds difficult words the child needs to practice
 *  2. System generates stories using these words
 *  3. Per-word progress is tracked (accuracy, mastery, trend)
 *  4. Therapist removes mastered words and adds new ones
 *
 * Research basis:
 *  - Nation (2001): Word known after >80% accuracy + 5+ encounters
 *  - Leitner (1972): Box system for mastery classification
 *  - Ebbinghaus (1885): Forgetting curve / retention strength
 */
export default function WordManager({ userId }) {
  const [words, setWords] = useState([]);
  const [wordProgress, setWordProgress] = useState({});
  const [summary, setSummary] = useState({});
  const [newWord, setNewWord] = useState('');
  const [loading, setLoading] = useState(true);
  const [adding, setAdding] = useState(false);
  const [error, setError] = useState('');
  const [viewMode, setViewMode] = useState('therapist');
  const [sortBy, setSortBy] = useState('recommendation'); // 'recommendation', 'accuracy', 'attempts'

  // ----------------- Fetch data -----------------
  const fetchData = useCallback(async () => {
    try {
      const [wordsRes, progressRes] = await Promise.all([
        fetch(`${API_BASE_URL}/words/${userId}`),
        fetch(`${API_BASE_URL}/words/progress/${userId}`)
      ]);
      const wordsData = await wordsRes.json();
      const progressData = await progressRes.json();
      
      setWords(wordsData.words || []);
      setWordProgress(progressData.word_progress || {});
      setSummary(progressData.summary || {});
    } catch (err) {
      console.error('Error fetching word data:', err);
    } finally {
      setLoading(false);
    }
  }, [userId]);

  useEffect(() => { fetchData(); }, [fetchData]);

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
        fetchData();
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
      if (data.success) fetchData();
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

  const masteryConfig = {
    mastered:   { label: 'Mastered',    emoji: '🏆', color: '#22c55e', bg: 'rgba(34,197,94,0.2)' },
    familiar:   { label: 'Familiar',    emoji: '📗', color: '#3b82f6', bg: 'rgba(59,130,246,0.2)' },
    learning:   { label: 'Learning',    emoji: '📙', color: '#eab308', bg: 'rgba(234,179,8,0.2)' },
    struggling: { label: 'Struggling',  emoji: '📕', color: '#ef4444', bg: 'rgba(239,68,68,0.2)' },
    new:        { label: 'New',         emoji: '🆕', color: '#9ca3af', bg: 'rgba(156,163,175,0.2)' },
  };

  const trendConfig = {
    improving: { label: '↗ Improving', color: '#22c55e' },
    stable:    { label: '→ Stable',    color: '#94a3b8' },
    declining: { label: '↘ Declining', color: '#ef4444' },
  };

  const recConfig = {
    ready_to_remove: { label: '✅ Ready to Remove', color: '#22c55e', bg: 'rgba(34,197,94,0.15)' },
    keep_practicing:  { label: '🔄 Keep Practicing', color: '#eab308', bg: 'rgba(234,179,8,0.15)' },
    needs_attention:  { label: '⚠️ Needs Attention', color: '#ef4444', bg: 'rgba(239,68,68,0.15)' },
  };

  // Sort words
  const sortWords = (wordList) => {
    return [...wordList].sort((a, b) => {
      const pa = wordProgress[a] || {};
      const pb = wordProgress[b] || {};
      if (sortBy === 'recommendation') {
        const order = { needs_attention: 0, keep_practicing: 1, ready_to_remove: 2 };
        return (order[pa.recommendation] ?? 1) - (order[pb.recommendation] ?? 1);
      }
      if (sortBy === 'accuracy') return (pa.accuracy ?? 0) - (pb.accuracy ?? 0);
      if (sortBy === 'attempts') return (pb.attempts ?? 0) - (pa.attempts ?? 0);
      return 0;
    });
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

  // Compute display lists
  const allWords = Object.keys(wordProgress);
  const practicedOnly = allWords.filter(w => !words.includes(w) && (wordProgress[w]?.attempts || 0) > 0);
  const displayWords = viewMode === 'therapist' ? words
    : viewMode === 'practiced' ? practicedOnly
    : allWords;
  const sortedWords = sortWords(displayWords);

  return (
    <div style={styles.container}>
      {/* --------- Header --------- */}
      <div style={styles.header}>
        <span style={{ fontSize: 32 }}>📝</span>
        <div>
          <h2 style={styles.title}>Word Progress Manager</h2>
          <p style={styles.subtitle}>වචන ප්‍රගති කළමනාකරු · Track mastery & manage therapy words</p>
        </div>
      </div>

      {/* --------- Summary Cards --------- */}
      {summary.total_words > 0 && (
        <div style={styles.summaryRow}>
          <div style={{ ...styles.summaryCard, borderColor: '#3b82f6' }}>
            <span style={styles.summaryNum}>{summary.total_words}</span>
            <span style={styles.summaryLabel}>Total</span>
          </div>
          <div style={{ ...styles.summaryCard, borderColor: '#22c55e' }}>
            <span style={{ ...styles.summaryNum, color: '#22c55e' }}>{summary.mastered || 0}</span>
            <span style={styles.summaryLabel}>Mastered</span>
          </div>
          <div style={{ ...styles.summaryCard, borderColor: '#ef4444' }}>
            <span style={{ ...styles.summaryNum, color: '#ef4444' }}>{summary.struggling || 0}</span>
            <span style={styles.summaryLabel}>Struggling</span>
          </div>
          <div style={{ ...styles.summaryCard, borderColor: '#eab308' }}>
            <span style={{ ...styles.summaryNum, color: '#eab308' }}>{summary.overall_accuracy || 0}%</span>
            <span style={styles.summaryLabel}>Avg Accuracy</span>
          </div>
        </div>
      )}

      {/* --------- Recommendation Alerts --------- */}
      {(summary.ready_to_remove || []).length > 0 && (
        <div style={{ ...styles.alertBox, background: 'rgba(34,197,94,0.1)', borderColor: '#22c55e' }}>
          <span style={{ fontSize: 18 }}>✅</span>
          <div>
            <strong style={{ color: '#22c55e' }}>Ready to Remove:</strong>
            <span style={{ color: '#cbd5e1', marginLeft: 8 }}>
              {summary.ready_to_remove.join(', ')}
            </span>
          </div>
        </div>
      )}
      {(summary.needs_attention || []).length > 0 && (
        <div style={{ ...styles.alertBox, background: 'rgba(239,68,68,0.1)', borderColor: '#ef4444' }}>
          <span style={{ fontSize: 18 }}>⚠️</span>
          <div>
            <strong style={{ color: '#ef4444' }}>Needs Attention:</strong>
            <span style={{ color: '#cbd5e1', marginLeft: 8 }}>
              {summary.needs_attention.join(', ')}
            </span>
          </div>
        </div>
      )}

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

      {/* --------- View mode + Sort tabs --------- */}
      <div style={styles.controlRow}>
        <div style={styles.viewTabs}>
          {[
            { key: 'therapist', label: '📋 Therapist', count: words.length },
            { key: 'all', label: '🔤 All', count: allWords.length },
            { key: 'practiced', label: '🎮 From Stories', count: practicedOnly.length },
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
        <div style={styles.viewTabs}>
          {[
            { key: 'recommendation', label: 'Priority' },
            { key: 'accuracy', label: 'Accuracy' },
            { key: 'attempts', label: 'Most Practiced' },
          ].map(s => (
            <button
              key={s.key}
              onClick={() => setSortBy(s.key)}
              style={{
                ...styles.sortTab,
                ...(sortBy === s.key ? styles.sortTabActive : {}),
              }}
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>

      {/* --------- Word list --------- */}
      {sortedWords.length === 0 ? (
        <div style={styles.empty}>
          <span style={{ fontSize: 48 }}>📚</span>
          <p style={{ color: '#94a3b8', marginTop: 8 }}>
            {viewMode === 'practiced'
              ? 'No words from stories yet. Play some stories!'
              : 'No words added. Add difficult words above so they appear in stories.'}
          </p>
        </div>
      ) : (
        <div style={styles.list}>
          {sortedWords.map((word) => {
            const p = wordProgress[word] || {};
            const acc = p.accuracy ?? 0;
            const recentAcc = p.recent_accuracy ?? 0;
            const attempts = p.attempts ?? 0;
            const correct = p.correct ?? 0;
            const ml = p.mastery_level || 'new';
            const trend = p.trend || 'stable';
            const rec = p.recommendation || 'keep_practicing';
            const mc = masteryConfig[ml] || masteryConfig.new;
            const tc = trendConfig[trend] || trendConfig.stable;
            const rc = recConfig[rec] || recConfig.keep_practicing;
            const inList = words.includes(word);

            return (
              <div key={word} style={{
                ...styles.card,
                borderLeft: `4px solid ${mc.color}`,
              }}>
                {/* Top row: word + badges */}
                <div style={styles.cardTop}>
                  <div style={styles.cardLeft}>
                    <span style={styles.word}>{word}</span>
                    <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                      <span style={{ ...styles.badge, background: mc.bg, color: mc.color }}>
                        {mc.emoji} {mc.label}
                      </span>
                      <span style={{ ...styles.badge, color: tc.color, background: 'rgba(255,255,255,0.08)' }}>
                        {tc.label}
                      </span>
                      {!inList && (
                        <span style={{ ...styles.badge, background: 'rgba(99,102,241,0.2)', color: '#818cf8' }}>🎮 Story</span>
                      )}
                    </div>
                  </div>

                  {/* Action buttons */}
                  <div style={{ display: 'flex', gap: 6 }}>
                    {inList ? (
                      <button style={styles.removeBtn} onClick={() => handleRemove(word)} title="Remove word">
                        ✕
                      </button>
                    ) : (
                      <button
                        style={styles.addSmallBtn}
                        onClick={async () => {
                          await fetch(`${API_BASE_URL}/words/add`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ user_id: userId, word }),
                          });
                          fetchData();
                        }}
                        title="Add to therapist list"
                      >
                        ➕
                      </button>
                    )}
                  </div>
                </div>

                {/* Stats row */}
                <div style={styles.statsRow}>
                  {/* Accuracy */}
                  <div style={styles.statBlock}>
                    <span style={styles.statLabel}>Overall</span>
                    <div style={styles.barBg}>
                      <div style={{ ...styles.barFill, width: `${acc}%`, background: accuracyColor(acc) }} />
                    </div>
                    <span style={{ ...styles.statValue, color: accuracyColor(acc), fontSize: 13 }}>{acc}%</span>
                  </div>

                  {/* Recent accuracy */}
                  <div style={styles.statBlock}>
                    <span style={styles.statLabel}>Recent</span>
                    <div style={styles.barBg}>
                      <div style={{ ...styles.barFill, width: `${recentAcc}%`, background: accuracyColor(recentAcc) }} />
                    </div>
                    <span style={{ ...styles.statValue, color: accuracyColor(recentAcc), fontSize: 13 }}>{recentAcc}%</span>
                  </div>

                  {/* Attempts */}
                  <div style={styles.statBlock}>
                    <span style={styles.statLabel}>Attempts</span>
                    <span style={{ ...styles.statValue, fontSize: 15 }}>{attempts}</span>
                  </div>

                  {/* Correct */}
                  <div style={styles.statBlock}>
                    <span style={styles.statLabel}>Correct</span>
                    <span style={{ ...styles.statValue, color: '#22c55e', fontSize: 15 }}>{correct}</span>
                  </div>

                  {/* Leitner Box visual */}
                  <div style={styles.statBlock}>
                    <span style={styles.statLabel}>Box</span>
                    <div style={{ display: 'flex', gap: 2 }}>
                      {[1, 2, 3, 4, 5].map(b => (
                        <div key={b} style={{
                          width: 8, height: 14, borderRadius: 2,
                          background: b <= (p.leitner_box || 1) ? mc.color : 'rgba(255,255,255,0.15)',
                        }} />
                      ))}
                    </div>
                  </div>
                </div>

                {/* Recommendation bar */}
                <div style={{ ...styles.recBar, background: rc.bg }}>
                  <span style={{ color: rc.color, fontSize: 12, fontWeight: 600 }}>{rc.label}</span>
                  {p.interval_days > 0 && (
                    <span style={{ color: '#94a3b8', fontSize: 11, marginLeft: 'auto' }}>
                      Next review: {Math.round(p.interval_days)}d
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ======================= Inline Styles =======================

const styles = {
  container: { maxWidth: 850, margin: '0 auto', padding: 16 },
  center: { display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '60vh' },
  spinner: { fontSize: 40, animation: 'spin 1s linear infinite' },
  header: { display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 },
  title: { margin: 0, fontSize: 22, fontWeight: 800, color: '#fff' },
  subtitle: { margin: 0, fontSize: 13, color: '#cbd5e1' },
  summaryRow: { display: 'flex', gap: 10, marginBottom: 14, flexWrap: 'wrap' },
  summaryCard: {
    flex: '1 1 0', minWidth: 80, padding: '10px 14px', borderRadius: 12,
    background: 'rgba(255,255,255,0.08)', border: '1px solid', textAlign: 'center',
  },
  summaryNum: { display: 'block', fontSize: 22, fontWeight: 800, color: '#fff' },
  summaryLabel: { fontSize: 11, color: '#94a3b8', fontWeight: 600 },
  alertBox: {
    display: 'flex', alignItems: 'center', gap: 10, padding: '10px 14px',
    borderRadius: 10, border: '1px solid', marginBottom: 10, fontSize: 13,
  },
  addRow: { display: 'flex', gap: 8, marginBottom: 8 },
  input: {
    flex: 1, padding: '10px 14px', borderRadius: 12,
    border: '2px solid rgba(255,255,255,0.3)', background: 'rgba(255,255,255,0.15)',
    color: '#fff', fontSize: 16, outline: 'none', backdropFilter: 'blur(6px)',
  },
  addBtn: {
    padding: '10px 20px', borderRadius: 12, border: 'none',
    background: 'linear-gradient(135deg, #22c55e, #16a34a)',
    color: '#fff', fontWeight: 700, fontSize: 15, cursor: 'pointer', whiteSpace: 'nowrap',
  },
  error: { color: '#f87171', fontSize: 13, margin: '4px 0 0 4px' },
  controlRow: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12, flexWrap: 'wrap', gap: 8 },
  viewTabs: { display: 'flex', gap: 6, flexWrap: 'wrap' },
  viewTab: {
    padding: '6px 14px', borderRadius: 20, border: '1px solid rgba(255,255,255,0.2)',
    background: 'rgba(255,255,255,0.08)', color: '#cbd5e1', fontSize: 13, fontWeight: 600, cursor: 'pointer',
  },
  viewTabActive: {
    background: 'linear-gradient(135deg, #6366f1, #8b5cf6)', color: '#fff', border: '1px solid transparent',
  },
  sortTab: {
    padding: '4px 10px', borderRadius: 12, border: '1px solid rgba(255,255,255,0.15)',
    background: 'transparent', color: '#94a3b8', fontSize: 11, fontWeight: 600, cursor: 'pointer',
  },
  sortTabActive: { background: 'rgba(255,255,255,0.15)', color: '#fff' },
  empty: {
    textAlign: 'center', padding: 40, borderRadius: 16,
    background: 'rgba(255,255,255,0.07)', border: '2px dashed rgba(255,255,255,0.2)',
  },
  list: { display: 'flex', flexDirection: 'column', gap: 10 },
  card: {
    display: 'flex', flexDirection: 'column', gap: 8, padding: '14px 16px',
    borderRadius: 14, background: 'rgba(255,255,255,0.1)', backdropFilter: 'blur(8px)',
    border: '1px solid rgba(255,255,255,0.15)',
  },
  cardTop: { display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' },
  cardLeft: { display: 'flex', flexDirection: 'column', gap: 4 },
  word: { fontSize: 20, fontWeight: 800, color: '#fff' },
  badge: {
    padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 700, display: 'inline-block',
  },
  statsRow: { display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap' },
  statBlock: { display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2, minWidth: 60 },
  statLabel: { fontSize: 10, color: '#94a3b8', fontWeight: 600, textTransform: 'uppercase' },
  statValue: { fontWeight: 700, color: '#fff' },
  barBg: { width: 60, height: 5, borderRadius: 3, background: 'rgba(255,255,255,0.15)', overflow: 'hidden' },
  barFill: { height: '100%', borderRadius: 3, transition: 'width 0.5s ease' },
  recBar: {
    display: 'flex', alignItems: 'center', padding: '4px 10px', borderRadius: 8, gap: 6,
  },
  removeBtn: {
    width: 30, height: 30, borderRadius: '50%', border: 'none',
    background: 'rgba(239,68,68,0.25)', color: '#f87171', fontSize: 15,
    fontWeight: 700, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
  },
  addSmallBtn: {
    width: 30, height: 30, borderRadius: '50%', border: 'none',
    background: 'rgba(34,197,94,0.25)', color: '#4ade80', fontSize: 15,
    fontWeight: 700, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
  },
};
