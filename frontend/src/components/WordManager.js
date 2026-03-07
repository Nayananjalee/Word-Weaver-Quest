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
    mastered:   { label: 'Mastered',    emoji: '🏆', color: '#15803d', bg: '#dcfce7' },
    familiar:   { label: 'Familiar',    emoji: '📗', color: '#1d4ed8', bg: '#dbeafe' },
    learning:   { label: 'Learning',    emoji: '📙', color: '#a16207', bg: '#fef9c3' },
    struggling: { label: 'Struggling',  emoji: '📕', color: '#b91c1c', bg: '#fee2e2' },
    new:        { label: 'New',         emoji: '🆕', color: '#4b5563', bg: '#f3f4f6' },
  };

  const trendConfig = {
    improving: { label: '↗ Improving', color: '#15803d' },
    stable:    { label: '→ Stable',    color: '#475569' },
    declining: { label: '↘ Declining', color: '#b91c1c' },
  };

  const recConfig = {
    ready_to_remove: { label: '✅ Ready to Remove', color: '#15803d', bg: '#dcfce7' },
    keep_practicing:  { label: '🔄 Keep Practicing', color: '#a16207', bg: '#fef9c3' },
    needs_attention:  { label: '⚠️ Needs Attention', color: '#b91c1c', bg: '#fee2e2' },
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
        <p style={{ color: '#1e293b', marginTop: 12, fontWeight: 600 }}>Loading words…</p>
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
        <div style={{ ...styles.alertBox, background: '#f0fdf4', borderColor: '#22c55e' }}>
          <span style={{ fontSize: 18 }}>✅</span>
          <div>
            <strong style={{ color: '#15803d' }}>Ready to Remove:</strong>
            <span style={{ color: '#1e293b', marginLeft: 8, fontWeight: 600 }}>
              {summary.ready_to_remove.join(', ')}
            </span>
          </div>
        </div>
      )}
      {(summary.needs_attention || []).length > 0 && (
        <div style={{ ...styles.alertBox, background: '#fef2f2', borderColor: '#ef4444' }}>
          <span style={{ fontSize: 18 }}>⚠️</span>
          <div>
            <strong style={{ color: '#b91c1c' }}>Needs Attention:</strong>
            <span style={{ color: '#1e293b', marginLeft: 8, fontWeight: 600 }}>
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
          <p style={{ color: '#475569', marginTop: 8, fontWeight: 600 }}>
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
                      <span style={{ ...styles.badge, color: tc.color, background: '#f1f5f9' }}>
                        {tc.label}
                      </span>
                      {!inList && (
                        <span style={{ ...styles.badge, background: '#eef2ff', color: '#4f46e5' }}>🎮 Story</span>
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
                          background: b <= (p.leitner_box || 1) ? mc.color : '#e2e8f0',
                        }} />
                      ))}
                    </div>
                  </div>
                </div>

                {/* Recommendation bar */}
                <div style={{ ...styles.recBar, background: rc.bg }}>
                  <span style={{ color: rc.color, fontSize: 12, fontWeight: 600 }}>{rc.label}</span>
                  {p.interval_days > 0 && (
                    <span style={{ color: '#64748b', fontSize: 11, fontWeight: 600, marginLeft: 'auto' }}>
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
  title: { margin: 0, fontSize: 24, fontWeight: 800, color: '#1e293b' },
  subtitle: { margin: 0, fontSize: 13, color: '#475569' },
  summaryRow: { display: 'flex', gap: 10, marginBottom: 14, flexWrap: 'wrap' },
  summaryCard: {
    flex: '1 1 0', minWidth: 90, padding: '12px 14px', borderRadius: 14,
    background: '#fff', border: '2px solid', textAlign: 'center',
    boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
  },
  summaryNum: { display: 'block', fontSize: 26, fontWeight: 900, color: '#1e293b' },
  summaryLabel: { fontSize: 12, color: '#64748b', fontWeight: 700 },
  alertBox: {
    display: 'flex', alignItems: 'center', gap: 10, padding: '12px 16px',
    borderRadius: 12, border: '2px solid', marginBottom: 10, fontSize: 14,
    background: '#fff', boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
  },
  addRow: { display: 'flex', gap: 8, marginBottom: 10 },
  input: {
    flex: 1, padding: '12px 16px', borderRadius: 14,
    border: '2px solid #cbd5e1', background: '#fff',
    color: '#1e293b', fontSize: 17, outline: 'none', fontWeight: 500,
  },
  addBtn: {
    padding: '12px 22px', borderRadius: 14, border: 'none',
    background: 'linear-gradient(135deg, #22c55e, #16a34a)',
    color: '#fff', fontWeight: 800, fontSize: 16, cursor: 'pointer', whiteSpace: 'nowrap',
    boxShadow: '0 2px 8px rgba(34,197,94,0.3)',
  },
  error: { color: '#dc2626', fontSize: 14, fontWeight: 600, margin: '4px 0 0 4px' },
  controlRow: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12, flexWrap: 'wrap', gap: 8 },
  viewTabs: { display: 'flex', gap: 6, flexWrap: 'wrap' },
  viewTab: {
    padding: '8px 16px', borderRadius: 22, border: '2px solid #cbd5e1',
    background: '#fff', color: '#475569', fontSize: 14, fontWeight: 700, cursor: 'pointer',
  },
  viewTabActive: {
    background: 'linear-gradient(135deg, #6366f1, #8b5cf6)', color: '#fff',
    border: '2px solid transparent', boxShadow: '0 2px 8px rgba(99,102,241,0.3)',
  },
  sortTab: {
    padding: '5px 12px', borderRadius: 14, border: '2px solid #e2e8f0',
    background: '#f8fafc', color: '#64748b', fontSize: 12, fontWeight: 700, cursor: 'pointer',
  },
  sortTabActive: { background: '#334155', color: '#fff', border: '2px solid #334155' },
  empty: {
    textAlign: 'center', padding: 40, borderRadius: 16,
    background: '#fff', border: '2px dashed #cbd5e1',
    boxShadow: '0 2px 8px rgba(0,0,0,0.05)',
  },
  list: { display: 'flex', flexDirection: 'column', gap: 12 },
  card: {
    display: 'flex', flexDirection: 'column', gap: 10, padding: '16px 18px',
    borderRadius: 16, background: '#fff',
    border: '1px solid #e2e8f0', boxShadow: '0 2px 12px rgba(0,0,0,0.07)',
  },
  cardTop: { display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' },
  cardLeft: { display: 'flex', flexDirection: 'column', gap: 6 },
  word: { fontSize: 24, fontWeight: 900, color: '#1e293b', letterSpacing: 0.5 },
  badge: {
    padding: '3px 10px', borderRadius: 8, fontSize: 12, fontWeight: 800, display: 'inline-block',
  },
  statsRow: { display: 'flex', gap: 20, alignItems: 'center', flexWrap: 'wrap', padding: '6px 0' },
  statBlock: { display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3, minWidth: 65 },
  statLabel: { fontSize: 11, color: '#64748b', fontWeight: 700, textTransform: 'uppercase', letterSpacing: 0.5 },
  statValue: { fontWeight: 800, color: '#1e293b' },
  barBg: { width: 70, height: 8, borderRadius: 4, background: '#e2e8f0', overflow: 'hidden' },
  barFill: { height: '100%', borderRadius: 4, transition: 'width 0.5s ease' },
  recBar: {
    display: 'flex', alignItems: 'center', padding: '6px 12px', borderRadius: 10, gap: 6,
  },
  removeBtn: {
    width: 34, height: 34, borderRadius: '50%', border: '2px solid #fca5a5',
    background: '#fef2f2', color: '#dc2626', fontSize: 16,
    fontWeight: 800, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
  },
  addSmallBtn: {
    width: 34, height: 34, borderRadius: '50%', border: '2px solid #86efac',
    background: '#f0fdf4', color: '#16a34a', fontSize: 16,
    fontWeight: 800, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
  },
};
