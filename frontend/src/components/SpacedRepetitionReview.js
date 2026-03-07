/**
 * SpacedRepetitionReview - Phoneme-Aware Spaced Repetition Review Component
 * 
 * Implements an interactive review session using SM-2 + Leitner hybrid algorithm
 * with phoneme confusion weighting for speech therapy optimization.
 * 
 * Features:
 * - Due word queue with priority ordering
 * - Visual Leitner box indicator
 * - Forgetting curve visualization
 * - Phoneme mastery heatmap
 * - Session statistics
 * 
 * Research basis:
 * - Settles & Meeder (2016): Trainable Spaced Repetition Model
 * - Nakata & Elgort (2021): SRS for Vocabulary Acquisition
 * - Kang (2020): Spaced Repetition for Disabilities
 * - Tabibian et al. (2019): Enhancing Human Learning via Spaced Repetition
 */

import React, { useState, useEffect, useCallback } from 'react';
import API_BASE_URL from '../config';

const SpacedRepetitionReview = ({ userId }) => {
  const [dueWords, setDueWords] = useState([]);
  const [currentWord, setCurrentWord] = useState(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [statistics, setStatistics] = useState(null);
  const [sessionActive, setSessionActive] = useState(false);
  const [sessionResults, setSessionResults] = useState([]);
  const [loading, setLoading] = useState(true);
  const [reviewStartTime, setReviewStartTime] = useState(null);
  const [selectedQuality, setSelectedQuality] = useState(null);
  const [showResult, setShowResult] = useState(false);
  const [activeView, setActiveView] = useState('overview'); // overview, review, stats

  // Fetch SRS statistics and due words
  const fetchData = useCallback(async () => {
    if (!userId) return;
    setLoading(true);
    try {
      const [statsRes, dueRes] = await Promise.all([
        fetch(`${API_BASE_URL}/srs/statistics/${userId}`),
        fetch(`${API_BASE_URL}/srs/due-words/${userId}?max_count=20`)
      ]);
      
      const statsData = await statsRes.json();
      const dueData = await dueRes.json();
      
      if (statsData) setStatistics(statsData);
      if (dueData?.words) setDueWords(dueData.words);
    } catch (err) {
      console.error('SRS data fetch error:', err);
    } finally {
      setLoading(false);
    }
  }, [userId]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Start a review session
  const startSession = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/srs/generate-session`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, max_words: 8, include_new: 2 })
      });
      const data = await res.json();
      
      if (data?.session?.words_to_review?.length > 0) {
        const words = data.session.words_to_review.map(w => ({
          word: w,
          ...(data.word_details?.[w] || {})
        }));
        setDueWords(words);
        setCurrentIndex(0);
        setCurrentWord(words[0]);
        setSessionActive(true);
        setSessionResults([]);
        setReviewStartTime(Date.now());
        setActiveView('review');
      }
    } catch (err) {
      console.error('Session start error:', err);
    }
  };

  // Submit review quality
  const submitReview = async (quality) => {
    if (!currentWord) return;
    setSelectedQuality(quality);
    setShowResult(true);

    const responseTime = (Date.now() - reviewStartTime) / 1000;

    try {
      const res = await fetch(`${API_BASE_URL}/srs/review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          word: currentWord.word,
          quality: quality,
          response_time: responseTime
        })
      });
      const result = await res.json();
      
      setSessionResults(prev => [...prev, {
        word: currentWord.word,
        quality,
        responseTime,
        result: result?.review_result
      }]);
    } catch (err) {
      console.error('Review submit error:', err);
    }

    // Move to next word after delay
    setTimeout(() => {
      setShowResult(false);
      setSelectedQuality(null);
      
      if (currentIndex + 1 < dueWords.length) {
        setCurrentIndex(prev => prev + 1);
        setCurrentWord(dueWords[currentIndex + 1]);
        setReviewStartTime(Date.now());
      } else {
        // Session complete
        setSessionActive(false);
        setActiveView('stats');
        fetchData(); // Refresh stats
      }
    }, 1500);
  };

  // Quality rating buttons with descriptions
  const qualityOptions = [
    { value: 0, label: '😰', desc: 'මතක නැහැ', descEn: 'Blackout', color: 'red' },
    { value: 1, label: '😟', desc: 'වැරදියි', descEn: 'Wrong', color: 'orange' },
    { value: 2, label: '🤔', desc: 'අමාරුයි', descEn: 'Hard', color: 'yellow' },
    { value: 3, label: '😊', desc: 'හරි - අමාරුයි', descEn: 'Correct-Hard', color: 'lime' },
    { value: 4, label: '😄', desc: 'හරි', descEn: 'Correct', color: 'green' },
    { value: 5, label: '🌟', desc: 'පරිපූර්ණයි!', descEn: 'Perfect!', color: 'emerald' },
  ];

  // Leitner box visual indicator
  const LeitnerBoxIndicator = ({ box }) => {
    const boxes = [1, 2, 3, 4, 5];
    const colors = ['bg-red-400', 'bg-orange-400', 'bg-yellow-400', 'bg-green-400', 'bg-emerald-400'];
    return (
      <div className="flex gap-1 items-center">
        <span className="text-xs text-gray-500 mr-1">Box:</span>
        {boxes.map(b => (
          <div
            key={b}
            className={`w-4 h-4 rounded ${b <= box ? colors[b - 1] : 'bg-gray-200'} transition-all ${b === box ? 'ring-2 ring-white scale-125' : ''}`}
            title={`Box ${b}`}
          />
        ))}
      </div>
    );
  };

  // Stat card
  const StatCard = ({ title, value, icon, color = 'blue' }) => (
    <div className={`bg-gradient-to-br from-${color}-50 to-${color}-100 p-4 rounded-2xl shadow-lg border-2 border-${color}-200`}>
      <div className="text-2xl mb-1">{icon}</div>
      <div className="text-2xl font-bold text-gray-800">{value}</div>
      <div className="text-xs text-gray-600">{title}</div>
    </div>
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin text-4xl">🔄</div>
        <p className="ml-3 text-gray-700 text-lg font-semibold">Loading Review Data...</p>
      </div>
    );
  }

  return (
    <div className="p-4 space-y-4">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-2xl font-bold text-gray-800 mb-1">
          🧠 පුනරාවර්තන සමාලෝචනය
        </h2>
        <p className="text-gray-500 text-sm">Spaced Repetition Review</p>
      </div>

      {/* View Tabs */}
      <div className="flex gap-2 justify-center">
        {[
          { key: 'overview', label: '📋 Overview' },
          { key: 'review', label: '🎯 Review' },
          { key: 'stats', label: '📊 Statistics' }
        ].map(tab => (
          <button
            key={tab.key}
            onClick={() => setActiveView(tab.key)}
            className={`px-4 py-2 rounded-full text-sm font-bold transition-all ${
              activeView === tab.key
                ? 'bg-white text-purple-700 shadow-lg scale-105'
                : 'bg-white text-gray-600 shadow hover:bg-gray-50'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* OVERVIEW VIEW */}
      {activeView === 'overview' && (
        <div className="space-y-4">
          {/* Quick Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <StatCard
              title="Total Words"
              value={statistics?.statistics?.total_words || 0}
              icon="📚"
              color="blue"
            />
            <StatCard
              title="Mastered"
              value={statistics?.statistics?.words_learned || 0}
              icon="✅"
              color="green"
            />
            <StatCard
              title="Due Now"
              value={dueWords.length}
              icon="⏰"
              color="orange"
            />
            <StatCard
              title="Struggling"
              value={statistics?.statistics?.words_struggling || 0}
              icon="⚠️"
              color="red"
            />
          </div>

          {/* Due Words List */}
          {dueWords.length > 0 && (
            <div className="bg-white rounded-2xl p-4 shadow-md border border-gray-200">
              <h3 className="text-gray-800 font-bold mb-3">⏰ Due for Review</h3>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-2 max-h-48 overflow-y-auto">
                {dueWords.map((word, i) => (
                  <div key={i} className="bg-gray-50 rounded-xl p-2 flex items-center justify-between border border-gray-200">
                    <span className="text-gray-800 font-semibold text-sm">
                      {word.word || word}
                    </span>
                    {word.leitner_box && <LeitnerBoxIndicator box={word.leitner_box} />}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Start Review Button */}
          <button
            onClick={startSession}
            disabled={dueWords.length === 0}
            className="w-full py-3 bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white font-bold rounded-full text-lg shadow-lg transform hover:scale-105 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {dueWords.length > 0
              ? `🎯 Start Review (${dueWords.length} words)`
              : '✅ All caught up! No words due.'}
          </button>
        </div>
      )}

      {/* REVIEW VIEW */}
      {activeView === 'review' && sessionActive && currentWord && (
        <div className="space-y-4">
          {/* Progress bar */}
          <div className="bg-gray-200 rounded-full h-3 overflow-hidden">
            <div
              className="bg-gradient-to-r from-green-400 to-blue-500 h-full rounded-full transition-all duration-500"
              style={{ width: `${((currentIndex + 1) / dueWords.length) * 100}%` }}
            />
          </div>
          <p className="text-gray-500 text-center text-sm">
            {currentIndex + 1} / {dueWords.length}
          </p>

          {/* Word Card */}
          <div className={`bg-white rounded-3xl p-8 text-center shadow-xl border-2 ${
            showResult
              ? selectedQuality >= 3
                ? 'border-green-400 bg-green-50'
                : 'border-red-400 bg-red-50'
              : 'border-gray-200'
          } transition-all duration-300`}>
            <div className="text-5xl font-bold text-gray-800 mb-4">
              {currentWord.word}
            </div>
            
            {currentWord.phonemes && (
              <div className="flex gap-2 justify-center mb-3">
                {(Array.isArray(currentWord.phonemes) ? currentWord.phonemes : []).map((p, i) => (
                  <span key={i} className="px-2 py-1 bg-indigo-100 rounded-lg text-indigo-700 text-xs font-semibold">
                    {p}
                  </span>
                ))}
              </div>
            )}

            {currentWord.leitner_box && (
              <div className="flex justify-center mb-3">
                <LeitnerBoxIndicator box={currentWord.leitner_box} />
              </div>
            )}

            {showResult && (
              <div className={`text-lg font-bold ${selectedQuality >= 3 ? 'text-green-600' : 'text-red-600'} animate-bounce`}>
                {selectedQuality >= 3 ? '✅ නිවැරදියි!' : '❌ තවත් පුහුණු වෙන්න!'}
              </div>
            )}
          </div>

          {/* Quality Rating */}
          {!showResult && (
            <div className="space-y-2">
              <p className="text-gray-800 text-center font-semibold">
                ඔබට මතකද? (How well do you remember?)
              </p>
              <div className="grid grid-cols-3 md:grid-cols-6 gap-2">
                {qualityOptions.map(opt => (
                  <button
                    key={opt.value}
                    onClick={() => submitReview(opt.value)}
                    className={`p-3 rounded-2xl bg-white hover:bg-gray-50 border-2 border-gray-200 hover:border-indigo-300 transition-all transform hover:scale-105 text-center shadow-sm`}
                  >
                    <div className="text-2xl mb-1">{opt.label}</div>
                    <div className="text-gray-800 text-xs font-bold">{opt.desc}</div>
                    <div className="text-gray-500 text-[10px]">{opt.descEn}</div>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Review not active prompt */}
      {activeView === 'review' && !sessionActive && (
        <div className="text-center py-12">
          <div className="text-6xl mb-4">🎯</div>
          <p className="text-gray-700 text-lg mb-4">No active review session</p>
          <button
            onClick={startSession}
            className="px-8 py-3 bg-gradient-to-r from-purple-500 to-pink-500 text-white font-bold rounded-full shadow-lg hover:scale-105 transition-all"
          >
            Start Review
          </button>
        </div>
      )}

      {/* STATISTICS VIEW */}
      {activeView === 'stats' && (
        <div className="space-y-4">
          {/* Session Results */}
          {sessionResults.length > 0 && (
            <div className="bg-white rounded-2xl p-4 shadow-md border border-gray-200">
              <h3 className="text-gray-800 font-bold mb-3">📝 Session Results</h3>
              <div className="space-y-2">
                {sessionResults.map((r, i) => (
                  <div key={i} className="flex items-center justify-between bg-gray-50 rounded-xl p-3 border border-gray-200">
                    <span className="text-gray-800 font-semibold">{r.word}</span>
                    <div className="flex items-center gap-3">
                      <span className="text-gray-500 text-sm">{r.responseTime.toFixed(1)}s</span>
                      <span className="text-xl">{qualityOptions[r.quality]?.label || '❓'}</span>
                    </div>
                  </div>
                ))}
              </div>
              <div className="mt-3 text-center">
                <span className="text-gray-600 text-sm">
                  Average Quality: {(sessionResults.reduce((s, r) => s + r.quality, 0) / sessionResults.length).toFixed(1)} / 5
                </span>
              </div>
            </div>
          )}

          {/* Phoneme Mastery */}
          {statistics?.phoneme_mastery && Object.keys(statistics.phoneme_mastery).length > 0 && (
            <div className="bg-white rounded-2xl p-4 shadow-md border border-gray-200">
              <h3 className="text-gray-800 font-bold mb-3">🗣️ Phoneme Mastery</h3>
              <div className="grid grid-cols-3 md:grid-cols-5 gap-2">
                {Object.entries(statistics.phoneme_mastery).map(([phoneme, mastery]) => {
                  const pct = Math.round((mastery || 0) * 100);
                  const color = pct >= 80 ? 'green' : pct >= 50 ? 'yellow' : 'red';
                  return (
                    <div key={phoneme} className="text-center bg-gray-50 rounded-xl p-2 border border-gray-200">
                      <div className="text-gray-800 font-bold text-lg">{phoneme}</div>
                      <div className={`text-${color}-600 font-bold text-sm`}>{pct}%</div>
                      <div className={`w-full bg-gray-200 rounded-full h-1.5 mt-1`}>
                        <div
                          className={`bg-${color}-400 h-1.5 rounded-full transition-all`}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Forgetting Curves */}
          {statistics?.forgetting_curves && Object.keys(statistics.forgetting_curves).length > 0 && (
            <div className="bg-white rounded-2xl p-4 shadow-md border border-gray-200">
              <h3 className="text-gray-800 font-bold mb-3">📉 Forgetting Curves</h3>
              <div className="space-y-2 max-h-48 overflow-y-auto">
                {Object.entries(statistics.forgetting_curves).slice(0, 10).map(([word, curve]) => (
                  <div key={word} className="flex items-center gap-3 bg-gray-50 rounded-xl p-2 border border-gray-200">
                    <span className="text-gray-800 font-semibold w-24 truncate">{word}</span>
                    <div className="flex-1">
                      <div className="flex gap-0.5">
                        {(curve.retention_points || []).slice(0, 7).map((rp, i) => (
                          <div
                            key={i}
                            className="flex-1 rounded"
                            style={{
                              height: '20px',
                              backgroundColor: `rgba(${255 - Math.round(rp.retention * 255)}, ${Math.round(rp.retention * 255)}, 100, 0.8)`
                            }}
                            title={`Day ${rp.day}: ${Math.round(rp.retention * 100)}%`}
                          />
                        ))}
                      </div>
                    </div>
                    <span className="text-gray-600 text-xs w-12 text-right">
                      {Math.round((curve.current_retention || 0) * 100)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Overall Statistics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <StatCard
              title="Total Reviews"
              value={statistics?.statistics?.total_reviews || 0}
              icon="🔄"
              color="purple"
            />
            <StatCard
              title="Avg Easiness"
              value={(statistics?.statistics?.average_easiness || 2.5).toFixed(1)}
              icon="📊"
              color="blue"
            />
            <StatCard
              title="Retention Rate"
              value={`${Math.round((statistics?.statistics?.retention_rate || 0) * 100)}%`}
              icon="🧠"
              color="green"
            />
            <StatCard
              title="Learning Rate"
              value={`${Math.round((statistics?.statistics?.words_learned || 0) / Math.max(1, statistics?.statistics?.total_words || 1) * 100)}%`}
              icon="📈"
              color="teal"
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default SpacedRepetitionReview;
