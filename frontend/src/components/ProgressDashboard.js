/**
 * ProgressDashboard - Learning Progress & Analytics Component
 * 
 * Displays comprehensive learning analytics including:
 * - Session performance metrics (LEI, Flow Ratio, ZPD alignment)
 * - Learning trajectory visualization
 * - Word mastery tracking
 * - Therapist recommendations (bilingual)
 * 
 * Research basis:
 * - Plass & Pawar (2020): Micro/Macro Adaptivity Framework
 * - Khosravi et al. (2022): Explainable AI for Education
 * - Lim et al. (2023): Game-Based Learning Analytics
 */

import React, { useState, useEffect, useCallback } from 'react';
import API_BASE_URL from '../config';

const ProgressDashboard = ({ userId }) => {
  const [trajectoryData, setTrajectoryData] = useState(null);
  const [sessionMetrics, setSessionMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeSection, setActiveSection] = useState('overview');

  const fetchData = useCallback(async () => {
    if (!userId) return;
    setLoading(true);

    try {
      // Fetch learning trajectory
      const trajRes = await fetch(`${API_BASE_URL}/learning-trajectory/${userId}`);
      const trajData = await trajRes.json();
      if (!trajData.error) {
        setTrajectoryData(trajData);
      }

      // Fetch real-time session metrics
      const metricsRes = await fetch(`${API_BASE_URL}/session/metrics/${userId}`);
      const metricsData = await metricsRes.json();
      if (!metricsData.error) {
        setSessionMetrics(metricsData);
      }
    } catch (err) {
      setError('Failed to load analytics data');
      console.error('ProgressDashboard error:', err);
    } finally {
      setLoading(false);
    }
  }, [userId]);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, [fetchData]);

  // Metric card component
  const MetricCard = ({ title, value, subtitle, icon, color = 'blue' }) => (
    <div className={`bg-gradient-to-br from-${color}-50 to-${color}-100 p-4 rounded-2xl shadow-lg border-2 border-${color}-200 transition-transform hover:scale-105`}>
      <div className="text-2xl mb-1">{icon}</div>
      <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">{title}</p>
      <p className={`text-2xl font-bold text-${color}-600`}>{value}</p>
      {subtitle && <p className="text-xs text-gray-500 mt-1">{subtitle}</p>}
    </div>
  );

  // Progress bar component  
  const ProgressBar = ({ value, max = 100, color = '#4285F4', label }) => (
    <div className="mb-3">
      <div className="flex justify-between text-xs mb-1">
        <span className="font-semibold text-gray-600">{label}</span>
        <span className="font-bold" style={{ color }}>{Math.round(value)}%</span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
        <div 
          className="h-full rounded-full transition-all duration-1000 ease-out"
          style={{ 
            width: `${Math.min(100, Math.max(0, value))}%`, 
            backgroundColor: color,
            boxShadow: `0 0 8px ${color}40`
          }}
        />
      </div>
    </div>
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="text-4xl animate-spin mb-3">📊</div>
          <p className="text-gray-700 font-semibold animate-pulse">Analytics Loading...</p>
        </div>
      </div>
    );
  }

  if (error && !trajectoryData && !sessionMetrics) {
    return (
      <div className="bg-white/90 rounded-2xl p-8 text-center">
        <div className="text-4xl mb-3">📈</div>
        <h3 className="text-xl font-bold text-gray-700 mb-2">No Data Yet</h3>
        <p className="text-gray-500">Play some stories to see your learning analytics!</p>
        <p className="text-gray-400 text-sm mt-2">
          ඔබේ ඉගෙනුම් ප්‍රගතිය බැලීමට කතා කිහිපයක් සෙල්ලම් කරන්න!
        </p>
      </div>
    );
  }

  return (
    <div className="h-full overflow-auto space-y-4 pb-4">
      {/* Section Toggle */}
      <div className="flex gap-2 justify-center flex-wrap">
        {[
          { key: 'overview', label: '📊 Overview', labelSi: 'සාරාංශය' },
          { key: 'metrics', label: '🧪 Metrics', labelSi: 'මිණුම්' },
          { key: 'words', label: '📚 Words', labelSi: 'වචන' },
          { key: 'recommendations', label: '💡 Tips', labelSi: 'උපදෙස්' }
        ].map(tab => (
          <button
            key={tab.key}
            onClick={() => setActiveSection(tab.key)}
            className={`px-4 py-2 rounded-full text-sm font-bold transition-all ${
              activeSection === tab.key
                ? 'bg-gradient-to-r from-indigo-500 to-purple-500 text-white shadow-lg scale-105'
                : 'bg-white/70 text-gray-600 hover:bg-white/90 shadow'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* OVERVIEW SECTION */}
      {activeSection === 'overview' && (
        <div className="space-y-4">
          {/* Summary Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <MetricCard
              title="Sessions"
              value={trajectoryData?.total_sessions || 0}
              subtitle="සම්පූර්ණ සැසි"
              icon="🎮"
              color="blue"
            />
            <MetricCard
              title="Overall Accuracy"
              value={`${Math.round((trajectoryData?.overall_accuracy || 0) * 100)}%`}
              subtitle="සමස්ත නිරවද්‍යතාව"
              icon="🎯"
              color="green"
            />
            <MetricCard
              title="Total Stars"
              value={trajectoryData?.total_stars || 0}
              subtitle="මුළු තරු"
              icon="⭐"
              color="yellow"
            />
            <MetricCard
              title="Learning Rate"
              value={trajectoryData?.learning_rate > 0 ? '↑' : trajectoryData?.learning_rate < 0 ? '↓' : '→'}
              subtitle={trajectoryData?.accuracy_trend || 'stable'}
              icon="📈"
              color="purple"
            />
          </div>

          {/* Engagement & Accuracy Trends */}
          <div className="bg-white/90 rounded-2xl p-4 shadow-lg">
            <h3 className="font-bold text-gray-700 mb-3 flex items-center gap-2">
              <span>📉</span> Learning Trends / ඉගෙනුම් ප්‍රවණතා
            </h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <ProgressBar 
                  label="Accuracy Trend / නිරවද්‍යතාව"
                  value={(trajectoryData?.overall_accuracy || 0.5) * 100}
                  color={trajectoryData?.accuracy_trend === 'improving' ? '#10B981' : 
                         trajectoryData?.accuracy_trend === 'declining' ? '#EF4444' : '#F59E0B'}
                />
                <p className="text-xs text-center font-semibold" style={{
                  color: trajectoryData?.accuracy_trend === 'improving' ? '#10B981' : 
                         trajectoryData?.accuracy_trend === 'declining' ? '#EF4444' : '#F59E0B'
                }}>
                  {trajectoryData?.accuracy_trend === 'improving' ? '📈 Improving / වැඩිවෙමින්' :
                   trajectoryData?.accuracy_trend === 'declining' ? '📉 Needs Support / සහාය අවශ්‍යයි' :
                   '➡️ Stable / ස්ථාවරයි'}
                </p>
              </div>
              <div>
                <ProgressBar
                  label="Engagement Trend / සහභාගීත්වය"
                  value={70}
                  color={trajectoryData?.engagement_trend === 'improving' ? '#10B981' :
                         trajectoryData?.engagement_trend === 'declining' ? '#EF4444' : '#F59E0B'}
                />
                <p className="text-xs text-center font-semibold" style={{
                  color: trajectoryData?.engagement_trend === 'improving' ? '#10B981' :
                         trajectoryData?.engagement_trend === 'declining' ? '#EF4444' : '#F59E0B'
                }}>
                  {trajectoryData?.engagement_trend === 'improving' ? '🔥 Engaged / සහභාගි' :
                   trajectoryData?.engagement_trend === 'declining' ? '😴 Low / අඩුයි' :
                   '😊 Good / හොඳයි'}
                </p>
              </div>
            </div>
          </div>

          {/* ZPD Recommendation */}
          {trajectoryData?.zpd_recommendation && (
            <div className={`rounded-2xl p-4 shadow-lg border-2 ${
              trajectoryData.zpd_recommendation === 'increase_difficulty' 
                ? 'bg-green-50 border-green-300' 
                : trajectoryData.zpd_recommendation === 'decrease_difficulty'
                  ? 'bg-red-50 border-red-300'
                  : 'bg-blue-50 border-blue-300'
            }`}>
              <h3 className="font-bold text-gray-700 mb-1 flex items-center gap-2">
                <span>🎓</span> Zone of Proximal Development
              </h3>
              <p className="text-sm text-gray-600">
                {trajectoryData.zpd_recommendation === 'increase_difficulty' 
                  ? '✅ ළමයා සූදානමයි! අභියෝගය වැඩි කරන්න. / Ready for more challenge!' 
                  : trajectoryData.zpd_recommendation === 'decrease_difficulty'
                    ? '⚠️ දුෂ්කරතාව අඩු කරන්න. / Consider reducing difficulty.'
                    : '👍 වර්තමාන මට්ටම හොඳයි. / Current level is optimal.'}
              </p>
            </div>
          )}
        </div>
      )}

      {/* METRICS SECTION */}
      {activeSection === 'metrics' && sessionMetrics && (
        <div className="space-y-4">
          <div className="bg-white/90 rounded-2xl p-4 shadow-lg">
            <h3 className="font-bold text-gray-700 mb-3 flex items-center gap-2">
              <span>🧪</span> Research Metrics / පර්යේෂණ මිණුම්
            </h3>
            <div className="space-y-3">
              <ProgressBar 
                label="Learning Efficiency Index (LEI)"
                value={Math.min(100, (sessionMetrics.metrics?.learning_efficiency || 0) * 100)}
                color="#4285F4"
              />
              <ProgressBar
                label="Flow State Ratio (Csikszentmihalyi)"
                value={(sessionMetrics.metrics?.flow_ratio || 0) * 100}
                color="#9333EA"
              />
              <ProgressBar
                label="ZPD Alignment (Vygotsky)"
                value={(sessionMetrics.metrics?.zpd_alignment || 0) * 100}
                color="#10B981"
              />
              <ProgressBar
                label="Resilience Score (Kapur, 2020)"
                value={(sessionMetrics.metrics?.resilience || 0) * 100}
                color="#F59E0B"
              />
              <ProgressBar
                label="Engagement Consistency"
                value={(sessionMetrics.metrics?.engagement_consistency || 0) * 100}
                color="#EC4899"
              />
            </div>
          </div>

          {/* Streak & Performance */}
          <div className="grid grid-cols-2 gap-3">
            <MetricCard
              title="Current Streak"
              value={sessionMetrics.metrics?.current_streak || 0}
              subtitle="වර්තමාන ධාවනය"
              icon="🔥"
              color="orange"
            />
            <MetricCard
              title="Best Streak"
              value={sessionMetrics.metrics?.best_streak || 0}
              subtitle="හොඳම ධාවනය"
              icon="🏆"
              color="yellow"
            />
          </div>

          {/* Behavioral Indicators */}
          <div className="bg-white/90 rounded-2xl p-4 shadow-lg">
            <h3 className="font-bold text-gray-700 mb-2">⚡ Behavioral Indicators</h3>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="flex items-center gap-2">
                <span className={sessionMetrics.metrics?.frustration_episodes > 0 ? 'text-red-500' : 'text-green-500'}>
                  {sessionMetrics.metrics?.frustration_episodes > 0 ? '😤' : '😊'}
                </span>
                <span>Frustration: {sessionMetrics.metrics?.frustration_episodes || 0}</span>
              </div>
              <div className="flex items-center gap-2">
                <span className={sessionMetrics.metrics?.boredom_episodes > 0 ? 'text-yellow-500' : 'text-green-500'}>
                  {sessionMetrics.metrics?.boredom_episodes > 0 ? '😴' : '🎯'}
                </span>
                <span>Boredom: {sessionMetrics.metrics?.boredom_episodes || 0}</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* WORDS SECTION */}
      {activeSection === 'words' && (
        <div className="space-y-4">
          <div className="bg-white/90 rounded-2xl p-4 shadow-lg">
            <h3 className="font-bold text-gray-700 mb-3 flex items-center gap-2">
              <span>📚</span> Word Mastery / වචන ප්‍රගතිය
            </h3>
            
            {trajectoryData?.word_mastery && Object.keys(trajectoryData.word_mastery).length > 0 ? (
              <div className="space-y-2">
                {Object.entries(trajectoryData.word_mastery)
                  .sort(([,a], [,b]) => b - a)
                  .map(([word, mastery]) => (
                    <div key={word} className="flex items-center gap-3">
                      <span className="text-lg font-bold w-24 truncate">{word}</span>
                      <div className="flex-1">
                        <ProgressBar
                          label=""
                          value={mastery}
                          color={mastery >= 80 ? '#10B981' : mastery >= 50 ? '#F59E0B' : '#EF4444'}
                        />
                      </div>
                      <span className="text-xs">
                        {mastery >= 80 ? '✅' : mastery >= 50 ? '📖' : '🔄'}
                      </span>
                    </div>
                  ))
                }
              </div>
            ) : (
              <p className="text-gray-500 text-center py-4">
                No word data yet. Play some stories! / වචන දත්ත නැත.
              </p>
            )}
          </div>
        </div>
      )}

      {/* RECOMMENDATIONS SECTION */}
      {activeSection === 'recommendations' && (
        <div className="space-y-4">
          {trajectoryData?.recommendations && trajectoryData.recommendations.length > 0 ? (
            trajectoryData.recommendations.map((rec, idx) => (
              <div 
                key={idx}
                className={`rounded-2xl p-4 shadow-lg border-l-4 ${
                  rec.priority === 'high' ? 'bg-red-50 border-red-500' :
                  rec.priority === 'medium' ? 'bg-yellow-50 border-yellow-500' :
                  'bg-blue-50 border-blue-500'
                }`}
              >
                <div className="flex items-start gap-3">
                  <span className="text-2xl">
                    {rec.type === 'difficulty' ? '🎚️' :
                     rec.type === 'engagement' ? '💪' :
                     rec.type === 'duration' ? '⏱️' :
                     rec.type === 'emotional' ? '❤️' : '💡'}
                  </span>
                  <div>
                    <p className="font-bold text-sm text-gray-700">
                      [{rec.priority.toUpperCase()}] {rec.message_en}
                    </p>
                    <p className="text-sm text-gray-600 mt-1">{rec.message}</p>
                  </div>
                </div>
              </div>
            ))
          ) : (
            <div className="bg-white/90 rounded-2xl p-6 text-center shadow-lg">
              <div className="text-4xl mb-2">🎉</div>
              <h3 className="font-bold text-gray-700">Great Progress!</h3>
              <p className="text-gray-500 text-sm mt-1">
                No recommendations at this time. Keep going!
              </p>
              <p className="text-gray-400 text-sm mt-1">
                දැනට නිර්දේශ නැත. දිගටම කරගෙන යන්න!
              </p>
            </div>
          )}

          {/* Frustration Summary */}
          {trajectoryData?.total_frustration_episodes > 0 && (
            <div className="bg-orange-50 rounded-2xl p-4 shadow-lg border-2 border-orange-200">
              <h3 className="font-bold text-orange-700 mb-1">⚠️ Frustration Alert</h3>
              <p className="text-sm text-orange-600">
                {trajectoryData.total_frustration_episodes} frustration episode(s) detected across sessions.
                Consider more encouragement and adaptive breaks.
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ProgressDashboard;
