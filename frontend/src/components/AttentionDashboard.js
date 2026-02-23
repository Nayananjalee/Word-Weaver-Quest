import React, { useState, useEffect } from 'react';
import API_BASE_URL from '../config';
import { Line } from 'react-chartjs-2';
import './AttentionDashboard.css';

/**
 * AttentionDashboard Component
 * 
 * Analytics dashboard for visual attention tracking.
 * Displays focus quality metrics, attention timeline, and heatmap insights.
 */

const AttentionDashboard = ({ userId }) => {
  const [attentionData, setAttentionData] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

  // Fetch attention data
  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch heatmap data
        const heatmapResponse = await fetch(`${API_BASE_URL}/attention-heatmap/${userId}`);
        const heatmapData = await heatmapResponse.json();

        // Fetch recommendations
        const recResponse = await fetch(`${API_BASE_URL}/attention-recommendations/${userId}`);
        const recData = await recResponse.json();

        // Only set data if no error returned
        if (!heatmapData.error) {
          setAttentionData(heatmapData);
        }

        // Recommendations may be an object {placement, content, intervention} — convert to array
        const rawRec = recData.recommendations || {};
        if (Array.isArray(rawRec)) {
          setRecommendations(rawRec);
        } else {
          const recArray = Object.entries(rawRec).map(([type, message]) => ({
            type,
            message: typeof message === 'string' ? message : String(message),
            severity: 'medium'
          }));
          setRecommendations(recArray);
        }

        setIsLoading(false);
      } catch (err) {
        console.error('Failed to fetch attention data:', err);
        setIsLoading(false);
      }
    };

    if (userId) {
      fetchData();
      const interval = setInterval(fetchData, 30000); // Update every 30s
      return () => clearInterval(interval);
    }
  }, [userId]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="text-4xl animate-spin mb-3">👁️</div>
          <p className="text-white font-semibold animate-pulse">Loading attention analytics...</p>
        </div>
      </div>
    );
  }

  const focusData = attentionData?.visualizations?.focus_dashboard || {};
  const timeline = attentionData?.visualizations?.attention_timeline || {};

  // Chart.js data for attention timeline
  const timelineChartData = {
    labels: timeline.labels || [],
    datasets: [
      {
        label: 'Attention Score',
        data: timeline.attention_scores || [],
        borderColor: '#4CAF50',
        backgroundColor: 'rgba(76, 175, 80, 0.1)',
        tension: 0.4,
        fill: true
      },
      {
        label: 'Gaze Confidence',
        data: timeline.confidence_scores || [],
        borderColor: '#2196F3',
        backgroundColor: 'rgba(33, 150, 243, 0.1)',
        tension: 0.4,
        fill: true
      }
    ]
  };

  const timelineOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: true, position: 'top' } },
    scales: { y: { beginAtZero: true, max: 100 } }
  };

  return (
    <div className="p-4 space-y-4">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-2xl font-bold text-white mb-1">👁️ දෘශ්‍ය අවධානය</h2>
        <p className="text-white/80 text-sm">Visual Attention Analytics</p>
      </div>

      {/* Gaze data status banner */}
      {!attentionData ? (
        <div className="bg-blue-500/20 backdrop-blur rounded-2xl p-4 border border-blue-400/30 text-center">
          <div className="text-3xl mb-2">🎮</div>
          <p className="text-white font-semibold">Attention tracking active!</p>
          <p className="text-white/70 text-sm mt-1">
            Gaze data is being recorded as you play. Complete a few sessions and check back here.
          </p>
          <p className="text-white/50 text-xs mt-1">
            දෘශ්‍ය දත්ත රේකෝඩ් කරමින් සිටී. සෙල්ලම් කිරීමෙන් පසු නැවත පරීක්ෂා කරන්න.
          </p>
        </div>
      ) : (
        <>
          {/* Focus Quality Card */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="bg-white/10 backdrop-blur rounded-2xl p-4 text-center border border-white/20">
              <div className="text-2xl mb-1">🎯</div>
              <div className="text-2xl font-bold text-white">{Math.round(focusData.focus_quality || 0)}%</div>
              <div className="text-white/70 text-xs">Focus Quality</div>
              <div className="text-white/50 text-xs">{focusData.quality_label || ''}</div>
            </div>
            <div className="bg-white/10 backdrop-blur rounded-2xl p-4 text-center border border-white/20">
              <div className="text-2xl mb-1">⏱️</div>
              <div className="text-2xl font-bold text-white">{((focusData.total_fixation_time || 0) / 1000).toFixed(1)}s</div>
              <div className="text-white/70 text-xs">Fixation Time</div>
            </div>
            <div className="bg-white/10 backdrop-blur rounded-2xl p-4 text-center border border-white/20">
              <div className="text-2xl mb-1">👁️</div>
              <div className="text-2xl font-bold text-white">{attentionData.heatmap_data?.statistics?.total_gaze_points || 0}</div>
              <div className="text-white/70 text-xs">Gaze Points</div>
            </div>
            <div className="bg-white/10 backdrop-blur rounded-2xl p-4 text-center border border-white/20">
              <div className="text-2xl mb-1">🌊</div>
              <div className={`text-lg font-bold ${focusData.drift_severity === 'High' ? 'text-red-300' : focusData.drift_severity === 'Medium' ? 'text-yellow-300' : 'text-green-300'}`}>
                {focusData.drift_severity || 'Low'}
              </div>
              <div className="text-white/70 text-xs">Drift Severity</div>
            </div>
          </div>

          {/* Attention Timeline */}
          {timeline.labels && timeline.labels.length > 0 && (
            <div className="bg-white/10 backdrop-blur rounded-2xl p-4 border border-white/20">
              <h3 className="text-white font-bold mb-3">📈 Attention Timeline</h3>
              <div style={{ height: '180px' }}>
                <Line data={timelineChartData} options={timelineOptions} />
              </div>
            </div>
          )}

          {/* Top Attention Zones */}
          {focusData.top_zones && focusData.top_zones.length > 0 && (
            <div className="bg-white/10 backdrop-blur rounded-2xl p-4 border border-white/20">
              <h3 className="text-white font-bold mb-3">🗺️ Top Attention Zones</h3>
              <div className="space-y-2">
                {focusData.top_zones.slice(0, 5).map((zone, idx) => (
                  <div key={zone.zone_id} className="flex items-center gap-3">
                    <span className="text-white/60 text-xs w-6">#{idx + 1}</span>
                    <span className="text-white text-sm font-medium w-24">{zone.zone_id}</span>
                    <div className="flex-1 bg-white/10 rounded-full h-2">
                      <div className="bg-blue-400 h-2 rounded-full" style={{ width: `${Math.min(100, (zone.score || 0) * 100)}%` }} />
                    </div>
                    <span className="text-white/60 text-xs w-12 text-right">{zone.visits || 0} visits</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}

      {/* UI Recommendations — always shown */}
      <div className="bg-white/10 backdrop-blur rounded-2xl p-4 border border-white/20">
        <h3 className="text-white font-bold mb-3">💡 UI Recommendations / අෙනු දෙස</h3>
        {recommendations.length > 0 ? (
          <div className="space-y-3">
            {recommendations.map((rec, idx) => (
              <div key={idx} className={`flex items-start gap-3 p-3 rounded-xl ${
                rec.severity === 'high' ? 'bg-red-500/20 border border-red-400/30' :
                rec.severity === 'medium' ? 'bg-yellow-500/20 border border-yellow-400/30' :
                'bg-green-500/20 border border-green-400/30'
              }`}>
                <span className="text-xl flex-shrink-0">
                  {rec.severity === 'high' ? '🔴' : rec.severity === 'medium' ? '🟡' : '🟢'}
                </span>
                <div>
                  <div className="text-white/80 text-xs font-bold uppercase tracking-wide mb-0.5">
                    {rec.type?.replace(/_/g, ' ')}
                  </div>
                  <div className="text-white text-sm">{rec.message}</div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-white/60 text-sm text-center py-2">
            No recommendations yet — keep playing to generate insights!
          </p>
        )}
      </div>
    </div>
  );
};

// Helper function to get color based on score
const getColorForScore = (score) => {
  if (score >= 80) return '#4CAF50'; // Excellent - Green
  if (score >= 60) return '#8BC34A'; // Good - Light Green
  if (score >= 40) return '#FFC107'; // Fair - Yellow
  return '#F44336'; // Poor - Red
};

export default AttentionDashboard;
