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

        setAttentionData(heatmapData);
        setRecommendations(recData.recommendations || []);
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
      <div className="attention-dashboard-loading">
        <div className="loading-spinner"></div>
        <p>Loading attention analytics...</p>
      </div>
    );
  }

  if (!attentionData) {
    return (
      <div className="attention-dashboard-empty">
        <p>No attention data available yet. Start tracking gaze to see analytics.</p>
      </div>
    );
  }

  const focusData = attentionData.visualizations?.focus_dashboard || {};
  const timeline = attentionData.visualizations?.attention_timeline || {};

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
    plugins: {
      legend: {
        display: true,
        position: 'top'
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100
      }
    }
  };

  return (
    <div className="attention-dashboard">
      <h1 className="dashboard-title">Visual Attention Analytics</h1>

      {/* Focus Quality Card */}
      <div className="dashboard-grid">
        <div className="dashboard-card focus-quality-card">
          <h2>Focus Quality</h2>
          <div className="focus-gauge">
            <svg viewBox="0 0 200 120" className="gauge-svg">
              <path
                d="M 20 100 A 80 80 0 0 1 180 100"
                fill="none"
                stroke="#e0e0e0"
                strokeWidth="20"
                strokeLinecap="round"
              />
              <path
                d="M 20 100 A 80 80 0 0 1 180 100"
                fill="none"
                stroke={getColorForScore(focusData.focus_quality || 0)}
                strokeWidth="20"
                strokeLinecap="round"
                strokeDasharray={`${((focusData.focus_quality || 0) / 100) * 251.2} 251.2`}
                className="gauge-arc"
              />
              <text x="100" y="90" textAnchor="middle" className="gauge-score">
                {Math.round(focusData.focus_quality || 0)}
              </text>
              <text x="100" y="110" textAnchor="middle" className="gauge-label">
                {focusData.quality_label || 'N/A'}
              </text>
            </svg>
          </div>

          <div className="focus-metrics">
            <div className="metric">
              <span className="metric-label">Total Fixation Time</span>
              <span className="metric-value">{((focusData.total_fixation_time || 0) / 1000).toFixed(1)}s</span>
            </div>
            <div className="metric">
              <span className="metric-label">Drift Severity</span>
              <span className={`metric-value drift-${(focusData.drift_severity || 'Low').toLowerCase()}`}>
                {focusData.drift_severity || 'Low'}
              </span>
            </div>
          </div>
        </div>

        {/* Attention Timeline */}
        <div className="dashboard-card timeline-card">
          <h2>Attention Timeline</h2>
          <div className="chart-container">
            {timeline.labels && timeline.labels.length > 0 ? (
              <Line data={timelineChartData} options={timelineOptions} />
            ) : (
              <p className="no-data">Not enough data for timeline</p>
            )}
          </div>
        </div>

        {/* Top Attention Zones */}
        <div className="dashboard-card zones-card">
          <h2>Top Attention Zones</h2>
          <div className="zones-list">
            {focusData.top_zones && focusData.top_zones.length > 0 ? (
              focusData.top_zones.slice(0, 5).map((zone, idx) => (
                <div key={zone.zone_id} className="zone-item">
                  <div className="zone-rank">#{idx + 1}</div>
                  <div className="zone-info">
                    <div className="zone-id">{zone.zone_id}</div>
                    <div className="zone-stats">
                      <span>Score: {zone.score?.toFixed(2) || 0}</span>
                      <span>Visits: {zone.visits || 0}</span>
                    </div>
                  </div>
                  <div className="zone-bar">
                    <div 
                      className="zone-bar-fill" 
                      style={{ width: `${(zone.score || 0) * 100}%` }}
                    ></div>
                  </div>
                </div>
              ))
            ) : (
              <p className="no-data">No attention zones recorded yet</p>
            )}
          </div>
        </div>

        {/* UI Recommendations */}
        <div className="dashboard-card recommendations-card">
          <h2>UI Optimization Recommendations</h2>
          <div className="recommendations-list">
            {recommendations.length > 0 ? (
              recommendations.map((rec, idx) => (
                <div key={idx} className={`recommendation-item severity-${rec.severity}`}>
                  <div className="recommendation-icon">
                    {rec.severity === 'high' && '🔴'}
                    {rec.severity === 'medium' && '🟡'}
                    {rec.severity === 'low' && '🟢'}
                  </div>
                  <div className="recommendation-content">
                    <div className="recommendation-type">{rec.type?.replace(/_/g, ' ')}</div>
                    <div className="recommendation-message">{rec.message}</div>
                  </div>
                </div>
              ))
            ) : (
              <p className="no-data">No recommendations at this time</p>
            )}
          </div>
        </div>
      </div>

      {/* Statistics Summary */}
      <div className="stats-summary">
        <div className="stat-box">
          <div className="stat-value">{attentionData.heatmap_data?.statistics?.total_gaze_points || 0}</div>
          <div className="stat-label">Gaze Points</div>
        </div>
        <div className="stat-box">
          <div className="stat-value">{attentionData.heatmap_data?.hotspots?.length || 0}</div>
          <div className="stat-label">Hotspots</div>
        </div>
        <div className="stat-box">
          <div className="stat-value">{((attentionData.heatmap_data?.statistics?.total_fixation_time || 0) / 1000).toFixed(1)}s</div>
          <div className="stat-label">Fixation Time</div>
        </div>
        <div className="stat-box">
          <div className="stat-value">{attentionData.heatmap_data?.statistics?.drift_events || 0}</div>
          <div className="stat-label">Drift Events</div>
        </div>
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
