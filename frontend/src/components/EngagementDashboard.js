import React, { useState, useEffect } from 'react';
import API_BASE_URL from '../config';
import { Line, Radar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  RadialLinearScale,
  Filler,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import './EngagementDashboard.css';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  RadialLinearScale,
  Filler,
  Title,
  Tooltip,
  Legend
);

/**
 * Real-time Engagement Dashboard Component
 * 
 * Features:
 * - Live engagement gauge (0-100)
 * - Timeline chart showing engagement over time
 * - Component breakdown (emotion, gesture, response time, attention)
 * - Risk alerts and intervention notifications
 * - Dropout risk prediction
 */
const EngagementDashboard = ({ userId, isVisible = true }) => {
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [interventionAlert, setInterventionAlert] = useState(null);

  // Fetch dashboard data
  useEffect(() => {
    if (!userId || !isVisible) return;

    const fetchDashboard = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/engagement-dashboard/${userId}`);
        const data = await response.json();
        
        if (data.error) {
          setError(data.error);
          setLoading(false);
          return;
        }
        
        setDashboardData(data);
        setLoading(false);
      } catch (err) {
        console.error('Error fetching dashboard:', err);
        setError('Failed to load engagement data');
        setLoading(false);
      }
    };

    fetchDashboard();
    
    // Refresh every 30 seconds
    const interval = setInterval(fetchDashboard, 30000);
    return () => clearInterval(interval);
  }, [userId, isVisible]);

  // Show intervention alert
  const showIntervention = (intervention) => {
    setInterventionAlert(intervention);
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
      setInterventionAlert(null);
    }, 5000);
  };

  if (!isVisible) return null;
  if (loading) return <div className="engagement-loading">Loading engagement data...</div>;
  if (error) return <div className="engagement-error">⚠️ {error}</div>;
  if (!dashboardData) return null;

  // Provide safe defaults to prevent NaN errors
  const { 
    statistics = { 
      average_engagement: 50, 
      total_signals: 0, 
      trend: 'stable' 
    }, 
    visualizations = {
      gauge: { score: 50, color: '#FFA500' },
      timeline: { labels: [], datasets: [] },
      components: { labels: [], datasets: [] }
    }, 
    dropout_prediction = { dropout_probability: 0, risk_level: 'low' }
  } = dashboardData;

  // Prepare timeline chart data
  const timelineData = {
    labels: visualizations.timeline.labels,
    datasets: visualizations.timeline.datasets.map(ds => ({
      ...ds,
      borderColor: ds.borderColor || '#4285F4',
      backgroundColor: ds.backgroundColor || 'rgba(66, 133, 244, 0.1)',
      tension: 0.4
    }))
  };

  // Prepare radar chart data
  const radarData = {
    labels: visualizations.components.labels,
    datasets: [{
      label: 'Current Session',
      data: visualizations.components.datasets[0].data,
      backgroundColor: 'rgba(66, 133, 244, 0.2)',
      borderColor: '#4285F4',
      pointBackgroundColor: '#4285F4',
      pointBorderColor: '#fff',
      pointHoverBackgroundColor: '#fff',
      pointHoverBorderColor: '#4285F4'
    }]
  };

  // Get gauge color
  const getGaugeColor = (score) => {
    if (score >= 70) return '#00C851';
    if (score >= 40) return '#FFBB33';
    return '#FF4444';
  };

  const gaugeScore = isNaN(visualizations.gauge.score) || visualizations.gauge.score === null || visualizations.gauge.score === undefined 
    ? 50 
    : visualizations.gauge.score;
  const gaugeColor = getGaugeColor(gaugeScore);

  return (
    <div className="engagement-dashboard">
      {/* Intervention Alert */}
      {interventionAlert && (
        <div className={`intervention-alert ${interventionAlert.type}`}>
          <div className="alert-content">
            <span className="alert-icon">
              {interventionAlert.type === 'break' && '☕'}
              {interventionAlert.type === 'reward' && '⭐'}
              {interventionAlert.type === 'difficulty_adjust' && '🎯'}
              {interventionAlert.type === 'encouragement' && '💪'}
            </span>
            <span className="alert-message">{interventionAlert.message}</span>
          </div>
        </div>
      )}

      {/* Top Row: Gauge + Stats */}
      <div className="dashboard-row">
        {/* Engagement Gauge */}
        <div className="dashboard-card gauge-card">
          <h3>Real-Time Engagement</h3>
          <div className="gauge-container">
            <svg viewBox="0 0 200 120" className="gauge-svg">
              {/* Background arc */}
              <path
                d="M 20 100 A 80 80 0 0 1 180 100"
                fill="none"
                stroke="#e0e0e0"
                strokeWidth="15"
                strokeLinecap="round"
              />
              {/* Colored arc based on score */}
              <path
                d="M 20 100 A 80 80 0 0 1 180 100"
                fill="none"
                stroke={gaugeColor}
                strokeWidth="15"
                strokeLinecap="round"
                strokeDasharray={`${(gaugeScore / 100) * 251} 251`}
              />
              {/* Score text */}
              <text x="100" y="85" textAnchor="middle" fontSize="32" fontWeight="bold" fill={gaugeColor}>
                {Math.round(gaugeScore)}
              </text>
              <text x="100" y="105" textAnchor="middle" fontSize="12" fill="#666">
                Engagement
              </text>
            </svg>
            
            <div className="gauge-info">
              <div className="trend-indicator">
                Trend: {visualizations.gauge.trend_arrow} {statistics.current_trend}
              </div>
              <div className={`risk-badge ${visualizations.gauge.risk_level}`}>
                {visualizations.gauge.risk_level.toUpperCase()} RISK
              </div>
            </div>
          </div>
        </div>

        {/* Statistics Card */}
        <div className="dashboard-card stats-card">
          <h3>Session Statistics</h3>
          <div className="stats-grid">
            <div className="stat-item">
              <span className="stat-label">Average</span>
              <span className="stat-value">
                {isNaN(statistics.average_engagement) || statistics.average_engagement === null || statistics.average_engagement === undefined 
                  ? 50 
                  : Math.round(statistics.average_engagement)}/100
              </span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Total Signals</span>
              <span className="stat-value">{statistics.total_signals || 0}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Trend</span>
              <span className="stat-value">{statistics.current_trend || 'stable'}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Risk Level</span>
              <span className={`stat-value risk-${statistics.current_risk || 'low'}`}>
                {statistics.current_risk || 'low'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Middle Row: Timeline Chart */}
      <div className="dashboard-row">
        <div className="dashboard-card timeline-card">
          <h3>Engagement Timeline</h3>
          <div className="chart-container">
            <Line 
              data={timelineData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                  legend: {
                    display: true,
                    position: 'top'
                  },
                  tooltip: {
                    mode: 'index',
                    intersect: false
                  }
                },
                scales: {
                  y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                      display: true,
                      text: 'Engagement Score'
                    }
                  }
                }
              }}
            />
          </div>
        </div>
      </div>

      {/* Bottom Row: Component Breakdown + Risk Dashboard */}
      <div className="dashboard-row">
        {/* Component Breakdown */}
        <div className="dashboard-card radar-card">
          <h3>Component Breakdown</h3>
          <div className="chart-container">
            <Radar 
              data={radarData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                  r: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                      stepSize: 20
                    }
                  }
                },
                plugins: {
                  legend: {
                    display: false
                  }
                }
              }}
            />
          </div>
        </div>

        {/* Risk Dashboard */}
        <div className="dashboard-card risk-card">
          <h3>Dropout Risk Assessment</h3>
          <div className="risk-content">
            <div className="risk-score-container">
              <div className={`risk-score-circle ${dropout_prediction.risk_level}`}>
                <span className="risk-score">{Math.round(dropout_prediction.risk_score)}</span>
                <span className="risk-label">Risk Score</span>
              </div>
            </div>
            
            <div className="risk-details">
              <div className={`risk-level-badge ${dropout_prediction.risk_level}`}>
                {dropout_prediction.risk_level.toUpperCase()}
              </div>
              
              {dropout_prediction.risk_factors.length > 0 ? (
                <div className="risk-factors">
                  <h4>Risk Factors:</h4>
                  <ul>
                    {dropout_prediction.risk_factors.map((factor, idx) => (
                      <li key={idx}>{factor.replace(/_/g, ' ')}</li>
                    ))}
                  </ul>
                </div>
              ) : (
                <div className="risk-good">
                  ✅ No significant risk factors detected
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EngagementDashboard;
