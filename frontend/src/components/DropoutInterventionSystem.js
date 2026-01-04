/**
 * Feature 5: Dropout Prevention Intervention UI
 * 
 * Real-time dropout risk monitoring with animated interventions.
 * Calls /predict-dropout every 10 seconds during active session.
 * 
 * Intervention Types:
 * - reward: Show star/badge animation
 * - easier_content: Insert confidence booster question
 * - break: Suggest animated break
 * - encouragement: Show cheerful character
 */

import React, { useState, useEffect, useCallback } from 'react';
import API_BASE_URL from '../config';
import './DropoutInterventionSystem.css';

const DropoutInterventionSystem = ({ 
  userId,
  sessionActive,
  currentEngagement,
  sessionStats,
  onInterventionTriggered
}) => {
  const [dropoutRisk, setDropoutRisk] = useState({
    probability: 0,
    riskLevel: 'low',
    interventionNeeded: false,
    interventionType: null
  });
  
  const [showIntervention, setShowIntervention] = useState(false);
  const [interventionMessage, setInterventionMessage] = useState(null);
  const [predictionHistory, setPredictionHistory] = useState([]);

  /**
   * Calculate session features from current stats
   */
  const calculateSessionFeatures = useCallback(() => {
    if (!sessionStats) return null;

    const {
      answers = [],
      engagementHistory = [],
      rewardTimes = [],
      sessionStartTime,
      currentDifficulty = 2,
      totalQuestions = 10
    } = sessionStats;

    // Calculate accuracy decline (slope of last 5 answers)
    const recentAnswers = answers.slice(-5);
    let accuracyDecline = 0;
    if (recentAnswers.length >= 3) {
      const correctCounts = recentAnswers.map((a, i) => 
        recentAnswers.slice(0, i+1).filter(x => x.correct).length / (i+1)
      );
      accuracyDecline = correctCounts.length > 1 
        ? (correctCounts[correctCounts.length-1] - correctCounts[0]) / correctCounts.length
        : 0;
    }

    // Consecutive errors
    let consecutiveErrors = 0;
    for (let i = answers.length - 1; i >= 0; i--) {
      if (!answers[i].correct) consecutiveErrors++;
      else break;
    }

    // Response time increase
    const recentResponseTimes = answers.slice(-5).map(a => a.responseTime);
    const earlyResponseTimes = answers.slice(0, 5).map(a => a.responseTime);
    const avgRecent = recentResponseTimes.length > 0 
      ? recentResponseTimes.reduce((a,b) => a+b, 0) / recentResponseTimes.length
      : 0;
    const avgEarly = earlyResponseTimes.length > 0
      ? earlyResponseTimes.reduce((a,b) => a+b, 0) / earlyResponseTimes.length
      : 0;
    const responseTimeIncrease = avgEarly > 0 ? (avgRecent - avgEarly) / avgEarly : 0;

    // Engagement moving average (last 30 seconds)
    const engagementMA = engagementHistory.length > 0
      ? engagementHistory.slice(-6).reduce((a,b) => a+b, 0) / Math.min(6, engagementHistory.length)
      : currentEngagement || 50;

    // Low engagement duration (seconds below 40%)
    let lowEngagementDuration = 0;
    for (let score of engagementHistory.slice(-12)) {  // Last 60 seconds (5s intervals)
      if (score < 40) lowEngagementDuration += 5;
    }

    // Session duration
    const sessionDuration = sessionStartTime 
      ? (Date.now() - sessionStartTime) / 1000 / 60
      : 0;

    // Time since last reward
    const lastRewardTime = rewardTimes.length > 0 
      ? rewardTimes[rewardTimes.length - 1]
      : sessionStartTime;
    const timeSinceReward = (Date.now() - lastRewardTime) / 1000;

    // Questions remaining
    const questionsRemaining = totalQuestions - answers.length;

    return {
      user_id: userId,
      accuracy_decline_rate: accuracyDecline,
      consecutive_errors: consecutiveErrors,
      avg_response_time_increase: responseTimeIncrease,
      engagement_score_ma: engagementMA,
      low_engagement_duration: lowEngagementDuration,
      session_duration_minutes: sessionDuration,
      time_since_last_reward: timeSinceReward,
      current_difficulty_level: currentDifficulty,
      questions_remaining: questionsRemaining,
      gesture_accuracy_decline: 0,  // Would come from gesture tracker
      audio_replay_frequency: 0,    // Would come from audio player
      pause_count: 0                 // Would come from pause tracker
    };
  }, [userId, sessionStats, currentEngagement]);

  /**
   * Predict dropout risk every 10 seconds
   */
  useEffect(() => {
    if (!sessionActive || !userId) return;

    const predictDropout = async () => {
      const features = calculateSessionFeatures();
      if (!features) return;

      try {
        const response = await fetch(`${API_BASE_URL}/predict-dropout`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(features)
        });

        if (response.ok) {
          const prediction = await response.json();
          
          setDropoutRisk({
            probability: prediction.dropout_probability,
            riskLevel: prediction.risk_level,
            interventionNeeded: prediction.intervention_needed,
            interventionType: prediction.intervention_type
          });

          // Add to history for visualization
          setPredictionHistory(prev => [
            ...prev.slice(-60),  // Keep last 10 minutes
            {
              timestamp: Date.now(),
              probability: prediction.dropout_probability,
              riskLevel: prediction.risk_level
            }
          ]);

          // Trigger intervention if needed
          if (prediction.intervention_needed && prediction.intervention_details) {
            setInterventionMessage(prediction.intervention_details);
            setShowIntervention(true);
            
            // Notify parent component
            if (onInterventionTriggered) {
              onInterventionTriggered(prediction);
            }
          }
        }
      } catch (error) {
        console.error('Dropout prediction failed:', error);
      }
    };

    // Predict every 10 seconds
    const interval = setInterval(predictDropout, 10000);
    predictDropout();  // Initial prediction

    return () => clearInterval(interval);
  }, [sessionActive, userId, calculateSessionFeatures, onInterventionTriggered]);

  /**
   * Close intervention modal after 5 seconds
   */
  useEffect(() => {
    if (showIntervention) {
      const timer = setTimeout(() => {
        setShowIntervention(false);
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [showIntervention]);

  /**
   * Get risk color
   */
  const getRiskColor = (level) => {
    switch (level) {
      case 'critical': return '#dc2626';
      case 'high': return '#ea580c';
      case 'medium': return '#f59e0b';
      default: return '#10b981';
    }
  };

  /**
   * Render intervention modal
   */
  const renderIntervention = () => {
    if (!showIntervention || !interventionMessage) return null;

    return (
      <div className="intervention-overlay">
        <div className={`intervention-modal ${dropoutRisk.interventionType}`}>
          {dropoutRisk.interventionType === 'reward' && (
            <div className="reward-animation">
              <div className="star-burst">⭐</div>
              <div className="star-burst delay-1">✨</div>
              <div className="star-burst delay-2">🌟</div>
            </div>
          )}
          
          {dropoutRisk.interventionType === 'encouragement' && (
            <div className="encouragement-animation">
              <div className="happy-character">😊</div>
            </div>
          )}
          
          {dropoutRisk.interventionType === 'break' && (
            <div className="break-animation">
              <div className="relax-icon">🌈</div>
            </div>
          )}

          <div className="intervention-message">
            <h3>{interventionMessage.message}</h3>
            <p className="intervention-action">{interventionMessage.action}</p>
          </div>

          <button 
            className="intervention-close"
            onClick={() => setShowIntervention(false)}
          >
            ✕
          </button>
        </div>
      </div>
    );
  };

  return (
    <div className="dropout-intervention-system">
      {/* Risk Indicator (always visible in corner) */}
      <div className="risk-indicator" style={{ borderColor: getRiskColor(dropoutRisk.riskLevel) }}>
        <div className="risk-gauge">
          <svg width="60" height="60" viewBox="0 0 60 60">
            <circle
              cx="30"
              cy="30"
              r="25"
              fill="none"
              stroke="#e5e7eb"
              strokeWidth="4"
            />
            <circle
              cx="30"
              cy="30"
              r="25"
              fill="none"
              stroke={getRiskColor(dropoutRisk.riskLevel)}
              strokeWidth="4"
              strokeDasharray={`${dropoutRisk.probability * 157} 157`}
              strokeDashoffset="0"
              transform="rotate(-90 30 30)"
              style={{ transition: 'stroke-dasharray 0.5s ease' }}
            />
          </svg>
          <div className="risk-label">
            {Math.round(dropoutRisk.probability * 100)}%
          </div>
        </div>
        <div className="risk-level-text" style={{ color: getRiskColor(dropoutRisk.riskLevel) }}>
          {dropoutRisk.riskLevel.toUpperCase()}
        </div>
      </div>

      {/* Intervention Modal */}
      {renderIntervention()}

      {/* Risk Trend (Mini chart - only show if high risk) */}
      {dropoutRisk.riskLevel !== 'low' && predictionHistory.length > 3 && (
        <div className="risk-trend">
          <div className="trend-label">Risk Trend (Last 10 min)</div>
          <svg width="120" height="30" viewBox="0 0 120 30">
            <polyline
              points={predictionHistory.map((p, i) => 
                `${i * (120 / predictionHistory.length)},${30 - p.probability * 25}`
              ).join(' ')}
              fill="none"
              stroke={getRiskColor(dropoutRisk.riskLevel)}
              strokeWidth="2"
            />
          </svg>
        </div>
      )}
    </div>
  );
};

export default DropoutInterventionSystem;
