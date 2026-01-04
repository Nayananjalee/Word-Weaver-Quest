import React, { useState, useEffect, useRef } from 'react';
import API_BASE_URL from '../config';
import './EngagementTracker.css';

/**
 * Real-time Engagement Tracker
 * 
 * This component:
 * 1. Tracks multimodal signals (gesture, response time, attention)
 * 2. Sends data to backend /track-engagement endpoint
 * 3. Displays live engagement gauge
 * 4. Shows intervention alerts
 * 
 * Integrates with:
 * - Hand gesture recognition (from HandGestureDetector)
 * - Response time tracking (timer)
 * - Attention detection (eye contact from face detection)
 */
const EngagementTracker = ({ 
  userId, 
  gestureAccuracy = 0.5,
  hasEyeContact = true,
  currentEmotion = 'neutral',
  onInterventionTriggered 
}) => {
  const [engagementScore, setEngagementScore] = useState(50);
  const [trend, setTrend] = useState('stable');
  const [riskLevel, setRiskLevel] = useState('low');
  const [interventionAlert, setInterventionAlert] = useState(null);
  const [responseStartTime, setResponseStartTime] = useState(null);
  const [isTracking, setIsTracking] = useState(false);

  // Track engagement signal
  const trackEngagement = async (gesture, responseTime, eyeContact, emotion = 'neutral') => {
    try {
      const response = await fetch(`${API_BASE_URL}/track-engagement`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          emotion: emotion || 'neutral',  // Default to 'neutral' if not provided
          gesture_accuracy: gesture,
          response_time_seconds: responseTime,
          has_eye_contact: eyeContact
        })
      });

      const result = await response.json();
      
      if (result.success) {
        const prediction = result.engagement_prediction;
        
        // Update state with validation to prevent NaN
        const score = prediction.engagement_score;
        setEngagementScore(isNaN(score) || score === null || score === undefined ? 50 : score);
        setTrend(prediction.trend || 'stable');
        setRiskLevel(prediction.risk_level || 'low');
        
        // Show intervention if needed
        if (result.intervention_action) {
          setInterventionAlert(result.intervention_action);
          
          // Notify parent component
          if (onInterventionTriggered) {
            onInterventionTriggered(result.intervention_action);
          }
          
          // Auto-hide after duration or 5 seconds
          const hideDelay = result.intervention_action.duration_seconds 
            ? result.intervention_action.duration_seconds * 1000 
            : 5000;
          
          setTimeout(() => {
            setInterventionAlert(null);
          }, hideDelay);
        }
      }
    } catch (error) {
      console.error('Error tracking engagement:', error);
    }
  };

  // Start response timer
  const startResponseTimer = () => {
    setResponseStartTime(Date.now());
    setIsTracking(true);
  };

  // End response timer and track engagement
  const endResponseTimer = () => {
    if (responseStartTime) {
      const responseTime = (Date.now() - responseStartTime) / 1000; // Convert to seconds
      
      // Track engagement with current state (including emotion from props)
      trackEngagement(
        gestureAccuracy,
        responseTime,
        hasEyeContact,
        currentEmotion || 'neutral'  // Pass emotion from props
      );
      
      setResponseStartTime(null);
      setIsTracking(false);
    }
  };

  // Get gauge color based on score
  const getGaugeColor = (score) => {
    if (score >= 70) return '#00C851';
    if (score >= 40) return '#FFBB33';
    return '#FF4444';
  };

  const gaugeColor = getGaugeColor(engagementScore);

  return (
    <div className="engagement-tracker">
      {/* Intervention Alert Overlay */}
      {interventionAlert && (
        <div className={`intervention-overlay ${interventionAlert.type}`}>
          <div className="intervention-content">
            <div className="intervention-icon">
              {interventionAlert.type === 'break' && '☕'}
              {interventionAlert.type === 'reward' && '⭐'}
              {interventionAlert.type === 'difficulty_adjust' && '🎯'}
              {interventionAlert.type === 'encouragement' && '💪'}
            </div>
            <h2>{interventionAlert.message}</h2>
            
            {interventionAlert.type === 'break' && interventionAlert.show_timer && (
              <div className="break-timer">
                <p>විවේක කාලය: {interventionAlert.duration_seconds / 60} මිනිත්තු</p>
                <div className="timer-progress">
                  <div className="timer-bar"></div>
                </div>
              </div>
            )}
            
            {interventionAlert.type === 'reward' && interventionAlert.animation === 'confetti' && (
              <div className="confetti-animation">
                {[...Array(20)].map((_, i) => (
                  <div key={i} className="confetti-piece" style={{
                    left: `${Math.random() * 100}%`,
                    animationDelay: `${Math.random() * 0.5}s`,
                    backgroundColor: `hsl(${Math.random() * 360}, 70%, 60%)`
                  }}></div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Mini Engagement Gauge */}
      <div className="mini-gauge">
        <svg viewBox="0 0 100 60" className="mini-gauge-svg">
          {/* Background arc */}
          <path
            d="M 10 50 A 40 40 0 0 1 90 50"
            fill="none"
            stroke="#e0e0e0"
            strokeWidth="8"
            strokeLinecap="round"
          />
          {/* Colored arc */}
          <path
            d="M 10 50 A 40 40 0 0 1 90 50"
            fill="none"
            stroke={gaugeColor}
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={`${(engagementScore / 100) * 125.6} 125.6`}
          />
          {/* Score text */}
          <text x="50" y="45" textAnchor="middle" fontSize="16" fontWeight="bold" fill={gaugeColor}>
            {isNaN(engagementScore) ? 50 : Math.round(engagementScore)}
          </text>
        </svg>
        
        <div className="gauge-label">
          <span className={`trend-arrow ${trend}`}>
            {trend === 'increasing' && '↑'}
            {trend === 'declining' && '↓'}
            {trend === 'stable' && '→'}
          </span>
          <span className={`risk-indicator ${riskLevel}`}>
            {riskLevel === 'low' && '🟢'}
            {riskLevel === 'medium' && '🟡'}
            {riskLevel === 'high' && '🔴'}
            {riskLevel === 'critical' && '⚠️'}
          </span>
        </div>
      </div>

      {/* Debug Info (hide in production) */}
      {process.env.NODE_ENV === 'development' && (
        <div className="tracker-debug">
          <small>
            Gesture: {(gestureAccuracy * 100).toFixed(0)}% | 
            Eye: {hasEyeContact ? '✓' : '✗'}
            {isTracking && ' | ⏱️ Tracking...'}
          </small>
        </div>
      )}

      {/* Expose methods to parent component via ref */}
      <div style={{ display: 'none' }}>
        <button ref={(el) => el && (el.startTimer = startResponseTimer)}>Start</button>
        <button ref={(el) => el && (el.endTimer = endResponseTimer)}>End</button>
      </div>
    </div>
  );
};

// Export methods for parent component to call
export const useEngagementTracker = () => {
  const trackerRef = useRef(null);
  
  return {
    trackerRef,
    startTracking: () => trackerRef.current?.startTimer?.(),
    endTracking: () => trackerRef.current?.endTimer?.()
  };
};

export default EngagementTracker;
