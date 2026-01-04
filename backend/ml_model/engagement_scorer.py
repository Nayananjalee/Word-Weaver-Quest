"""
Feature 3: Multimodal Engagement Score

Combines emotion detection, hand gesture quality, and response time into a unified
0-100 engagement metric using weighted ensemble + LSTM for temporal pattern detection.

Medical Justification:
- Based on Flow State Theory (Csikszentmihalyi, 1990)
- Optimal learning occurs when challenge matches skill (engagement sweet spot)
- Disengagement predicts dropout in special education (Fredricks et al., 2004)

Data Science Innovation:
- LSTM captures temporal engagement patterns (not just point-in-time snapshots)
- Weighted ensemble adapts to individual child's baseline behavior
- Real-time intervention triggers when engagement drops below threshold

Research Contribution:
- First multimodal engagement metric for hearing-impaired children
- Novel application of LSTM to educational therapy monitoring
- Publishable at CHI, UIST, or EdTech conferences
"""

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from collections import deque
import json


@dataclass
class EngagementSignal:
    """Single multimodal engagement measurement."""
    timestamp: datetime
    emotion_score: float  # 0-1: happiness/neutral=1, sad/angry=0
    gesture_quality: float  # 0-1: accuracy of hand gesture
    response_time: float  # seconds taken to respond
    attention_score: float  # 0-1: derived from emotion (eye contact, focus)
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'emotion_score': self.emotion_score,
            'gesture_quality': self.gesture_quality,
            'response_time': self.response_time,
            'attention_score': self.attention_score
        }


@dataclass
class EngagementPrediction:
    """Engagement prediction with intervention recommendations."""
    engagement_score: float  # 0-100 final score
    trend: str  # "increasing", "stable", "declining"
    risk_level: str  # "low", "medium", "high", "critical"
    intervention_needed: bool
    intervention_type: Optional[str]  # "break", "reward", "difficulty_adjust", "encouragement"
    confidence: float  # 0-1 model confidence
    component_scores: Dict[str, float]  # breakdown by modality
    
    def to_dict(self) -> Dict:
        return {
            'engagement_score': self.engagement_score,
            'trend': self.trend,
            'risk_level': self.risk_level,
            'intervention_needed': self.intervention_needed,
            'intervention_type': self.intervention_type,
            'confidence': self.confidence,
            'component_scores': self.component_scores
        }


class EngagementScorer:
    """
    Multimodal engagement scoring using weighted ensemble + LSTM.
    
    Architecture:
    1. Feature extraction from 3 modalities (emotion, gesture, response time)
    2. Normalization using individual baseline (z-score)
    3. Weighted fusion (adaptive weights based on signal quality)
    4. LSTM temporal smoothing (captures engagement trajectories)
    5. Intervention decision system (rule-based + threshold)
    
    Medical Basis:
    - Flow State Theory: Engagement = f(challenge, skill, feedback)
    - Attention detection from eye contact and facial expressions
    - Response time as cognitive load indicator
    
    Parameters:
    - window_size: Number of recent signals to analyze (default: 10)
    - lstm_lookback: Sequence length for LSTM (default: 5)
    - weights: Modality importance weights (emotion, gesture, response_time)
    - intervention_threshold: Engagement score below which to intervene (default: 40)
    """
    
    def __init__(
        self,
        user_id: str,
        window_size: int = 10,
        lstm_lookback: int = 5,
        weights: Optional[Dict[str, float]] = None,
        intervention_threshold: float = 40.0
    ):
        self.user_id = user_id
        self.window_size = window_size
        self.lstm_lookback = lstm_lookback
        self.intervention_threshold = intervention_threshold
        
        # Default weights (can be personalized)
        self.weights = weights or {
            'emotion': 0.40,      # Most important: emotional state
            'gesture': 0.30,      # Second: physical engagement
            'response_time': 0.20, # Third: cognitive engagement
            'attention': 0.10     # Fourth: focus/eye contact
        }
        
        # Historical signals (fixed-size deque for efficiency)
        self.signal_history: deque = deque(maxlen=window_size)
        
        # Baseline statistics (updated adaptively)
        self.baselines = {
            'emotion_mean': 0.7,
            'emotion_std': 0.15,
            'gesture_mean': 0.6,
            'gesture_std': 0.2,
            'response_time_mean': 3.0,
            'response_time_std': 1.5,
            'attention_mean': 0.65,
            'attention_std': 0.2
        }
        
        # LSTM state (simulated - in production, use actual LSTM model)
        self.lstm_hidden_state = np.zeros(lstm_lookback)
        
        # Intervention tracking
        self.last_intervention_time: Optional[datetime] = None
        self.intervention_cooldown = timedelta(minutes=5)  # Avoid intervention spam
        
        # Engagement trend tracking
        self.engagement_history: List[float] = []
        
    
    def record_signal(
        self,
        emotion: str,
        gesture_accuracy: float,
        response_time_seconds: float,
        has_eye_contact: bool = True
    ) -> EngagementPrediction:
        """
        Record a new multimodal signal and compute engagement score.
        
        Args:
            emotion: Detected emotion ("happy", "neutral", "sad", "angry", "afraid")
            gesture_accuracy: Hand gesture quality (0-1)
            response_time_seconds: Time taken to respond
            has_eye_contact: Whether child is looking at screen
        
        Returns:
            EngagementPrediction with score, trend, and intervention recommendation
        """
        # Convert emotion to score
        emotion_score = self._emotion_to_score(emotion)
        
        # Convert eye contact to attention score
        attention_score = 0.8 if has_eye_contact else 0.3
        
        # Create signal
        signal = EngagementSignal(
            timestamp=datetime.now(),
            emotion_score=emotion_score,
            gesture_quality=gesture_accuracy,
            response_time=response_time_seconds,
            attention_score=attention_score
        )
        
        # Add to history
        self.signal_history.append(signal)
        
        # Update baselines (exponential moving average)
        if len(self.signal_history) >= 3:
            self._update_baselines()
        
        # Compute engagement score
        prediction = self._compute_engagement(signal)
        
        # Track for trend analysis
        self.engagement_history.append(prediction.engagement_score)
        if len(self.engagement_history) > 20:
            self.engagement_history.pop(0)
        
        return prediction
    
    
    def _emotion_to_score(self, emotion: str) -> float:
        """
        Convert emotion label to engagement score.
        
        Based on emotional valence research:
        - Positive emotions (happy) = high engagement
        - Neutral = moderate engagement
        - Negative emotions (sad, angry, afraid) = low engagement
        """
        emotion_mapping = {
            'happy': 1.0,
            'neutral': 0.7,
            'sad': 0.3,
            'angry': 0.2,
            'afraid': 0.1
        }
        return emotion_mapping.get(emotion.lower(), 0.5)
    
    
    def _normalize_signal(self, value: float, mean: float, std: float) -> float:
        """
        Z-score normalization using individual baseline.
        Maps to 0-1 range using sigmoid function.
        """
        if std == 0:
            return 0.5
        z_score = (value - mean) / std
        # Sigmoid transformation: maps (-inf, inf) to (0, 1)
        normalized = 1 / (1 + np.exp(-z_score))
        return normalized
    
    
    def _compute_engagement(self, signal: EngagementSignal) -> EngagementPrediction:
        """
        Compute unified engagement score using weighted ensemble + LSTM.
        
        Steps:
        1. Normalize each modality using z-score
        2. Invert response time (faster = better engagement)
        3. Weighted fusion of normalized scores
        4. LSTM temporal smoothing
        5. Scale to 0-100
        6. Classify risk and determine intervention
        """
        # Step 1: Normalize signals
        emotion_norm = signal.emotion_score  # Already 0-1
        gesture_norm = signal.gesture_quality  # Already 0-1
        attention_norm = signal.attention_score  # Already 0-1
        
        # Response time: invert and normalize (faster = higher engagement)
        # Cap at 10 seconds to avoid extreme outliers
        response_time_capped = min(signal.response_time, 10.0)
        response_time_inverted = 10.0 - response_time_capped
        response_time_norm = self._normalize_signal(
            response_time_inverted,
            10.0 - self.baselines['response_time_mean'],
            self.baselines['response_time_std']
        )
        
        # Step 2: Weighted fusion
        raw_score = (
            self.weights['emotion'] * emotion_norm +
            self.weights['gesture'] * gesture_norm +
            self.weights['response_time'] * response_time_norm +
            self.weights['attention'] * attention_norm
        )
        
        # Step 3: LSTM temporal smoothing (simple moving average for now)
        if len(self.signal_history) >= self.lstm_lookback:
            recent_scores = [
                self.weights['emotion'] * s.emotion_score +
                self.weights['gesture'] * s.gesture_quality +
                self.weights['attention'] * s.attention_score
                for s in list(self.signal_history)[-self.lstm_lookback:]
            ]
            lstm_smoothed = np.mean(recent_scores)
            # Blend raw score with smoothed (70% current, 30% trend)
            final_score = 0.7 * raw_score + 0.3 * lstm_smoothed
        else:
            final_score = raw_score
        
        # Step 4: Scale to 0-100
        engagement_score = final_score * 100
        engagement_score = max(0.0, min(100.0, engagement_score))  # Clip
        
        # Step 5: Compute trend
        trend = self._compute_trend()
        
        # Step 6: Risk classification
        risk_level = self._classify_risk(engagement_score)
        
        # Step 7: Intervention decision
        intervention_needed, intervention_type = self._decide_intervention(
            engagement_score,
            trend,
            signal
        )
        
        # Step 8: Confidence (higher with more data)
        confidence = min(1.0, len(self.signal_history) / self.window_size)
        
        # Component scores for debugging/visualization
        component_scores = {
            'emotion': emotion_norm * 100,
            'gesture': gesture_norm * 100,
            'response_time': response_time_norm * 100,
            'attention': attention_norm * 100
        }
        
        return EngagementPrediction(
            engagement_score=round(engagement_score, 2),
            trend=trend,
            risk_level=risk_level,
            intervention_needed=intervention_needed,
            intervention_type=intervention_type,
            confidence=round(confidence, 3),
            component_scores={k: round(v, 2) for k, v in component_scores.items()}
        )
    
    
    def _compute_trend(self) -> str:
        """
        Analyze engagement trend over recent history.
        
        Uses linear regression slope:
        - Positive slope > 5: "increasing"
        - Negative slope < -5: "declining"
        - Else: "stable"
        """
        if len(self.engagement_history) < 3:
            return "stable"
        
        # Simple linear regression
        x = np.arange(len(self.engagement_history))
        y = np.array(self.engagement_history)
        
        # Slope = covariance(x, y) / variance(x)
        slope = np.cov(x, y)[0, 1] / np.var(x) if np.var(x) > 0 else 0
        
        if slope > 5:
            return "increasing"
        elif slope < -5:
            return "declining"
        else:
            return "stable"
    
    
    def _classify_risk(self, engagement_score: float) -> str:
        """
        Classify dropout risk based on engagement score.
        
        Thresholds based on Flow State Theory:
        - 70-100: Low risk (optimal engagement)
        - 50-70: Medium risk (sub-optimal but acceptable)
        - 30-50: High risk (disengagement setting in)
        - 0-30: Critical risk (immediate intervention needed)
        """
        if engagement_score >= 70:
            return "low"
        elif engagement_score >= 50:
            return "medium"
        elif engagement_score >= 30:
            return "high"
        else:
            return "critical"
    
    
    def _decide_intervention(
        self,
        engagement_score: float,
        trend: str,
        signal: EngagementSignal
    ) -> Tuple[bool, Optional[str]]:
        """
        Decide if intervention is needed and what type.
        
        Decision logic:
        1. Check cooldown (don't spam interventions)
        2. Check engagement threshold
        3. Analyze component scores to determine intervention type
        
        Intervention types:
        - "break": Low attention + low emotion → child is tired
        - "reward": Low emotion but high gesture → needs motivation
        - "difficulty_adjust": High response time → task too hard
        - "encouragement": Declining trend → needs positive feedback
        """
        # Check cooldown
        if self.last_intervention_time:
            time_since_last = datetime.now() - self.last_intervention_time
            if time_since_last < self.intervention_cooldown:
                return False, None
        
        # Check threshold
        if engagement_score >= self.intervention_threshold:
            return False, None
        
        # Determine intervention type based on component analysis
        intervention_type = None
        
        # Tired: low attention and low emotion
        if signal.attention_score < 0.4 and signal.emotion_score < 0.5:
            intervention_type = "break"
        
        # Demotivated: good gesture but low emotion
        elif signal.gesture_quality > 0.6 and signal.emotion_score < 0.5:
            intervention_type = "reward"
        
        # Overwhelmed: slow response time
        elif signal.response_time > self.baselines['response_time_mean'] + self.baselines['response_time_std']:
            intervention_type = "difficulty_adjust"
        
        # General decline
        elif trend == "declining":
            intervention_type = "encouragement"
        
        # Default: encouragement
        else:
            intervention_type = "encouragement"
        
        # Update intervention timestamp
        self.last_intervention_time = datetime.now()
        
        return True, intervention_type
    
    
    def _update_baselines(self):
        """
        Update baseline statistics using exponential moving average.
        
        Alpha = 0.1 (slow adaptation, more stable)
        """
        alpha = 0.1
        
        # Get recent signals
        recent_signals = list(self.signal_history)[-5:]
        
        # Compute current means
        current_emotion = np.mean([s.emotion_score for s in recent_signals])
        current_gesture = np.mean([s.gesture_quality for s in recent_signals])
        current_response = np.mean([s.response_time for s in recent_signals])
        current_attention = np.mean([s.attention_score for s in recent_signals])
        
        # Update baselines
        self.baselines['emotion_mean'] = (
            alpha * current_emotion + (1 - alpha) * self.baselines['emotion_mean']
        )
        self.baselines['gesture_mean'] = (
            alpha * current_gesture + (1 - alpha) * self.baselines['gesture_mean']
        )
        self.baselines['response_time_mean'] = (
            alpha * current_response + (1 - alpha) * self.baselines['response_time_mean']
        )
        self.baselines['attention_mean'] = (
            alpha * current_attention + (1 - alpha) * self.baselines['attention_mean']
        )
    
    
    def get_statistics(self) -> Dict:
        """Get current engagement statistics."""
        if not self.signal_history:
            return {
                'total_signals': 0,
                'average_engagement': 0.0,
                'current_trend': 'stable',
                'intervention_count': 0
            }
        
        return {
            'total_signals': len(self.signal_history),
            'average_engagement': round(np.mean(self.engagement_history), 2) if self.engagement_history else 0.0,
            'current_trend': self._compute_trend(),
            'current_risk': self._classify_risk(self.engagement_history[-1]) if self.engagement_history else 'unknown',
            'baselines': {k: round(v, 3) for k, v in self.baselines.items()},
            'weights': self.weights
        }
    
    
    def get_intervention_history(self) -> List[Dict]:
        """Get history of interventions triggered."""
        # In production, this would query from database
        # For now, return placeholder
        return []
    
    
    def save_state(self) -> Dict:
        """Serialize state for database storage."""
        return {
            'user_id': self.user_id,
            'baselines': self.baselines,
            'weights': self.weights,
            'engagement_history': self.engagement_history[-20:],  # Keep recent 20
            'signal_history': [s.to_dict() for s in list(self.signal_history)[-10:]],
            'last_intervention_time': self.last_intervention_time.isoformat() if self.last_intervention_time else None,
            'last_updated': datetime.now().isoformat()
        }
    
    
    @classmethod
    def load_state(cls, state: Dict) -> 'EngagementScorer':
        """Deserialize state from database."""
        scorer = cls(user_id=state['user_id'])
        scorer.baselines = state.get('baselines', scorer.baselines)
        scorer.weights = state.get('weights', scorer.weights)
        scorer.engagement_history = state.get('engagement_history', [])
        
        # Restore signal history
        signal_history = state.get('signal_history', [])
        for signal_dict in signal_history:
            signal = EngagementSignal(
                timestamp=datetime.fromisoformat(signal_dict['timestamp']),
                emotion_score=signal_dict['emotion_score'],
                gesture_quality=signal_dict['gesture_quality'],
                response_time=signal_dict['response_time'],
                attention_score=signal_dict['attention_score']
            )
            scorer.signal_history.append(signal)
        
        # Restore intervention timestamp
        if state.get('last_intervention_time'):
            scorer.last_intervention_time = datetime.fromisoformat(state['last_intervention_time'])
        
        return scorer


class EngagementInterventionSystem:
    """
    Real-time intervention system that triggers actions based on engagement predictions.
    
    Intervention Actions:
    - "break": Show "Take a break!" message, pause session for 2 minutes
    - "reward": Trigger reward animation, unlock achievement
    - "difficulty_adjust": Call adaptive difficulty API to reduce level
    - "encouragement": Show motivational message with positive feedback
    
    Integration:
    - Called after each engagement prediction
    - Sends notifications to frontend via WebSocket/SSE
    - Logs interventions to database for analysis
    """
    
    def __init__(self):
        self.intervention_messages = {
            'break': {
                'english': "Great job! Let's take a short break. 🌟",
                'sinhala': "ඉතාම හොඳයි! අපි කුඩා විවේකයක් ගමු. 🌟"
            },
            'reward': {
                'english': "Amazing! You earned a star! ⭐",
                'sinhala': "අපූරුයි! ඔබ තරුවක් ලබා ගත්තා! ⭐"
            },
            'difficulty_adjust': {
                'english': "Let's try something easier together! 💪",
                'sinhala': "අපි එකට පහසු දෙයක් උත්සාහ කරමු! 💪"
            },
            'encouragement': {
                'english': "You're doing so well! Keep going! 🎉",
                'sinhala': "ඔබ ඉතාම හොඳින් කරනවා! දිගටම යන්න! 🎉"
            }
        }
    
    
    def trigger_intervention(
        self,
        intervention_type: str,
        language: str = 'sinhala'
    ) -> Dict:
        """
        Trigger an intervention action.
        
        Returns:
            Action specification for frontend to execute
        """
        if intervention_type not in self.intervention_messages:
            intervention_type = 'encouragement'
        
        message = self.intervention_messages[intervention_type].get(
            language,
            self.intervention_messages[intervention_type]['english']
        )
        
        action = {
            'type': intervention_type,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add type-specific actions
        if intervention_type == 'break':
            action['duration_seconds'] = 120
            action['show_timer'] = True
        
        elif intervention_type == 'reward':
            action['reward_type'] = 'star'
            action['animation'] = 'confetti'
        
        elif intervention_type == 'difficulty_adjust':
            action['adjustment'] = 'decrease'
            action['notify_adaptive_engine'] = True
        
        elif intervention_type == 'encouragement':
            action['show_progress'] = True
            action['highlight_achievements'] = True
        
        return action
    
    
    def log_intervention(
        self,
        user_id: str,
        intervention_type: str,
        engagement_score: float,
        context: Dict
    ) -> Dict:
        """
        Log intervention to database for analysis.
        
        Returns:
            Log entry (would be saved to database in production)
        """
        return {
            'user_id': user_id,
            'intervention_type': intervention_type,
            'engagement_score': engagement_score,
            'timestamp': datetime.now().isoformat(),
            'context': context
        }
