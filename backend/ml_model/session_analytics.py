"""
Session Analytics & Learning Progress Tracker
===============================================

Research-backed comprehensive session analysis module that combines
all ML features into a unified analytics pipeline.

References (Post-2020):
- Dewan et al. (2023): "Engagement Detection in Online Learning: A Systematic Review"
  → Multimodal engagement fusion methodology
- Khosravi et al. (2022): "Explainable AI for Education: A Systematic Review"
  → Interpretable difficulty adaptation using SHAP-like feature importance
- Holstein et al. (2021): "Designing for Human-AI Complementarity in K-12 Education"
  → Human-centered adaptive learning design principles
- Sharma et al. (2020): "Multimodal Data Capabilities for Learning"
  → Temporal pattern analysis in learning sessions
- Molenaar (2022): "The Concept of Hybrid Human-AI Regulation"
  → Adaptive scaffolding intervention timing
- Plass & Pawar (2020): "Toward a Taxonomy of Adaptivity for Learning"
  → Micro/macro adaptivity framework for learning games

Author: Data Science Final Year Project
Date: January 2026
"""

import json
import math
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from collections import deque, defaultdict


@dataclass
class SessionSummary:
    """Complete session analytics summary."""
    session_id: str
    user_id: str
    start_time: float
    end_time: float
    duration_minutes: float
    
    # Performance metrics
    total_questions: int = 0
    correct_answers: int = 0
    accuracy_rate: float = 0.0
    avg_response_time: float = 0.0
    fastest_response: float = 0.0
    slowest_response: float = 0.0
    
    # Engagement metrics
    avg_engagement_score: float = 50.0
    min_engagement: float = 0.0
    max_engagement: float = 100.0
    engagement_stability: float = 0.0  # Standard deviation
    time_in_flow_state: float = 0.0    # % time in 60-80 engagement
    
    # Attention metrics
    avg_focus_quality: float = 0.0
    attention_drift_count: int = 0
    primary_attention_zone: str = "center"
    
    # Difficulty progression
    starting_difficulty: int = 1
    ending_difficulty: int = 1
    difficulty_changes: int = 0
    optimal_difficulty_time: float = 0.0  # % time in optimal zone
    
    # Learning gains
    mastered_words: List[str] = field(default_factory=list)
    struggling_words: List[str] = field(default_factory=list)
    phoneme_confusions: List[str] = field(default_factory=list)
    
    # Behavioral indicators
    frustration_episodes: int = 0
    boredom_episodes: int = 0
    help_seeking_count: int = 0
    audio_replay_count: int = 0
    
    # Dropout risk
    peak_dropout_risk: float = 0.0
    interventions_triggered: int = 0
    intervention_success_rate: float = 0.0
    
    # Achievement data
    stars_earned: int = 0
    streak_best: int = 0
    improvement_score: float = 0.0  # Compared to previous session
    
    def to_dict(self) -> Dict:
        return asdict(self)


class SessionAnalyticsEngine:
    """
    Unified analytics engine that processes real-time session data
    and generates comprehensive learning insights.
    
    Implements the Micro-Macro Adaptivity Framework 
    (Plass & Pawar, 2020):
    - Micro: Real-time within-session adjustments
    - Macro: Cross-session learning trajectory analysis
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.session_start = time.time()
        self.session_id = f"session_{user_id}_{int(self.session_start)}"
        
        # Real-time tracking buffers
        self.answers: List[Dict] = []
        self.engagement_scores: deque = deque(maxlen=500)
        self.response_times: List[float] = []
        self.difficulty_history: List[int] = []
        self.dropout_risks: deque = deque(maxlen=100)
        self.interventions: List[Dict] = []
        self.word_performance: defaultdict = defaultdict(lambda: {'correct': 0, 'incorrect': 0})
        
        # Flow state tracking (Csikszentmihalyi, 1990 - updated Hamari et al., 2023)
        self.flow_state_samples = 0
        self.total_samples = 0
        
        # Frustration/boredom detection
        self.frustration_count = 0
        self.boredom_count = 0
        
        # Streak tracking
        self.current_streak = 0
        self.best_streak = 0
    
    def record_answer(self, word: str, is_correct: bool, response_time: float,
                      difficulty: int, engagement: float = 50.0):
        """Record a single answer for session analytics."""
        self.answers.append({
            'word': word,
            'correct': is_correct,
            'response_time': response_time,
            'difficulty': difficulty,
            'engagement': engagement,
            'timestamp': time.time()
        })
        
        self.response_times.append(response_time)
        self.difficulty_history.append(difficulty)
        self.engagement_scores.append(engagement)
        self.word_performance[word]['correct' if is_correct else 'incorrect'] += 1
        
        # Update streak
        if is_correct:
            self.current_streak += 1
            self.best_streak = max(self.best_streak, self.current_streak)
        else:
            self.current_streak = 0
        
        # Flow state check (engagement between 60-80 = optimal challenge)
        self.total_samples += 1
        if 60 <= engagement <= 80:
            self.flow_state_samples += 1
        
        # Frustration detection (3+ errors, declining engagement)
        if not is_correct and self.current_streak == 0:
            consecutive_errors = 0
            for a in reversed(self.answers):
                if not a['correct']:
                    consecutive_errors += 1
                else:
                    break
            if consecutive_errors >= 3 and engagement < 40:
                self.frustration_count += 1
        
        # Boredom detection (high accuracy but very fast responses, low engagement)
        if is_correct and response_time < 2.0 and engagement < 50:
            self.boredom_count += 1
    
    def record_engagement(self, score: float):
        """Record an engagement score sample."""
        self.engagement_scores.append(score)
        self.total_samples += 1
        if 60 <= score <= 80:
            self.flow_state_samples += 1
    
    def record_dropout_risk(self, risk: float):
        """Record a dropout risk prediction."""
        self.dropout_risks.append({
            'risk': risk,
            'timestamp': time.time()
        })
    
    def record_intervention(self, intervention_type: str, was_effective: bool = None):
        """Record an intervention that was triggered."""
        self.interventions.append({
            'type': intervention_type,
            'effective': was_effective,
            'timestamp': time.time()
        })
    
    def generate_session_summary(self) -> SessionSummary:
        """Generate comprehensive session summary with all analytics."""
        end_time = time.time()
        duration = (end_time - self.session_start) / 60.0
        
        # Performance metrics
        total_q = len(self.answers)
        correct = sum(1 for a in self.answers if a['correct'])
        accuracy = correct / total_q if total_q > 0 else 0
        
        avg_rt = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        fastest = min(self.response_times) if self.response_times else 0
        slowest = max(self.response_times) if self.response_times else 0
        
        # Engagement metrics
        eng_list = list(self.engagement_scores)
        avg_eng = sum(eng_list) / len(eng_list) if eng_list else 50
        min_eng = min(eng_list) if eng_list else 0
        max_eng = max(eng_list) if eng_list else 100
        
        # Standard deviation for stability
        if len(eng_list) > 1:
            mean = avg_eng
            variance = sum((x - mean) ** 2 for x in eng_list) / len(eng_list)
            eng_stability = math.sqrt(variance)
        else:
            eng_stability = 0
        
        flow_pct = (self.flow_state_samples / self.total_samples * 100) if self.total_samples > 0 else 0
        
        # Difficulty progression
        start_diff = self.difficulty_history[0] if self.difficulty_history else 1
        end_diff = self.difficulty_history[-1] if self.difficulty_history else 1
        diff_changes = sum(1 for i in range(1, len(self.difficulty_history)) 
                         if self.difficulty_history[i] != self.difficulty_history[i-1])
        
        # Word analysis
        mastered = [w for w, stats in self.word_performance.items() 
                   if stats['correct'] >= 2 and stats['incorrect'] == 0]
        struggling = [w for w, stats in self.word_performance.items()
                     if stats['incorrect'] >= 2 and stats['correct'] < stats['incorrect']]
        
        # Dropout risk analysis
        peak_risk = max([d['risk'] for d in self.dropout_risks], default=0)
        
        # Intervention analysis
        total_interventions = len(self.interventions)
        effective_interventions = sum(1 for i in self.interventions if i.get('effective', False))
        intervention_success = effective_interventions / total_interventions if total_interventions > 0 else 0
        
        # Calculate improvement score (based on accuracy trend)
        if len(self.answers) >= 4:
            first_half = self.answers[:len(self.answers)//2]
            second_half = self.answers[len(self.answers)//2:]
            first_acc = sum(1 for a in first_half if a['correct']) / len(first_half)
            second_acc = sum(1 for a in second_half if a['correct']) / len(second_half)
            improvement = (second_acc - first_acc) * 100
        else:
            improvement = 0
        
        return SessionSummary(
            session_id=self.session_id,
            user_id=self.user_id,
            start_time=self.session_start,
            end_time=end_time,
            duration_minutes=round(duration, 2),
            total_questions=total_q,
            correct_answers=correct,
            accuracy_rate=round(accuracy, 3),
            avg_response_time=round(avg_rt, 2),
            fastest_response=round(fastest, 2),
            slowest_response=round(slowest, 2),
            avg_engagement_score=round(avg_eng, 1),
            min_engagement=round(min_eng, 1),
            max_engagement=round(max_eng, 1),
            engagement_stability=round(eng_stability, 2),
            time_in_flow_state=round(flow_pct, 1),
            starting_difficulty=start_diff,
            ending_difficulty=end_diff,
            difficulty_changes=diff_changes,
            mastered_words=mastered,
            struggling_words=struggling,
            frustration_episodes=self.frustration_count,
            boredom_episodes=self.boredom_count,
            peak_dropout_risk=round(peak_risk, 3),
            interventions_triggered=total_interventions,
            intervention_success_rate=round(intervention_success, 3),
            stars_earned=correct,
            streak_best=self.best_streak,
            improvement_score=round(improvement, 1)
        )


class LearningTrajectoryAnalyzer:
    """
    Cross-session macro-level learning trajectory analysis.
    
    Implements Item Response Theory (IRT) inspired analysis 
    for tracking learning progress over time.
    
    References:
    - Embretson & Reise (2013): Item Response Theory for Psychologists
    - Plass & Pawar (2020): Micro/Macro Adaptivity Framework
    - Pelánek (2016): Applications of the Elo Rating System in Adaptive Educational Systems
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.session_history: List[Dict] = []
        self.word_mastery: Dict[str, float] = {}  # Elo-like rating per word
        self.skill_levels: Dict[str, float] = {}  # Skill area ratings
        self.learning_rate: float = 0.0  # Estimated learning rate
    
    def add_session(self, summary: SessionSummary):
        """Add a session summary to trajectory analysis."""
        self.session_history.append(summary.to_dict())
        
        # Update word mastery using simplified Elo system
        for word in summary.mastered_words:
            current = self.word_mastery.get(word, 50.0)
            self.word_mastery[word] = min(100, current + 10)
        
        for word in summary.struggling_words:
            current = self.word_mastery.get(word, 50.0)
            self.word_mastery[word] = max(0, current - 5)
        
        # Calculate learning rate (improvement over sessions)
        if len(self.session_history) >= 3:
            recent = self.session_history[-3:]
            accuracies = [s['accuracy_rate'] for s in recent]
            if len(accuracies) >= 2:
                self.learning_rate = (accuracies[-1] - accuracies[0]) / len(accuracies)
    
    def get_learning_trajectory(self) -> Dict:
        """Get comprehensive learning trajectory analysis."""
        if not self.session_history:
            return {"status": "no_data", "sessions": 0}
        
        sessions = self.session_history
        n = len(sessions)
        
        # Overall trends
        accuracies = [s['accuracy_rate'] for s in sessions]
        engagements = [s['avg_engagement_score'] for s in sessions]
        durations = [s['duration_minutes'] for s in sessions]
        
        # Compute trends using simple linear regression
        accuracy_trend = self._compute_trend(accuracies)
        engagement_trend = self._compute_trend(engagements)
        
        # Identify learning plateaus (3+ sessions with < 2% improvement)
        plateaus = 0
        for i in range(2, len(accuracies)):
            if abs(accuracies[i] - accuracies[i-2]) < 0.02:
                plateaus += 1
        
        # Compute Zone of Proximal Development estimate
        # (Optimal challenge level based on recent performance)
        recent_accuracy = sum(accuracies[-5:]) / min(5, len(accuracies))
        if recent_accuracy > 0.85:
            zpd_recommendation = "increase_difficulty"
        elif recent_accuracy < 0.50:
            zpd_recommendation = "decrease_difficulty"
        else:
            zpd_recommendation = "maintain_difficulty"
        
        return {
            "user_id": self.user_id,
            "total_sessions": n,
            "total_questions_attempted": sum(s['total_questions'] for s in sessions),
            "overall_accuracy": round(sum(accuracies) / n, 3),
            "accuracy_trend": accuracy_trend,
            "engagement_trend": engagement_trend,
            "avg_session_duration": round(sum(durations) / n, 1),
            "learning_rate": round(self.learning_rate, 4),
            "learning_plateaus": plateaus,
            "zpd_recommendation": zpd_recommendation,
            "word_mastery": dict(sorted(
                self.word_mastery.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:20]),  # Top 20 words
            "best_session": max(sessions, key=lambda s: s['accuracy_rate']),
            "total_stars": sum(s['stars_earned'] for s in sessions),
            "total_frustration_episodes": sum(s['frustration_episodes'] for s in sessions),
            "recommendations": self._generate_recommendations(sessions)
        }
    
    def _compute_trend(self, values: List[float]) -> str:
        """Simple trend detection using moving averages."""
        if len(values) < 3:
            return "insufficient_data"
        
        recent = sum(values[-3:]) / 3
        earlier = sum(values[:3]) / 3
        
        diff = recent - earlier
        if diff > 0.05:
            return "improving"
        elif diff < -0.05:
            return "declining"
        else:
            return "stable"
    
    def _generate_recommendations(self, sessions: List[Dict]) -> List[Dict]:
        """Generate data-driven recommendations for therapists."""
        recommendations = []
        
        recent = sessions[-5:] if len(sessions) >= 5 else sessions
        
        # Accuracy recommendations
        avg_accuracy = sum(s['accuracy_rate'] for s in recent) / len(recent)
        if avg_accuracy < 0.50:
            recommendations.append({
                "type": "difficulty",
                "priority": "high",
                "message": "ළමයාට අපහසුතා ඇත. දුෂ්කරතා මට්ටම අඩු කරන්න.",
                "message_en": "Child is struggling. Consider reducing difficulty level.",
                "action": "decrease_difficulty"
            })
        elif avg_accuracy > 0.90:
            recommendations.append({
                "type": "difficulty",
                "priority": "medium",
                "message": "ළමයාට ඉතා පහසුයි. අභියෝගාත්මක අන්තර්ගතය ලබා දෙන්න.",
                "message_en": "Content may be too easy. Consider increasing challenge level.",
                "action": "increase_difficulty"
            })
        
        # Engagement recommendations
        avg_engagement = sum(s['avg_engagement_score'] for s in recent) / len(recent)
        if avg_engagement < 40:
            recommendations.append({
                "type": "engagement",
                "priority": "high",
                "message": "සහභාගීත්වය අඩුයි. කෙටි සැසි සමග ත්‍යාග වැඩි කරන්න.",
                "message_en": "Low engagement detected. Try shorter sessions with more rewards.",
                "action": "shorten_sessions"
            })
        
        # Session duration recommendations 
        avg_duration = sum(s['duration_minutes'] for s in recent) / len(recent)
        if avg_duration > 25:
            recommendations.append({
                "type": "duration",
                "priority": "medium",
                "message": "සැසි දිග වැඩියි. 15-20 මිනිත්තු සැසි උත්තම වේ.",
                "message_en": "Sessions may be too long. Optimal duration is 15-20 minutes.",
                "action": "reduce_duration"
            })
        
        # Frustration recommendations
        total_frustration = sum(s['frustration_episodes'] for s in recent)
        if total_frustration > 5:
            recommendations.append({
                "type": "emotional",
                "priority": "high",
                "message": "කලකිරීම් අත්දැකීම් වැඩියි. අනුවර්තී විවේක ලබා දෙන්න.",
                "message_en": "High frustration detected. Increase adaptive breaks and encouragement.",
                "action": "increase_support"
            })
        
        return recommendations
    
    def to_dict(self) -> Dict:
        """Serialize for database storage."""
        return {
            'user_id': self.user_id,
            'session_history': self.session_history,
            'word_mastery': self.word_mastery,
            'skill_levels': self.skill_levels,
            'learning_rate': self.learning_rate
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'LearningTrajectoryAnalyzer':
        """Deserialize from database."""
        analyzer = cls(data['user_id'])
        analyzer.session_history = data.get('session_history', [])
        analyzer.word_mastery = data.get('word_mastery', {})
        analyzer.skill_levels = data.get('skill_levels', {})
        analyzer.learning_rate = data.get('learning_rate', 0.0)
        return analyzer


class PerformanceMetricsCalculator:
    """
    Calculates research-grade performance metrics for the therapy platform.
    
    Implements metrics from:
    - Lim et al. (2023): "Game-Based Learning Analytics" 
    - Baker & Inventado (2014): Educational Data Mining approaches
    - Updated with post-2020 multimodal learning analytics
    """
    
    @staticmethod
    def calculate_learning_efficiency(correct: int, total: int, time_seconds: float) -> float:
        """
        Learning Efficiency Index (LEI)
        Ratio of correct answers to time spent, normalized.
        Higher = more efficient learning.
        """
        if total == 0 or time_seconds == 0:
            return 0.0
        accuracy = correct / total
        time_minutes = time_seconds / 60.0
        return round(accuracy / (time_minutes + 1), 4)  # +1 to avoid division issues
    
    @staticmethod
    def calculate_engagement_consistency(scores: List[float]) -> float:
        """
        Engagement Consistency Score (ECS)
        Measures how stable engagement is throughout session.
        1.0 = perfectly consistent, 0.0 = highly variable.
        """
        if len(scores) < 2:
            return 1.0
        mean = sum(scores) / len(scores)
        if mean == 0:
            return 0.0
        variance = sum((x - mean) ** 2 for x in scores) / len(scores)
        cv = math.sqrt(variance) / mean  # Coefficient of variation
        return round(max(0, 1 - cv), 4)
    
    @staticmethod
    def calculate_flow_ratio(engagement_scores: List[float], 
                            lower: float = 55.0, upper: float = 85.0) -> float:
        """
        Flow State Ratio
        Proportion of time spent in "flow zone" (optimal engagement).
        Based on Csikszentmihalyi's Flow Theory (Hamari et al., 2023 meta-analysis).
        """
        if not engagement_scores:
            return 0.0
        in_flow = sum(1 for s in engagement_scores if lower <= s <= upper)
        return round(in_flow / len(engagement_scores), 4)
    
    @staticmethod
    def calculate_zpd_alignment(accuracy: float, target_accuracy: float = 0.70) -> float:
        """
        Zone of Proximal Development Alignment Score
        How well the difficulty matches the child's ability.
        Perfect alignment = accuracy ≈ 70% (Vygotsky's ZPD).
        
        Returns: 0-1 (1 = perfectly aligned)
        """
        distance = abs(accuracy - target_accuracy)
        return round(max(0, 1 - distance * 2), 4)
    
    @staticmethod
    def calculate_resilience_score(answers: List[Dict]) -> float:
        """
        Resilience Score
        Measures child's ability to recover after errors.
        Based on productive failure research (Kapur, 2020).
        
        Higher = better recovery after mistakes.
        """
        if len(answers) < 3:
            return 0.5
        
        recoveries = 0
        error_sequences = 0
        
        for i in range(1, len(answers)):
            if not answers[i-1].get('correct', True):
                error_sequences += 1
                if answers[i].get('correct', False):
                    recoveries += 1
        
        if error_sequences == 0:
            return 1.0  # No errors = max resilience
        
        return round(recoveries / error_sequences, 4)
    
    @staticmethod
    def calculate_attention_quality_index(
        focus_quality: float, 
        drift_events: int,
        session_duration_minutes: float
    ) -> float:
        """
        Attention Quality Index (AQI)
        Normalized attention quality accounting for session length.
        Longer sessions naturally have more drift events.
        """
        if session_duration_minutes <= 0:
            return focus_quality
        
        drift_rate = drift_events / session_duration_minutes
        # Normalize: < 2 drifts/min = good, > 5 = poor
        drift_penalty = min(1.0, drift_rate / 5.0)
        
        return round(focus_quality * (1 - drift_penalty * 0.3), 4)
