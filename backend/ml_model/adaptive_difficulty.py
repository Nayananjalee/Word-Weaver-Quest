"""
Adaptive Difficulty Engine for Hearing Therapy
Uses Thompson Sampling (Contextual Multi-Armed Bandit) 

Medical Basis: Zone of Proximal Development (Vygotsky, 1978)
Keeps child in optimal learning zone by adjusting difficulty in real-time

Algorithm: Thompson Sampling with Beta-Bernoulli priors
- Each difficulty level is an "arm" 
- Reward = 1 if child succeeds with good engagement
- Adapts based on: accuracy, response time, emotion, consecutive errors
"""

import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import deque, defaultdict
import random

try:
    from ml_model.utils.sinhala_phonetics import (
        SinhalaPhoneticAnalyzer, 
        classify_word_difficulty,
        SAMPLE_WORDS_BY_DIFFICULTY
    )
except ImportError:
    from utils.sinhala_phonetics import (
        SinhalaPhoneticAnalyzer,
        classify_word_difficulty,
        SAMPLE_WORDS_BY_DIFFICULTY
    )


class DifficultyLevel:
    """Represents one difficulty configuration."""
    
    def __init__(self, level_id: int, config: Dict):
        self.level_id = level_id
        self.word_complexity = config.get('word_complexity', 'medium')
        self.distractor_similarity = config.get('distractor_similarity', 0.5)
        self.audio_speed = config.get('audio_speed', 1.0)
        self.max_syllables = config.get('max_syllables', 4)
        self.sentence_complexity = config.get('sentence_complexity', 'simple')
        
        # Thompson Sampling parameters (Beta distribution)
        self.alpha = 1.0  # Success count + 1 (prior)
        self.beta = 1.0   # Failure count + 1 (prior)
    
    def sample(self) -> float:
        """Sample from Beta distribution for Thompson Sampling."""
        return np.random.beta(self.alpha, self.beta)
    
    def update(self, success: bool):
        """Update parameters based on outcome."""
        if success:
            self.alpha += 1
        else:
            self.beta += 1
    
    def get_expected_success_rate(self) -> float:
        """Expected probability of success."""
        return self.alpha / (self.alpha + self.beta)
    
    def to_dict(self) -> Dict:
        return {
            'level_id': self.level_id,
            'word_complexity': self.word_complexity,
            'distractor_similarity': self.distractor_similarity,
            'audio_speed': self.audio_speed,
            'max_syllables': self.max_syllables,
            'sentence_complexity': self.sentence_complexity,
            'alpha': self.alpha,
            'beta': self.beta,
            'expected_success_rate': self.get_expected_success_rate()
        }


class AdaptiveDifficultyEngine:
    """
    Main Adaptive Difficulty System.
    
    Features:
    - Thompson Sampling for difficulty selection
    - Real-time adaptation based on performance
    - Emotion-aware adjustments
    - Prevents frustration and boredom
    """
    
    # Optimal response time range (seconds)
    OPTIMAL_RESPONSE_TIME = (2.0, 5.0)
    
    # Difficulty level configurations
    DIFFICULTY_CONFIGS = [
        # Level 0: Very Easy (confidence building)
        {
            'word_complexity': 'easy',
            'distractor_similarity': 0.2,  # Very different distractors
            'audio_speed': 0.9,  # Slightly slower
            'max_syllables': 2,
            'sentence_complexity': 'very_simple'
        },
        # Level 1: Easy
        {
            'word_complexity': 'easy',
            'distractor_similarity': 0.4,
            'audio_speed': 1.0,
            'max_syllables': 3,
            'sentence_complexity': 'simple'
        },
        # Level 2: Medium-Easy
        {
            'word_complexity': 'medium',
            'distractor_similarity': 0.5,
            'audio_speed': 1.0,
            'max_syllables': 4,
            'sentence_complexity': 'simple'
        },
        # Level 3: Medium
        {
            'word_complexity': 'medium',
            'distractor_similarity': 0.6,
            'audio_speed': 1.0,
            'max_syllables': 5,
            'sentence_complexity': 'moderate'
        },
        # Level 4: Medium-Hard
        {
            'word_complexity': 'hard',
            'distractor_similarity': 0.7,
            'audio_speed': 1.1,
            'max_syllables': 6,
            'sentence_complexity': 'moderate'
        },
        # Level 5: Hard
        {
            'word_complexity': 'hard',
            'distractor_similarity': 0.8,
            'audio_speed': 1.1,
            'max_syllables': 7,
            'sentence_complexity': 'complex'
        },
        # Level 6: Very Hard (challenge mode)
        {
            'word_complexity': 'very_hard',
            'distractor_similarity': 0.9,
            'audio_speed': 1.2,
            'max_syllables': 8,
            'sentence_complexity': 'complex'
        }
    ]
    
    def __init__(self, user_id: str, initial_level: int = 2):
        """
        Initialize adaptive difficulty system for a user.
        
        Args:
            user_id: Unique user identifier
            initial_level: Starting difficulty (0-6, default=2 medium-easy)
        """
        self.user_id = user_id
        self.phonetic_analyzer = SinhalaPhoneticAnalyzer()
        
        # Initialize all difficulty levels
        self.difficulty_levels = [
            DifficultyLevel(i, config) 
            for i, config in enumerate(self.DIFFICULTY_CONFIGS)
        ]
        
        # Current state
        self.current_level_id = initial_level
        self.consecutive_errors = 0
        self.consecutive_successes = 0
        
        # Performance tracking (last 10 attempts)
        self.recent_performance = deque(maxlen=10)
        self.recent_response_times = deque(maxlen=10)
        self.recent_emotions = deque(maxlen=10)
        
        # Session statistics
        self.session_start_time = datetime.now()
        self.total_questions = 0
        self.total_correct = 0
        
        # Intervention flags
        self.frustration_detected = False
        self.boredom_detected = False
    
    def select_difficulty_level(self, force_level: Optional[int] = None) -> DifficultyLevel:
        """
        Select next difficulty level using Thompson Sampling.
        
        Args:
            force_level: If provided, override Thompson Sampling
            
        Returns:
            Selected DifficultyLevel object
        """
        if force_level is not None:
            self.current_level_id = max(0, min(force_level, len(self.difficulty_levels) - 1))
            return self.difficulty_levels[self.current_level_id]
        
        # Thompson Sampling: sample from each level's Beta distribution
        samples = [level.sample() for level in self.difficulty_levels]
        
        # Select level with highest sample (exploration vs exploitation)
        selected_id = np.argmax(samples)
        
        # Apply constraints to prevent large jumps
        max_jump = 2
        if abs(selected_id - self.current_level_id) > max_jump:
            if selected_id > self.current_level_id:
                selected_id = self.current_level_id + max_jump
            else:
                selected_id = self.current_level_id - max_jump
        
        self.current_level_id = max(0, min(selected_id, len(self.difficulty_levels) - 1))
        return self.difficulty_levels[self.current_level_id]
    
    def update_performance(
        self,
        is_correct: bool,
        response_time: float,
        emotion: Optional[str] = None,
        engagement_score: Optional[float] = None
    ) -> Dict:
        """
        Update system based on user's performance on a question.
        
        Args:
            is_correct: Whether answer was correct
            response_time: Time taken to answer (seconds)
            emotion: Detected emotion ('happy', 'neutral', 'sad', 'frustrated')
            engagement_score: Optional 0-100 engagement score
            
        Returns:
            Dict with recommendations and interventions
        """
        self.total_questions += 1
        if is_correct:
            self.total_correct += 1
        
        # Track recent performance
        self.recent_performance.append(1 if is_correct else 0)
        self.recent_response_times.append(response_time)
        if emotion:
            self.recent_emotions.append(emotion)
        
        # Update consecutive counters
        if is_correct:
            self.consecutive_successes += 1
            self.consecutive_errors = 0
        else:
            self.consecutive_errors += 1
            self.consecutive_successes = 0
        
        # Calculate success based on multiple factors
        success = self._evaluate_success(is_correct, response_time, emotion, engagement_score)
        
        # Update current difficulty level's Thompson Sampling parameters
        current_level = self.difficulty_levels[self.current_level_id]
        current_level.update(success)
        
        # Detect emotional states
        self._detect_emotional_states()
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        return recommendations
    
    def _evaluate_success(
        self,
        is_correct: bool,
        response_time: float,
        emotion: Optional[str],
        engagement_score: Optional[float]
    ) -> bool:
        """
        Comprehensive success evaluation.
        True success = correct answer + good engagement + appropriate timing
        """
        if not is_correct:
            return False
        
        # Check response time (not too fast = guessing, not too slow = struggling)
        time_ok = self.OPTIMAL_RESPONSE_TIME[0] <= response_time <= self.OPTIMAL_RESPONSE_TIME[1] * 2
        
        # Check emotion (negative emotions despite correct answer = difficulty too high)
        emotion_ok = True
        if emotion in ['sad', 'frustrated', 'afraid']:
            emotion_ok = False
        
        # Check engagement
        engagement_ok = True
        if engagement_score is not None and engagement_score < 50:
            engagement_ok = False
        
        # Overall success requires all factors
        return is_correct and time_ok and emotion_ok and engagement_ok
    
    def _detect_emotional_states(self):
        """Detect frustration or boredom from recent patterns."""
        if len(self.recent_emotions) < 3:
            return
        
        recent_emotion_list = list(self.recent_emotions)[-3:]
        
        # Frustration: 3 negative emotions in a row
        negative_emotions = ['sad', 'frustrated', 'afraid']
        if all(e in negative_emotions for e in recent_emotion_list):
            self.frustration_detected = True
        else:
            self.frustration_detected = False
        
        # Boredom: high success rate + very fast responses + neutral emotion
        if len(self.recent_performance) >= 5:
            recent_accuracy = sum(list(self.recent_performance)[-5:]) / 5
            recent_avg_time = np.mean(list(self.recent_response_times)[-5:])
            
            if recent_accuracy >= 0.8 and recent_avg_time < 2.0 and all(e == 'neutral' for e in recent_emotion_list):
                self.boredom_detected = True
            else:
                self.boredom_detected = False
    
    def _generate_recommendations(self) -> Dict:
        """
        Generate recommendations for next action.
        
        Returns:
            {
                'next_difficulty_level': int,
                'intervention': str or None,
                'reason': str,
                'suggested_audio_speed': float,
                'suggested_break': bool
            }
        """
        intervention = None
        reason = ""
        suggested_break = False
        
        # RULE 1: Frustration detected - reduce difficulty
        if self.frustration_detected:
            intervention = "reduce_difficulty"
            next_level = max(0, self.current_level_id - 2)
            reason = "Frustration detected - reducing difficulty for confidence building"
        
        # RULE 2: Too many consecutive errors - reduce difficulty
        elif self.consecutive_errors >= 3:
            intervention = "reduce_difficulty"
            next_level = max(0, self.current_level_id - 1)
            reason = f"3 consecutive errors - reducing difficulty"
        
        # RULE 3: Boredom detected - increase difficulty
        elif self.boredom_detected:
            intervention = "increase_difficulty"
            next_level = min(len(self.difficulty_levels) - 1, self.current_level_id + 1)
            reason = "Boredom detected - increasing challenge"
        
        # RULE 4: High success rate - increase difficulty
        elif self.consecutive_successes >= 5:
            intervention = "increase_difficulty"
            next_level = min(len(self.difficulty_levels) - 1, self.current_level_id + 1)
            reason = "5 consecutive successes - ready for harder challenge"
        
        # RULE 5: Session too long - suggest break
        elif (datetime.now() - self.session_start_time).seconds > 900:  # 15 minutes
            intervention = "suggest_break"
            next_level = self.current_level_id
            suggested_break = True
            reason = "Session duration >15 minutes - break recommended"
        
        # RULE 6: Use Thompson Sampling
        else:
            next_level = self.current_level_id
            reason = "Continuing with adaptive selection"
        
        # Get next level config
        next_level_obj = self.difficulty_levels[next_level]
        
        return {
            'next_difficulty_level': next_level,
            'current_level': self.current_level_id,
            'intervention': intervention,
            'reason': reason,
            'suggested_audio_speed': next_level_obj.audio_speed,
            'suggested_break': suggested_break,
            'session_accuracy': self.total_correct / max(1, self.total_questions),
            'level_config': next_level_obj.to_dict(),
            'consecutive_errors': self.consecutive_errors,
            'consecutive_successes': self.consecutive_successes,
            'frustration_detected': self.frustration_detected,
            'boredom_detected': self.boredom_detected
        }
    
    def get_word_list_for_current_level(self, target_words: List[str] = None) -> List[str]:
        """
        Filter word list based on current difficulty level.
        
        Args:
            target_words: User's difficult words list
            
        Returns:
            Filtered list matching current difficulty
        """
        current_level = self.difficulty_levels[self.current_level_id]
        target_complexity = current_level.word_complexity
        max_syllables = current_level.max_syllables
        
        if target_words:
            # Filter user's words by complexity
            filtered = []
            for word in target_words:
                word_difficulty = classify_word_difficulty(word)
                syllables = self.phonetic_analyzer.count_syllables(word)
                
                if word_difficulty == target_complexity and syllables <= max_syllables:
                    filtered.append(word)
            
            if filtered:
                return filtered
        
        # Fallback to sample words
        if target_complexity in SAMPLE_WORDS_BY_DIFFICULTY:
            return SAMPLE_WORDS_BY_DIFFICULTY[target_complexity]
        
        return SAMPLE_WORDS_BY_DIFFICULTY['medium']
    
    def get_state_summary(self) -> Dict:
        """Get current state for logging/display."""
        return {
            'user_id': self.user_id,
            'current_level': self.current_level_id,
            'session_accuracy': self.total_correct / max(1, self.total_questions),
            'total_questions': self.total_questions,
            'consecutive_errors': self.consecutive_errors,
            'consecutive_successes': self.consecutive_successes,
            'frustration_detected': self.frustration_detected,
            'boredom_detected': self.boredom_detected,
            'all_levels': [level.to_dict() for level in self.difficulty_levels]
        }
    
    def save_state(self) -> str:
        """Serialize state to JSON string for database storage."""
        state = {
            'user_id': self.user_id,
            'current_level_id': self.current_level_id,
            'consecutive_errors': self.consecutive_errors,
            'consecutive_successes': self.consecutive_successes,
            'total_questions': self.total_questions,
            'total_correct': self.total_correct,
            'session_start_time': self.session_start_time.isoformat(),
            'levels': [
                {
                    'level_id': level.level_id,
                    'alpha': level.alpha,
                    'beta': level.beta
                }
                for level in self.difficulty_levels
            ]
        }
        return json.dumps(state)
    
    @classmethod
    def load_state(cls, user_id: str, state_json: str):
        """Load engine from saved state."""
        state = json.loads(state_json)
        
        engine = cls(user_id, initial_level=state['current_level_id'])
        engine.consecutive_errors = state['consecutive_errors']
        engine.consecutive_successes = state['consecutive_successes']
        engine.total_questions = state['total_questions']
        engine.total_correct = state['total_correct']
        engine.session_start_time = datetime.fromisoformat(state['session_start_time'])
        
        # Restore Thompson Sampling parameters
        for level_state in state['levels']:
            level = engine.difficulty_levels[level_state['level_id']]
            level.alpha = level_state['alpha']
            level.beta = level_state['beta']
        
        return engine


# Testing
if __name__ == "__main__":
    print("🧪 Testing Adaptive Difficulty Engine\n")
    
    engine = AdaptiveDifficultyEngine(user_id="test_user_123", initial_level=2)
    
    # Simulate a session
    print("📊 Initial State:")
    print(json.dumps(engine.get_state_summary(), indent=2))
    
    # Simulate 10 questions
    scenarios = [
        (True, 3.5, 'happy', 85),      # Success
        (True, 4.0, 'happy', 90),      # Success
        (False, 8.0, 'sad', 40),       # Failure - struggling
        (False, 7.5, 'frustrated', 35), # Failure - frustrated
        (False, 9.0, 'frustrated', 30), # Failure - very frustrated (should trigger intervention)
        (True, 2.5, 'neutral', 70),    # Success after intervention
        (True, 3.0, 'happy', 80),      # Success
        (True, 2.8, 'happy', 85),      # Success
        (True, 3.2, 'happy', 88),      # Success
        (True, 3.0, 'happy', 90),      # Success (should suggest level up)
    ]
    
    print("\n🎮 Simulating Session:")
    for i, (correct, time, emotion, engagement) in enumerate(scenarios, 1):
        print(f"\n--- Question {i} ---")
        print(f"   Result: {'✅' if correct else '❌'} | Time: {time}s | Emotion: {emotion} | Engagement: {engagement}")
        
        recommendations = engine.update_performance(correct, time, emotion, engagement)
        
        print(f"   📈 Recommendations:")
        print(f"      Current Level: {recommendations['current_level']} → Next Level: {recommendations['next_difficulty_level']}")
        print(f"      Intervention: {recommendations['intervention']}")
        print(f"      Reason: {recommendations['reason']}")
        
        if recommendations['suggested_break']:
            print(f"      ⏸️  BREAK SUGGESTED")
        
        # Apply the recommended level
        engine.current_level_id = recommendations['next_difficulty_level']
    
    print("\n" + "="*60)
    print("📊 Final Session Summary:")
    summary = engine.get_state_summary()
    print(json.dumps(summary, indent=2))
    
    # Test save/load
    print("\n💾 Testing State Persistence:")
    saved_state = engine.save_state()
    print(f"Saved state length: {len(saved_state)} characters")
    
    loaded_engine = AdaptiveDifficultyEngine.load_state("test_user_123", saved_state)
    print(f"✅ State loaded successfully")
    print(f"   Accuracy after load: {loaded_engine.total_correct}/{loaded_engine.total_questions}")
