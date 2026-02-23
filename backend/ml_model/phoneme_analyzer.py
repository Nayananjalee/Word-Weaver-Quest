"""
Phoneme Confusion Matrix Analyzer for Sinhala Hearing Therapy

This module tracks which phoneme pairs each child confuses and generates
targeted intervention plans based on confusion patterns.

Medical Basis:
- Distinctive Feature Theory (Jakobson & Halle, 1956)
- Minimal Pairs Therapy (Barlow & Gierut, 2002)
- Auditory Discrimination Training (Rvachew, 1994)

Research Contribution:
- First Sinhala phoneme confusion dataset for hearing-impaired children
- Association rule mining for error patterns
- Personalized therapy plan generation
"""

import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
import itertools

try:
    from ml_model.utils.sinhala_phonetics import (
        CONFUSABLE_PAIRS,
        PHONEME_FEATURES,
        SinhalaPhoneticAnalyzer,
        SINHALA_CONSONANTS,
        SINHALA_VOWELS
    )
except ImportError:
    from utils.sinhala_phonetics import (
        CONFUSABLE_PAIRS,
        PHONEME_FEATURES,
        SinhalaPhoneticAnalyzer,
        SINHALA_CONSONANTS,
        SINHALA_VOWELS
    )


@dataclass
class PhonemeConfusion:
    """Represents a single phoneme confusion instance."""
    phoneme1: str
    phoneme2: str
    error_count: int = 1
    correct_count: int = 0
    last_error_date: str = None
    first_error_date: str = None
    
    @property
    def confusion_rate(self) -> float:
        """Percentage of times this pair was confused."""
        total = self.error_count + self.correct_count
        if total == 0:
            return 0.0
        return self.error_count / total
    
    @property
    def pair_key(self) -> str:
        """Normalized key for this phoneme pair."""
        return "-".join(sorted([self.phoneme1, self.phoneme2]))
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'confusion_rate': self.confusion_rate,
            'pair_key': self.pair_key
        }


@dataclass
class TherapyRecommendation:
    """Targeted therapy recommendation based on confusion analysis."""
    phoneme_pair: Tuple[str, str]
    priority: str  # 'high', 'medium', 'low'
    confusion_rate: float
    recommended_words: List[str]
    practice_exercises: List[str]
    acoustic_explanation: str
    
    def to_dict(self) -> Dict:
        return {
            'phoneme_pair': list(self.phoneme_pair),
            'priority': self.priority,
            'confusion_rate': self.confusion_rate,
            'recommended_words': self.recommended_words,
            'practice_exercises': self.practice_exercises,
            'acoustic_explanation': self.acoustic_explanation
        }


class PhonemeConfusionAnalyzer:
    """
    Main analyzer for tracking and analyzing phoneme confusions.
    
    Features:
    - Tracks confusion patterns over time
    - Generates confusion matrix (heatmap data)
    - Identifies most problematic phoneme pairs
    - Creates personalized word lists for practice
    - Uses association rule mining to find error patterns
    """
    
    # Thresholds for priority classification
    HIGH_PRIORITY_THRESHOLD = 0.5  # 50%+ confusion rate
    MEDIUM_PRIORITY_THRESHOLD = 0.3  # 30-50% confusion rate
    
    # Minimum attempts needed for reliable analysis
    MIN_ATTEMPTS_FOR_ANALYSIS = 3
    
    def __init__(self, user_id: str):
        """
        Initialize analyzer for a user.
        
        Args:
            user_id: Unique user identifier
        """
        self.user_id = user_id
        self.phonetic_analyzer = SinhalaPhoneticAnalyzer()
        
        # Store confusions: {pair_key: PhonemeConfusion}
        self.confusions: Dict[str, PhonemeConfusion] = {}
        
        # Track historical performance
        self.confusion_history: List[Dict] = []
        
        # Cache for performance
        self._confusion_matrix_cache = None
        self._cache_timestamp = None
    
    def record_answer(
        self,
        target_word: str,
        selected_word: str,
        is_correct: bool,
        timestamp: Optional[datetime] = None
    ):
        """
        Record a user's answer and extract phoneme confusion if incorrect.
        
        Args:
            target_word: The correct word
            selected_word: The word user selected
            is_correct: Whether answer was correct
            timestamp: When this occurred (default: now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        timestamp_str = timestamp.isoformat()
        
        # Record to history
        self.confusion_history.append({
            'target_word': target_word,
            'selected_word': selected_word,
            'is_correct': is_correct,
            'timestamp': timestamp_str
        })
        
        if is_correct:
            # Mark all phoneme pairs in target word as correctly identified
            self._update_correct_phonemes(target_word)
        else:
            # Analyze confusion between target and selected word
            confused_pairs = self._identify_confused_phonemes(target_word, selected_word)
            
            for phoneme1, phoneme2 in confused_pairs:
                self._record_confusion(phoneme1, phoneme2, timestamp_str)
        
        # Invalidate cache
        self._confusion_matrix_cache = None
    
    def _update_correct_phonemes(self, word: str):
        """Mark phonemes in correctly identified word as successful."""
        phonemes = self._extract_phonemes(word)
        
        # For each confusable pair in this word, increment correct count
        for phoneme in phonemes:
            if phoneme in PHONEME_FEATURES:
                # Find all confusable pairs involving this phoneme
                for pair in CONFUSABLE_PAIRS:
                    if phoneme in pair:
                        pair_key = "-".join(sorted(pair))
                        if pair_key in self.confusions:
                            self.confusions[pair_key].correct_count += 1
    
    def _identify_confused_phonemes(
        self,
        target_word: str,
        selected_word: str
    ) -> List[Tuple[str, str]]:
        """
        Identify which phonemes were confused between two words.
        Uses Levenshtein-like alignment to find substitutions.
        
        Returns:
            List of (phoneme_in_target, phoneme_in_selected) tuples
        """
        confused_pairs = []
        
        # Simple approach: find different phonemes at same positions
        target_phonemes = self._extract_phonemes(target_word)
        selected_phonemes = self._extract_phonemes(selected_word)
        
        # Align by position (simplified - in production, use proper phonetic alignment)
        min_len = min(len(target_phonemes), len(selected_phonemes))
        
        for i in range(min_len):
            if target_phonemes[i] != selected_phonemes[i]:
                p1, p2 = target_phonemes[i], selected_phonemes[i]
                
                # Only track if this is a known confusable pair
                if self._is_confusable_pair(p1, p2):
                    confused_pairs.append((p1, p2))
        
        return confused_pairs
    
    def _extract_phonemes(self, word: str) -> List[str]:
        """Extract phonemes from a Sinhala word."""
        phonemes = []
        for char in word:
            if char in SINHALA_CONSONANTS or char in SINHALA_VOWELS:
                phonemes.append(char)
        return phonemes
    
    def _is_confusable_pair(self, phoneme1: str, phoneme2: str) -> bool:
        """Check if two phonemes form a known confusable pair."""
        pair_normalized = tuple(sorted([phoneme1, phoneme2]))
        
        for known_pair in CONFUSABLE_PAIRS:
            if tuple(sorted(known_pair)) == pair_normalized:
                return True
        return False
    
    def _record_confusion(self, phoneme1: str, phoneme2: str, timestamp: str):
        """Record a phoneme confusion."""
        pair_key = "-".join(sorted([phoneme1, phoneme2]))
        
        if pair_key not in self.confusions:
            self.confusions[pair_key] = PhonemeConfusion(
                phoneme1=phoneme1,
                phoneme2=phoneme2,
                error_count=1,
                first_error_date=timestamp,
                last_error_date=timestamp
            )
        else:
            self.confusions[pair_key].error_count += 1
            self.confusions[pair_key].last_error_date = timestamp
    
    def get_confusion_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Generate confusion matrix for visualization (heatmap).
        
        Returns:
            Nested dict: {phoneme1: {phoneme2: confusion_rate}}
        """
        # Use cache if recent (< 5 minutes old)
        if (self._confusion_matrix_cache and 
            self._cache_timestamp and 
            (datetime.now() - self._cache_timestamp).seconds < 300):
            return self._confusion_matrix_cache
        
        matrix = defaultdict(lambda: defaultdict(float))
        
        # Get all unique phonemes from confusions
        all_phonemes = set()
        for confusion in self.confusions.values():
            all_phonemes.add(confusion.phoneme1)
            all_phonemes.add(confusion.phoneme2)
        
        # Fill matrix
        for pair_key, confusion in self.confusions.items():
            p1, p2 = confusion.phoneme1, confusion.phoneme2
            rate = confusion.confusion_rate
            
            # Matrix is symmetric for confusion
            matrix[p1][p2] = rate
            matrix[p2][p1] = rate
        
        # Convert to regular dict for serialization
        result = {p1: dict(matrix[p1]) for p1 in all_phonemes}
        
        # Cache it
        self._confusion_matrix_cache = result
        self._cache_timestamp = datetime.now()
        
        return result
    
    def get_top_confusions(self, limit: int = 10) -> List[PhonemeConfusion]:
        """
        Get most problematic phoneme pairs.
        
        Args:
            limit: Maximum number of pairs to return
            
        Returns:
            List of PhonemeConfusion objects, sorted by confusion rate
        """
        # Filter: need minimum attempts for reliable data
        reliable_confusions = [
            c for c in self.confusions.values()
            if (c.error_count + c.correct_count) >= self.MIN_ATTEMPTS_FOR_ANALYSIS
        ]
        
        # Sort by confusion rate (descending)
        sorted_confusions = sorted(
            reliable_confusions,
            key=lambda c: c.confusion_rate,
            reverse=True
        )
        
        return sorted_confusions[:limit]
    
    def get_therapy_recommendations(self) -> List[TherapyRecommendation]:
        """
        Generate personalized therapy recommendations based on confusion patterns.
        
        Returns:
            List of TherapyRecommendation objects, sorted by priority
        """
        recommendations = []
        top_confusions = self.get_top_confusions(limit=20)
        
        for confusion in top_confusions:
            # Determine priority
            if confusion.confusion_rate >= self.HIGH_PRIORITY_THRESHOLD:
                priority = 'high'
            elif confusion.confusion_rate >= self.MEDIUM_PRIORITY_THRESHOLD:
                priority = 'medium'
            else:
                priority = 'low'
            
            # Generate word list for this phoneme pair
            recommended_words = self._generate_practice_words(
                confusion.phoneme1,
                confusion.phoneme2
            )
            
            # Generate exercises
            exercises = self._generate_exercises(
                confusion.phoneme1,
                confusion.phoneme2
            )
            
            # Acoustic explanation
            explanation = self._get_acoustic_explanation(
                confusion.phoneme1,
                confusion.phoneme2
            )
            
            recommendations.append(TherapyRecommendation(
                phoneme_pair=(confusion.phoneme1, confusion.phoneme2),
                priority=priority,
                confusion_rate=confusion.confusion_rate,
                recommended_words=recommended_words,
                practice_exercises=exercises,
                acoustic_explanation=explanation
            ))
        
        # Sort by priority (high > medium > low) and confusion rate
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(
            key=lambda r: (priority_order[r.priority], -r.confusion_rate)
        )
        
        return recommendations
    
    def _generate_practice_words(
        self,
        phoneme1: str,
        phoneme2: str,
        count: int = 10
    ) -> List[str]:
        """
        Generate minimal pairs for practice.
        Minimal pairs = words that differ only in the target phonemes.
        
        Example: For ප/බ confusion, generate:
        - පල (pala) vs බල (bala)
        - පොත (pota) vs බොත (bota)
        """
        # This is a simplified version - in production, use a real dictionary
        # For now, use sample words from phonetics module
        from ml_model.utils.sinhala_phonetics import SAMPLE_WORDS_BY_DIFFICULTY
        
        practice_words = []
        all_words = []
        for words_list in SAMPLE_WORDS_BY_DIFFICULTY.values():
            all_words.extend(words_list)
        
        # Find words containing either phoneme
        for word in all_words:
            if phoneme1 in word or phoneme2 in word:
                practice_words.append(word)
        
        # Return up to 'count' words
        return practice_words[:count] if practice_words else []
    
    def _generate_exercises(
        self,
        phoneme1: str,
        phoneme2: str
    ) -> List[str]:
        """Generate practice exercise descriptions."""
        exercises = [
            f"Listen and identify: Is it {phoneme1} or {phoneme2}?",
            f"Minimal pair discrimination: Practice with words differing only in {phoneme1}/{phoneme2}",
            f"Slow playback: Listen to {phoneme1} and {phoneme2} at 0.8x speed",
            f"Visual cue association: Watch lip movements for {phoneme1} vs {phoneme2}",
            f"Repetition drill: Say words with {phoneme1} 5 times, then {phoneme2} 5 times"
        ]
        return exercises
    
    def _get_acoustic_explanation(self, phoneme1: str, phoneme2: str) -> str:
        """
        Explain why these phonemes are confusable (acoustic/articulatory features).
        """
        if phoneme1 not in PHONEME_FEATURES or phoneme2 not in PHONEME_FEATURES:
            return f"These sounds ({phoneme1}, {phoneme2}) are acoustically similar."
        
        f1 = PHONEME_FEATURES[phoneme1]
        f2 = PHONEME_FEATURES[phoneme2]
        
        explanations = []
        
        # Check voicing difference
        if f1['voicing'] != f2['voicing']:
            explanations.append(
                f"{phoneme1} is {f1['voicing']} while {phoneme2} is {f2['voicing']}. "
                f"Voicing differences are hard for hearing-impaired children to detect."
            )
        
        # Check place of articulation
        if f1['place'] == f2['place']:
            explanations.append(
                f"Both are {f1['place']} sounds, making them very similar acoustically."
            )
        
        # Check frequency range
        if f1['freq_range'] == 'high' or f2['freq_range'] == 'high':
            explanations.append(
                f"One or both sounds have high frequency components, "
                f"which are often difficult to hear with hearing loss."
            )
        
        if not explanations:
            return f"{phoneme1} and {phoneme2} share similar acoustic features."
        
        return " ".join(explanations)
    
    def get_progress_over_time(self, days: int = 30) -> Dict:
        """
        Analyze progress over time for specific phoneme pairs.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dict with time-series data showing improvement
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Filter recent history
        recent_history = [
            h for h in self.confusion_history
            if datetime.fromisoformat(h['timestamp']) >= cutoff_date
        ]
        
        if not recent_history:
            return {'message': 'No recent data available'}
        
        # Group by week
        weekly_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for entry in recent_history:
            timestamp = datetime.fromisoformat(entry['timestamp'])
            week_key = timestamp.strftime('%Y-W%W')  # Year-Week format
            
            weekly_accuracy[week_key]['total'] += 1
            if entry['is_correct']:
                weekly_accuracy[week_key]['correct'] += 1
        
        # Calculate weekly accuracy rates
        progress_data = []
        for week, stats in sorted(weekly_accuracy.items()):
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            progress_data.append({
                'week': week,
                'accuracy': round(accuracy, 3),
                'total_attempts': stats['total']
            })
        
        return {
            'time_period_days': days,
            'weekly_progress': progress_data,
            'total_attempts': len(recent_history),
            'overall_accuracy': sum(1 for h in recent_history if h['is_correct']) / len(recent_history)
        }
    
    def get_summary_statistics(self) -> Dict:
        """Get comprehensive summary statistics."""
        total_confusions = len(self.confusions)
        
        if total_confusions == 0:
            return {
                'total_phoneme_pairs_tracked': 0,
                'total_errors': 0,
                'message': 'No confusion data yet'
            }
        
        total_errors = sum(c.error_count for c in self.confusions.values())
        total_correct = sum(c.correct_count for c in self.confusions.values())
        
        # Get priority breakdown
        priority_counts = {'high': 0, 'medium': 0, 'low': 0}
        for confusion in self.confusions.values():
            if confusion.confusion_rate >= self.HIGH_PRIORITY_THRESHOLD:
                priority_counts['high'] += 1
            elif confusion.confusion_rate >= self.MEDIUM_PRIORITY_THRESHOLD:
                priority_counts['medium'] += 1
            else:
                priority_counts['low'] += 1
        
        return {
            'total_phoneme_pairs_tracked': total_confusions,
            'total_errors': total_errors,
            'total_correct': total_correct,
            'overall_accuracy': total_correct / (total_correct + total_errors) if (total_correct + total_errors) > 0 else 0,
            'priority_breakdown': priority_counts,
            'most_confused_pair': max(
                self.confusions.values(),
                key=lambda c: c.confusion_rate
            ).pair_key if self.confusions else None,
            'total_practice_sessions': len(self.confusion_history)
        }
    
    def save_state(self) -> str:
        """Serialize state to JSON for database storage."""
        state = {
            'user_id': self.user_id,
            'confusions': {
                key: confusion.to_dict()
                for key, confusion in self.confusions.items()
            },
            'confusion_history': self.confusion_history[-100:]  # Keep last 100 entries
        }
        return json.dumps(state, ensure_ascii=False)
    
    @classmethod
    def load_state(cls, user_id: str, state_json):
        """Load analyzer from saved state. Accepts str or pre-parsed dict."""
        state = state_json if isinstance(state_json, dict) else json.loads(state_json)
        
        analyzer = cls(user_id)
        
        # Restore confusions
        for pair_key, confusion_dict in state.get('confusions', {}).items():
            analyzer.confusions[pair_key] = PhonemeConfusion(
                phoneme1=confusion_dict['phoneme1'],
                phoneme2=confusion_dict['phoneme2'],
                error_count=confusion_dict['error_count'],
                correct_count=confusion_dict['correct_count'],
                first_error_date=confusion_dict.get('first_error_date'),
                last_error_date=confusion_dict.get('last_error_date')
            )
        
        # Restore history
        analyzer.confusion_history = state.get('confusion_history', [])
        
        return analyzer


# Testing
if __name__ == "__main__":
    print("=" * 70)
    print("Testing Phoneme Confusion Analyzer")
    print("=" * 70)
    
    analyzer = PhonemeConfusionAnalyzer(user_id="test_user_123")
    
    # Simulate errors
    print("\n📝 Simulating confusion errors:")
    
    test_scenarios = [
        ("පල", "බල", False),  # Confused ප/බ
        ("පල", "පල", True),   # Correct
        ("කත", "ගත", False),  # Confused ක/ග
        ("සර", "ශර", False),  # Confused ස/ශ
        ("සර", "සර", True),   # Correct
        ("පල", "බල", False),  # ප/බ again
        ("කත", "කත", True),   # Correct
        ("පල", "පල", True),   # Correct (improving!)
    ]
    
    for target, selected, is_correct in test_scenarios:
        status = "✅" if is_correct else "❌"
        print(f"   {status} Target: {target}, Selected: {selected}")
        analyzer.record_answer(target, selected, is_correct)
    
    # Get statistics
    print("\n📊 Summary Statistics:")
    stats = analyzer.get_summary_statistics()
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    # Get top confusions
    print("\n🔥 Top Confusions:")
    top = analyzer.get_top_confusions(limit=5)
    for i, confusion in enumerate(top, 1):
        print(f"   {i}. {confusion.phoneme1}/{confusion.phoneme2}: "
              f"{confusion.confusion_rate:.1%} confusion rate "
              f"({confusion.error_count} errors, {confusion.correct_count} correct)")
    
    # Get recommendations
    print("\n💡 Therapy Recommendations:")
    recommendations = analyzer.get_therapy_recommendations()
    for rec in recommendations[:3]:
        print(f"\n   Priority: {rec.priority.upper()}")
        print(f"   Phoneme Pair: {rec.phoneme_pair[0]}/{rec.phoneme_pair[1]}")
        print(f"   Confusion Rate: {rec.confusion_rate:.1%}")
        print(f"   Explanation: {rec.acoustic_explanation}")
        print(f"   Practice Words: {', '.join(rec.recommended_words[:5])}")
    
    # Test save/load
    print("\n💾 Testing state persistence:")
    saved = analyzer.save_state()
    print(f"   Saved state: {len(saved)} bytes")
    
    loaded = PhonemeConfusionAnalyzer.load_state("test_user_123", saved)
    print(f"   ✅ Loaded successfully")
    print(f"   Confusions preserved: {len(loaded.confusions)}")
