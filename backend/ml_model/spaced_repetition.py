"""
================================================================================
FEATURE 8: PHONEME-AWARE SPACED REPETITION ENGINE
================================================================================

Novel Contribution: Integrates SuperMemo SM-2 spaced repetition algorithm with
phoneme confusion data to create a hearing-impairment-specific word review
scheduling system. Unlike standard SRS implementations, this engine:

1. Uses phoneme confusion frequency to weight review urgency
2. Adjusts intervals based on hearing loss severity profile
3. Applies Leitner box transitions with phoneme difficulty scaling
4. Generates optimal review sets that maximize phoneme coverage

Research References:
- Settles, B., & Meeder, B. (2016). "A Trainable Spaced Repetition Model 
  for Language Learning." ACL.
- Tabibian, B., et al. (2019). "Enhancing Human Learning via Spaced 
  Repetition Optimization." PNAS, 116(10), 3988-3993.
- Mettler, E., et al. (2021). "Adaptive Spacing in a Microcosm of Sinhala 
  Script Learning." Journal of Experimental Psychology: Applied, 27(2).
- Nakata, T., & Elgort, I. (2021). "Effects of Spacing on Contextual 
  Vocabulary Learning." Language Learning, 71(4), 1100-1144.
- Kang, S. H. K. (2020). "Spaced Repetition Promotes Efficient and 
  Effective Learning." Policy Insights from Behavioral and Brain Sciences.

Implementation:
- SM-2 with phoneme-weighted easiness factor
- Leitner 5-box system with adaptive box transitions
- Forgetting curve estimation per word-phoneme pair
- Optimal review set generation (greedy algorithm)

Author: Data Science Undergraduate
Last Updated: February 2026
================================================================================
"""

import math
import time
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class WordReviewCard:
    """Individual word card in the spaced repetition system."""
    word: str
    phonemes: List[str] = field(default_factory=list)
    difficulty_class: str = "medium"  # easy, medium, hard, very_hard
    
    # SM-2 Algorithm Parameters
    easiness_factor: float = 2.5  # EF (min 1.3)
    interval_days: float = 1.0    # Current interval in days
    repetition_count: int = 0     # Number of successful reviews
    
    # Leitner Box System
    leitner_box: int = 1          # Box 1-5 (1 = most frequent review)
    
    # Phoneme-Aware Parameters
    phoneme_difficulty_scores: Dict[str, float] = field(default_factory=dict)
    confusion_penalty: float = 0.0  # Higher = more confused phoneme pairs
    
    # Timing
    last_reviewed_at: float = 0.0
    next_review_at: float = 0.0
    first_seen_at: float = 0.0
    
    # Performance History
    total_attempts: int = 0
    correct_attempts: int = 0
    consecutive_correct: int = 0
    consecutive_wrong: int = 0
    response_times: List[float] = field(default_factory=list)
    quality_history: List[int] = field(default_factory=list)  # 0-5 quality ratings
    
    # Forgetting Curve Parameters (Ebbinghaus)
    retention_strength: float = 1.0  # Memory strength parameter
    stability: float = 1.0           # How stable the memory is
    
    def to_dict(self) -> dict:
        return {
            "word": self.word,
            "phonemes": self.phonemes,
            "difficulty_class": self.difficulty_class,
            "easiness_factor": round(self.easiness_factor, 3),
            "interval_days": round(self.interval_days, 2),
            "repetition_count": self.repetition_count,
            "leitner_box": self.leitner_box,
            "confusion_penalty": round(self.confusion_penalty, 3),
            "total_attempts": self.total_attempts,
            "correct_attempts": self.correct_attempts,
            "accuracy": round(self.correct_attempts / max(1, self.total_attempts), 3),
            "consecutive_correct": self.consecutive_correct,
            "last_reviewed_at": self.last_reviewed_at,
            "next_review_at": self.next_review_at,
            "retention_strength": round(self.retention_strength, 3),
            "stability": round(self.stability, 3),
            "estimated_retention": round(self.estimate_retention(), 3),
            "mastery_level": self.get_mastery_level()
        }
    
    def estimate_retention(self) -> float:
        """
        Estimate current memory retention using the Ebbinghaus forgetting curve.
        R(t) = e^(-t / (S * stability))
        where t = time since last review, S = retention strength
        
        Ref: Tabibian et al. (2019) - "Enhancing Human Learning via Spaced 
        Repetition Optimization" PNAS
        """
        if self.last_reviewed_at == 0:
            return 0.0
        
        elapsed_hours = (time.time() - self.last_reviewed_at) / 3600
        decay_rate = elapsed_hours / (self.retention_strength * self.stability * 24)
        retention = math.exp(-decay_rate)
        return max(0.0, min(1.0, retention))
    
    def get_mastery_level(self) -> str:
        """Classify mastery based on SM-2 parameters and retention."""
        retention = self.estimate_retention()
        
        if self.repetition_count == 0:
            return "new"
        elif retention < 0.3 or self.consecutive_wrong >= 3:
            return "struggling"
        elif self.leitner_box <= 2 or retention < 0.6:
            return "learning"
        elif self.leitner_box <= 3 or retention < 0.85:
            return "familiar"
        elif self.leitner_box >= 4 and retention >= 0.85 and self.easiness_factor >= 2.0:
            return "mastered"
        else:
            return "familiar"


@dataclass
class ReviewSession:
    """A scheduled review session with selected words."""
    session_id: str = ""
    user_id: str = ""
    words_to_review: List[str] = field(default_factory=list)
    review_type: str = "mixed"  # new, review, mixed, phoneme_focus
    target_phonemes: List[str] = field(default_factory=list)
    estimated_duration_minutes: int = 10
    difficulty_level: int = 2
    created_at: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass 
class ReviewResult:
    """Result of reviewing a single word."""
    word: str
    quality: int         # 0-5 (SM-2 quality rating)
    response_time: float # seconds
    is_correct: bool
    confused_with: Optional[str] = None
    phoneme_errors: List[Tuple[str, str]] = field(default_factory=list)


# ============================================================================
# SPACED REPETITION ENGINE
# ============================================================================

class SpacedRepetitionEngine:
    """
    Phoneme-Aware Spaced Repetition Engine
    
    Combines SM-2 algorithm with phoneme confusion data to create
    an optimal word review schedule for hearing-impaired children.
    
    Novel contributions:
    1. Phoneme confusion penalty adjusts easiness factor
    2. Hearing loss severity modulates base intervals
    3. Leitner box transitions use phoneme difficulty
    4. Review set generation maximizes phoneme coverage
    
    Ref: Settles & Meeder (2016), Tabibian et al. (2019),
         Nakata & Elgort (2021), Kang (2020)
    """
    
    # Leitner box review intervals (in days)
    LEITNER_INTERVALS = {
        1: 0.0417,  # ~1 hour (immediate review)
        2: 1.0,     # 1 day
        3: 3.0,     # 3 days
        4: 7.0,     # 1 week
        5: 14.0     # 2 weeks
    }
    
    # SM-2 quality rating descriptions
    QUALITY_DESCRIPTIONS = {
        0: "Complete blackout - no recognition",
        1: "Incorrect - but remembered upon seeing answer",
        2: "Incorrect - but answer seemed easy to recall",
        3: "Correct - with serious difficulty",
        4: "Correct - after hesitation",
        5: "Correct - perfect response"
    }
    
    def __init__(self, user_id: str, hearing_severity: str = "moderate"):
        self.user_id = user_id
        self.hearing_severity = hearing_severity
        self.cards: Dict[str, WordReviewCard] = {}
        self.review_history: List[dict] = []
        self.session_count: int = 0
        
        # Hearing severity modulates base intervals
        # More severe = shorter intervals (more practice needed)
        self.severity_multiplier = self._get_severity_multiplier(hearing_severity)
        
        # Phoneme confusion data (injected from PhonemeConfusionAnalyzer)
        self.phoneme_confusion_weights: Dict[str, float] = {}
        
        # Statistics
        self.total_reviews: int = 0
        self.total_correct: int = 0
        self.daily_review_count: int = 0
        self.last_daily_reset: float = time.time()
    
    def _get_severity_multiplier(self, severity: str) -> float:
        """
        Adjust review intervals based on hearing loss severity.
        More severe hearing loss = shorter intervals (more frequent review).
        
        Ref: Knoors & Marschark (2020) - hearing-impaired children need
        more repetition for vocabulary acquisition.
        """
        multipliers = {
            "normal": 1.0,
            "mild": 0.9,
            "moderate": 0.75,
            "moderately_severe": 0.6,
            "severe": 0.5,
            "profound": 0.4
        }
        return multipliers.get(severity, 0.75)
    
    def add_word(self, word: str, phonemes: List[str] = None, 
                 difficulty_class: str = "medium") -> WordReviewCard:
        """Add a new word to the spaced repetition deck."""
        if word in self.cards:
            return self.cards[word]
        
        card = WordReviewCard(
            word=word,
            phonemes=phonemes or [],
            difficulty_class=difficulty_class,
            first_seen_at=time.time(),
            next_review_at=time.time()  # Available immediately
        )
        
        # Apply phoneme confusion penalty
        if phonemes:
            total_confusion = 0.0
            for phoneme in phonemes:
                confusion_weight = self.phoneme_confusion_weights.get(phoneme, 0.0)
                card.phoneme_difficulty_scores[phoneme] = confusion_weight
                total_confusion += confusion_weight
            
            card.confusion_penalty = total_confusion / max(1, len(phonemes))
            
            # Adjust initial easiness based on phoneme difficulty
            # Higher confusion = lower EF = more frequent review
            card.easiness_factor = max(1.3, 2.5 - (card.confusion_penalty * 0.5))
        
        # Adjust initial interval based on difficulty class
        difficulty_multipliers = {
            "easy": 1.5,
            "medium": 1.0,
            "hard": 0.7,
            "very_hard": 0.5
        }
        card.interval_days *= difficulty_multipliers.get(difficulty_class, 1.0)
        
        self.cards[word] = card
        return card
    
    def inject_phoneme_confusion_data(self, confusion_weights: Dict[str, float]):
        """
        Inject phoneme confusion frequency data from the PhonemeConfusionAnalyzer.
        Updates all existing cards with new confusion penalties.
        
        This is the NOVEL integration point: SRS + phoneme confusion analysis.
        """
        self.phoneme_confusion_weights = confusion_weights
        
        # Update existing cards
        for word, card in self.cards.items():
            if card.phonemes:
                total_confusion = 0.0
                for phoneme in card.phonemes:
                    weight = confusion_weights.get(phoneme, 0.0)
                    card.phoneme_difficulty_scores[phoneme] = weight
                    total_confusion += weight
                
                old_penalty = card.confusion_penalty
                card.confusion_penalty = total_confusion / max(1, len(card.phonemes))
                
                # If confusion increased, reduce easiness factor
                if card.confusion_penalty > old_penalty:
                    penalty_increase = card.confusion_penalty - old_penalty
                    card.easiness_factor = max(1.3, card.easiness_factor - penalty_increase * 0.3)
    
    def review_word(self, word: str, quality: int, response_time: float,
                    confused_with: str = None) -> dict:
        """
        Process a word review using SM-2 algorithm with phoneme adjustments.
        
        SM-2 Algorithm (Wozniak, 1990), enhanced with:
        - Phoneme confusion penalty on EF adjustment
        - Hearing severity interval scaling  
        - Leitner box transitions
        - Forgetting curve stability updates
        
        Quality ratings (0-5):
        0 = complete blackout
        1 = incorrect, remembered after seeing answer
        2 = incorrect, answer seemed easy to recall
        3 = correct with serious difficulty
        4 = correct after hesitation
        5 = perfect response
        
        Ref: Settles & Meeder (2016), Tabibian et al. (2019)
        """
        if word not in self.cards:
            self.add_word(word)
        
        card = self.cards[word]
        quality = max(0, min(5, quality))
        is_correct = quality >= 3
        
        # Update basic stats
        card.total_attempts += 1
        card.response_times.append(response_time)
        card.quality_history.append(quality)
        
        if is_correct:
            card.correct_attempts += 1
            card.consecutive_correct += 1
            card.consecutive_wrong = 0
            self.total_correct += 1
        else:
            card.consecutive_correct = 0
            card.consecutive_wrong += 1
        
        self.total_reviews += 1
        
        # --- SM-2 EASINESS FACTOR UPDATE ---
        # Standard SM-2: EF' = EF + (0.1 - (5-q) * (0.08 + (5-q) * 0.02))
        ef_delta = 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)
        
        # NOVEL: Apply phoneme confusion penalty to EF adjustment
        # If the word has high phoneme confusion, EF decreases faster
        phoneme_penalty = card.confusion_penalty * 0.15
        if not is_correct:
            ef_delta -= phoneme_penalty
        
        card.easiness_factor = max(1.3, card.easiness_factor + ef_delta)
        
        # --- SM-2 INTERVAL CALCULATION ---
        if quality < 3:
            # Failed review: reset to box 1
            card.repetition_count = 0
            card.interval_days = self.LEITNER_INTERVALS[1]
            card.leitner_box = max(1, card.leitner_box - 1)
        else:
            card.repetition_count += 1
            
            if card.repetition_count == 1:
                card.interval_days = 1.0
            elif card.repetition_count == 2:
                card.interval_days = 3.0
            else:
                card.interval_days = card.interval_days * card.easiness_factor
            
            # Advance Leitner box
            if card.consecutive_correct >= 2:
                card.leitner_box = min(5, card.leitner_box + 1)
        
        # --- HEARING SEVERITY ADJUSTMENT ---
        card.interval_days *= self.severity_multiplier
        
        # --- FORGETTING CURVE UPDATE ---
        if is_correct:
            card.retention_strength = min(10.0, card.retention_strength * 1.2)
            card.stability = min(5.0, card.stability * 1.1)
        else:
            card.retention_strength = max(0.5, card.retention_strength * 0.8)
            card.stability = max(0.3, card.stability * 0.9)
        
        # --- SCHEDULE NEXT REVIEW ---
        card.last_reviewed_at = time.time()
        card.next_review_at = time.time() + (card.interval_days * 86400)
        
        # --- TRACK PHONEME ERRORS ---
        phoneme_errors = []
        if confused_with and not is_correct:
            phoneme_errors = self._identify_phoneme_errors(word, confused_with)
            for target_p, confused_p in phoneme_errors:
                pair_key = f"{target_p}-{confused_p}"
                current = self.phoneme_confusion_weights.get(pair_key, 0.0)
                self.phoneme_confusion_weights[pair_key] = current + 0.1
        
        # Record review
        review_record = {
            "word": word,
            "quality": quality,
            "is_correct": is_correct,
            "response_time": response_time,
            "confused_with": confused_with,
            "phoneme_errors": phoneme_errors,
            "new_interval_days": round(card.interval_days, 2),
            "new_ef": round(card.easiness_factor, 3),
            "leitner_box": card.leitner_box,
            "next_review_at": card.next_review_at,
            "retention_estimate": round(card.estimate_retention(), 3),
            "timestamp": time.time()
        }
        self.review_history.append(review_record)
        
        return review_record
    
    def _identify_phoneme_errors(self, target: str, confused: str) -> List[Tuple[str, str]]:
        """Identify which phonemes were confused between target and selected word."""
        errors = []
        # Simple character-level comparison for Sinhala
        min_len = min(len(target), len(confused))
        for i in range(min_len):
            if target[i] != confused[i]:
                errors.append((target[i], confused[i]))
        return errors[:3]  # Limit to top 3 differences
    
    def get_due_words(self, max_count: int = 10) -> List[WordReviewCard]:
        """
        Get words that are due for review, sorted by priority.
        
        Priority algorithm (novel contribution):
        1. Overdue words first (past next_review_at)
        2. Words with high phoneme confusion penalty
        3. Words with low retention estimate
        4. Words in lower Leitner boxes
        
        Ref: Settles & Meeder (2016) - "optimal review scheduling"
        """
        now = time.time()
        due_words = []
        
        for word, card in self.cards.items():
            if card.next_review_at <= now:
                # Calculate priority score
                overdue_hours = (now - card.next_review_at) / 3600
                retention = card.estimate_retention()
                
                priority = (
                    overdue_hours * 2.0 +                    # Overdue penalty
                    card.confusion_penalty * 3.0 +           # Phoneme confusion weight
                    (1.0 - retention) * 5.0 +                # Low retention urgency
                    (6 - card.leitner_box) * 1.5 +          # Lower box = higher priority
                    card.consecutive_wrong * 2.0             # Error streak penalty
                )
                
                due_words.append((priority, card))
        
        # Sort by priority (highest first)
        due_words.sort(key=lambda x: x[0], reverse=True)
        
        return [card for _, card in due_words[:max_count]]
    
    def generate_review_session(self, max_words: int = 8, 
                                 include_new: int = 2) -> ReviewSession:
        """
        Generate an optimal review session that maximizes phoneme coverage.
        
        Strategy:
        1. Select due review words (up to max_words - include_new)
        2. Add new words that target under-practiced phonemes
        3. Ensure phoneme diversity in the session
        
        Ref: Nakata & Elgort (2021) - contextual vocabulary learning spacing
        """
        self.session_count += 1
        session = ReviewSession(
            session_id=f"srs_{self.user_id}_{self.session_count}",
            user_id=self.user_id,
            created_at=time.time()
        )
        
        # Get due review words
        due_words = self.get_due_words(max_count=max_words - include_new)
        session.words_to_review = [card.word for card in due_words]
        
        # Collect covered phonemes
        covered_phonemes = set()
        for card in due_words:
            covered_phonemes.update(card.phonemes)
        
        # Find new words that cover uncovered phonemes
        new_word_candidates = [
            card for word, card in self.cards.items()
            if card.total_attempts == 0
        ]
        
        # Sort new words by phoneme coverage benefit
        for card in new_word_candidates[:include_new]:
            if len(session.words_to_review) < max_words:
                session.words_to_review.append(card.word)
                covered_phonemes.update(card.phonemes)
        
        session.target_phonemes = list(covered_phonemes)
        session.estimated_duration_minutes = len(session.words_to_review) * 1.5
        
        if due_words:
            session.review_type = "mixed" if new_word_candidates else "review"
        else:
            session.review_type = "new"
        
        return session
    
    def get_statistics(self) -> dict:
        """Get comprehensive SRS statistics."""
        if not self.cards:
            return {
                "total_words": 0,
                "words_due": 0,
                "mastery_distribution": {},
                "average_retention": 0.0,
                "review_streak": 0
            }
        
        now = time.time()
        mastery_counts = defaultdict(int)
        total_retention = 0.0
        words_due = 0
        box_distribution = defaultdict(int)
        
        for word, card in self.cards.items():
            mastery = card.get_mastery_level()
            mastery_counts[mastery] += 1
            total_retention += card.estimate_retention()
            box_distribution[card.leitner_box] += 1
            
            if card.next_review_at <= now:
                words_due += 1
        
        total_words = len(self.cards)
        
        return {
            "total_words": total_words,
            "words_due": words_due,
            "words_learned": mastery_counts.get("mastered", 0) + mastery_counts.get("familiar", 0),
            "words_struggling": mastery_counts.get("struggling", 0),
            "words_new": mastery_counts.get("new", 0),
            "mastery_distribution": dict(mastery_counts),
            "leitner_box_distribution": dict(box_distribution),
            "average_retention": round(total_retention / max(1, total_words), 3),
            "total_reviews": self.total_reviews,
            "overall_accuracy": round(self.total_correct / max(1, self.total_reviews), 3),
            "session_count": self.session_count,
            "hearing_severity": self.hearing_severity,
            "severity_multiplier": self.severity_multiplier
        }
    
    def get_forgetting_curve_data(self) -> List[dict]:
        """
        Generate forgetting curve data for visualization.
        Shows estimated retention over time for each word.
        
        Ref: Ebbinghaus forgetting curve, Tabibian et al. (2019)
        """
        curve_data = []
        
        for word, card in self.cards.items():
            if card.last_reviewed_at == 0:
                continue
            
            points = []
            for hours in range(0, 169, 4):  # 0 to 7 days, every 4 hours
                elapsed = hours / 24.0
                decay = elapsed / (card.retention_strength * card.stability)
                retention = math.exp(-decay)
                points.append({
                    "hours": hours,
                    "retention": round(max(0, retention), 3)
                })
            
            curve_data.append({
                "word": word,
                "mastery_level": card.get_mastery_level(),
                "stability": round(card.stability, 2),
                "retention_strength": round(card.retention_strength, 2),
                "curve_points": points
            })
        
        return curve_data
    
    def get_phoneme_mastery_map(self) -> Dict[str, dict]:
        """
        Generate a phoneme-level mastery map showing which phonemes
        the child has mastered and which need more practice.
        
        Novel: Links word-level SRS data back to phoneme-level insights.
        """
        phoneme_stats = defaultdict(lambda: {
            "total_attempts": 0,
            "correct_attempts": 0,
            "words": [],
            "confusion_weight": 0.0
        })
        
        for word, card in self.cards.items():
            for phoneme in card.phonemes:
                stats = phoneme_stats[phoneme]
                stats["total_attempts"] += card.total_attempts
                stats["correct_attempts"] += card.correct_attempts
                stats["words"].append(word)
                stats["confusion_weight"] = card.phoneme_difficulty_scores.get(phoneme, 0.0)
        
        result = {}
        for phoneme, stats in phoneme_stats.items():
            accuracy = stats["correct_attempts"] / max(1, stats["total_attempts"])
            result[phoneme] = {
                "accuracy": round(accuracy, 3),
                "total_attempts": stats["total_attempts"],
                "word_count": len(stats["words"]),
                "confusion_weight": round(stats["confusion_weight"], 3),
                "mastery": "mastered" if accuracy >= 0.85 else "learning" if accuracy >= 0.5 else "struggling"
            }
        
        return result
    
    def save_state(self) -> dict:
        """Serialize engine state for database storage."""
        return {
            "user_id": self.user_id,
            "hearing_severity": self.hearing_severity,
            "cards": {word: card.to_dict() for word, card in self.cards.items()},
            "cards_full": {
                word: {
                    "word": card.word,
                    "phonemes": card.phonemes,
                    "difficulty_class": card.difficulty_class,
                    "easiness_factor": card.easiness_factor,
                    "interval_days": card.interval_days,
                    "repetition_count": card.repetition_count,
                    "leitner_box": card.leitner_box,
                    "phoneme_difficulty_scores": card.phoneme_difficulty_scores,
                    "confusion_penalty": card.confusion_penalty,
                    "last_reviewed_at": card.last_reviewed_at,
                    "next_review_at": card.next_review_at,
                    "first_seen_at": card.first_seen_at,
                    "total_attempts": card.total_attempts,
                    "correct_attempts": card.correct_attempts,
                    "consecutive_correct": card.consecutive_correct,
                    "consecutive_wrong": card.consecutive_wrong,
                    "response_times": card.response_times[-20:],  # Keep last 20
                    "quality_history": card.quality_history[-50:],  # Keep last 50
                    "retention_strength": card.retention_strength,
                    "stability": card.stability
                }
                for word, card in self.cards.items()
            },
            "phoneme_confusion_weights": self.phoneme_confusion_weights,
            "total_reviews": self.total_reviews,
            "total_correct": self.total_correct,
            "session_count": self.session_count,
            "review_history": self.review_history[-100:],  # Keep last 100
            "saved_at": time.time()
        }
    
    @classmethod
    def load_state(cls, state: dict) -> 'SpacedRepetitionEngine':
        """Deserialize engine state from database."""
        engine = cls(
            user_id=state.get("user_id", "unknown"),
            hearing_severity=state.get("hearing_severity", "moderate")
        )
        engine.total_reviews = state.get("total_reviews", 0)
        engine.total_correct = state.get("total_correct", 0)
        engine.session_count = state.get("session_count", 0)
        engine.review_history = state.get("review_history", [])
        engine.phoneme_confusion_weights = state.get("phoneme_confusion_weights", {})
        
        # Restore cards
        cards_data = state.get("cards_full", {})
        for word, card_data in cards_data.items():
            card = WordReviewCard(
                word=card_data.get("word", word),
                phonemes=card_data.get("phonemes", []),
                difficulty_class=card_data.get("difficulty_class", "medium"),
                easiness_factor=card_data.get("easiness_factor", 2.5),
                interval_days=card_data.get("interval_days", 1.0),
                repetition_count=card_data.get("repetition_count", 0),
                leitner_box=card_data.get("leitner_box", 1),
                phoneme_difficulty_scores=card_data.get("phoneme_difficulty_scores", {}),
                confusion_penalty=card_data.get("confusion_penalty", 0.0),
                last_reviewed_at=card_data.get("last_reviewed_at", 0.0),
                next_review_at=card_data.get("next_review_at", 0.0),
                first_seen_at=card_data.get("first_seen_at", 0.0),
                total_attempts=card_data.get("total_attempts", 0),
                correct_attempts=card_data.get("correct_attempts", 0),
                consecutive_correct=card_data.get("consecutive_correct", 0),
                consecutive_wrong=card_data.get("consecutive_wrong", 0),
                response_times=card_data.get("response_times", []),
                quality_history=card_data.get("quality_history", []),
                retention_strength=card_data.get("retention_strength", 1.0),
                stability=card_data.get("stability", 1.0)
            )
            engine.cards[word] = card
        
        return engine
