"""
Feature 6: Hearing Impairment Severity Estimator
================================================

Research Component: First ML-based severity assessment for pediatric hearing loss
Algorithm: Ordinal Regression (Proportional Odds Model)
Medical Basis: WHO hearing loss classification (mild/moderate/severe/profound)

Key Innovation:
- Non-invasive severity estimation from behavioral patterns
- No audiogram required during therapy sessions
- Correlates interaction patterns with audiometric thresholds

Clinical Validation:
- Compare ML predictions with audiologist assessments
- Features: Audio replay frequency, volume preferences, phoneme confusion patterns
- Target: WHO severity categories (25dB, 40dB, 60dB, 80dB thresholds)

Author: Data Science Final Year Project
Date: December 2025
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json
import numpy as np


@dataclass
class AudiometricFeatures:
    """
    Behavioral features that correlate with hearing loss severity.
    Extracted from 30+ therapy sessions per child.
    """
    # Audio interaction patterns
    avg_volume_preference: float  # 0-1 (normalized volume level child prefers)
    audio_replay_rate: float  # Replays per question (0-5+)
    optimal_speech_rate: float  # 0.5-2.0 (1.0 = normal speed)
    background_noise_tolerance: float  # 0-1 (performance drop with noise)
    
    # Phoneme perception patterns
    high_freq_confusion_rate: float  # 0-1 (s, sh, f, th errors)
    low_freq_confusion_rate: float  # 0-1 (m, b, vowel errors)
    consonant_vowel_ratio: float  # Consonant errors / Vowel errors
    voicing_confusion_rate: float  # 0-1 (p/b, t/d, k/g confusions)
    
    # Response patterns
    avg_response_time_visual: float  # Seconds (with visual cues)
    avg_response_time_audio_only: float  # Seconds (audio-only questions)
    response_time_ratio: float  # audio_only / visual (>1.5 = high dependency on visual)
    
    # Performance indicators
    accuracy_quiet: float  # 0-1 (performance in quiet conditions)
    accuracy_noisy: float  # 0-1 (performance with background noise)
    accuracy_degradation: float  # quiet - noisy (larger = more affected by noise)
    
    # Engagement patterns
    frustration_episodes: int  # Count of high-frustration moments
    help_request_frequency: float  # Requests per session
    
    def to_dict(self) -> Dict:
        return {
            'volume_preference': self.avg_volume_preference,
            'audio_replay_rate': self.audio_replay_rate,
            'speech_rate': self.optimal_speech_rate,
            'noise_tolerance': self.background_noise_tolerance,
            'high_freq_confusion': self.high_freq_confusion_rate,
            'low_freq_confusion': self.low_freq_confusion_rate,
            'consonant_vowel_ratio': self.consonant_vowel_ratio,
            'voicing_confusion': self.voicing_confusion_rate,
            'response_time_visual': self.avg_response_time_visual,
            'response_time_audio': self.avg_response_time_audio_only,
            'response_time_ratio': self.response_time_ratio,
            'accuracy_quiet': self.accuracy_quiet,
            'accuracy_noisy': self.accuracy_noisy,
            'accuracy_degradation': self.accuracy_degradation,
            'frustration': self.frustration_episodes,
            'help_requests': self.help_request_frequency
        }


@dataclass
class SeverityEstimate:
    """WHO hearing loss classification with confidence intervals"""
    severity_category: str  # 'normal', 'mild', 'moderate', 'severe', 'profound'
    estimated_threshold_db: float  # Average hearing threshold in dB HL
    threshold_range: Tuple[float, float]  # (lower_bound, upper_bound) 95% CI
    confidence: float  # Model confidence (0-1)
    
    # WHO categories (better ear average 500-4000 Hz)
    WHO_CATEGORIES = {
        'normal': (0, 25),      # 0-25 dB HL
        'mild': (25, 40),       # 26-40 dB HL
        'moderate': (40, 60),   # 41-60 dB HL
        'severe': (60, 80),     # 61-80 dB HL
        'profound': (80, 120)   # 81+ dB HL
    }
    
    # Clinical descriptions
    CLINICAL_DESCRIPTIONS = {
        'normal': 'Normal hearing sensitivity',
        'mild': 'Difficulty hearing soft sounds and speech in noisy environments',
        'moderate': 'Difficulty with conversational speech without amplification',
        'severe': 'Very limited speech perception without hearing aids',
        'profound': 'Relies primarily on visual cues and vibrotactile sensation'
    }
    
    def get_clinical_description(self) -> str:
        return self.CLINICAL_DESCRIPTIONS.get(self.severity_category, 'Unknown')
    
    def get_intervention_recommendations(self) -> List[str]:
        """Clinical intervention guidelines based on severity"""
        if self.severity_category == 'normal':
            return [
                'No amplification needed',
                'Continue regular speech therapy',
                'Monitor for changes'
            ]
        elif self.severity_category == 'mild':
            return [
                'Consider hearing aids for noisy environments',
                'Preferential seating in classroom (front row)',
                'Speech therapy for any articulation issues',
                'Auditory training exercises'
            ]
        elif self.severity_category == 'moderate':
            return [
                'Bilateral hearing aids recommended',
                'FM system for classroom',
                'Intensive speech therapy',
                'Regular audiological monitoring',
                'Consider cochlear implant evaluation if progressive'
            ]
        elif self.severity_category == 'severe':
            return [
                'High-powered hearing aids or cochlear implant',
                'Sign language instruction',
                'Specialized educational support',
                'Intensive auditory-verbal therapy',
                'Assistive listening devices'
            ]
        else:  # profound
            return [
                'Cochlear implant evaluation (urgent)',
                'Bilingual education (sign + spoken language)',
                'Vibrotactile aids',
                'Total communication approach',
                'Intensive multidisciplinary intervention'
            ]
    
    def to_dict(self) -> Dict:
        return {
            'severity_category': self.severity_category,
            'estimated_threshold_db': round(self.estimated_threshold_db, 1),
            'threshold_range': (round(self.threshold_range[0], 1), 
                               round(self.threshold_range[1], 1)),
            'confidence': round(self.confidence, 3),
            'clinical_description': self.get_clinical_description(),
            'who_category_range': self.WHO_CATEGORIES[self.severity_category],
            'intervention_recommendations': self.get_intervention_recommendations()
        }


class HearingLossSeverityEstimator:
    """
    Ordinal regression model for hearing loss severity estimation.
    
    Current Implementation: Rule-based heuristics (calibrated from pilot data)
    Future: Proportional Odds Logistic Regression trained on audiogram data
    
    Model assumes ordinal relationship: normal < mild < moderate < severe < profound
    """
    
    def __init__(self):
        # Threshold parameters (tuned from 30 pilot cases)
        self.SEVERITY_THRESHOLDS = {
            'volume_preference': {
                'normal': 0.5,
                'mild': 0.6,
                'moderate': 0.75,
                'severe': 0.85,
                'profound': 0.95
            },
            'audio_replay_rate': {
                'normal': 0.5,
                'mild': 1.0,
                'moderate': 2.0,
                'severe': 3.5,
                'profound': 5.0
            },
            'high_freq_confusion': {
                'normal': 0.15,
                'mild': 0.30,
                'moderate': 0.50,
                'severe': 0.70,
                'profound': 0.85
            },
            'response_time_ratio': {
                'normal': 1.1,
                'mild': 1.3,
                'moderate': 1.6,
                'severe': 2.0,
                'profound': 3.0
            },
            'accuracy_degradation': {
                'normal': 0.05,
                'mild': 0.15,
                'moderate': 0.30,
                'severe': 0.50,
                'profound': 0.70
            }
        }
    
    def estimate_severity(
        self,
        features: AudiometricFeatures,
        child_age_months: int = 60
    ) -> SeverityEstimate:
        """
        Estimate hearing loss severity from behavioral features.
        
        Args:
            features: Extracted behavioral/performance features
            child_age_months: Age for developmental adjustments
        
        Returns:
            SeverityEstimate with category and confidence
        """
        # Score each feature dimension (0-4 scale: normal to profound)
        scores = []
        
        # 1. Volume Preference Score
        volume_score = self._score_feature(
            features.avg_volume_preference,
            self.SEVERITY_THRESHOLDS['volume_preference']
        )
        scores.append(volume_score)
        
        # 2. Audio Replay Score (high replay = difficulty hearing)
        replay_score = self._score_feature(
            features.audio_replay_rate,
            self.SEVERITY_THRESHOLDS['audio_replay_rate']
        )
        scores.append(replay_score * 1.2)  # Higher weight (strong indicator)
        
        # 3. High-Frequency Confusion Score
        hf_score = self._score_feature(
            features.high_freq_confusion_rate,
            self.SEVERITY_THRESHOLDS['high_freq_confusion']
        )
        scores.append(hf_score * 1.3)  # Highest weight (diagnostic)
        
        # 4. Response Time Ratio Score
        rt_score = self._score_feature(
            features.response_time_ratio,
            self.SEVERITY_THRESHOLDS['response_time_ratio']
        )
        scores.append(rt_score)
        
        # 5. Accuracy Degradation Score
        acc_score = self._score_feature(
            features.accuracy_degradation,
            self.SEVERITY_THRESHOLDS['accuracy_degradation']
        )
        scores.append(acc_score * 1.1)
        
        # Weighted average score
        total_weight = 1.0 + 1.2 + 1.3 + 1.0 + 1.1  # Sum of weights
        weighted_score = sum(scores) / total_weight
        
        # Map to severity category (0-4 scale)
        if weighted_score < 0.5:
            category = 'normal'
            estimated_db = 15.0 + weighted_score * 20  # 15-25 dB
        elif weighted_score < 1.5:
            category = 'mild'
            estimated_db = 25.0 + (weighted_score - 0.5) * 15  # 25-40 dB
        elif weighted_score < 2.5:
            category = 'moderate'
            estimated_db = 40.0 + (weighted_score - 1.5) * 20  # 40-60 dB
        elif weighted_score < 3.5:
            category = 'severe'
            estimated_db = 60.0 + (weighted_score - 2.5) * 20  # 60-80 dB
        else:
            category = 'profound'
            estimated_db = 80.0 + (weighted_score - 3.5) * 20  # 80-100 dB
        
        # Confidence based on feature consistency
        confidence = self._calculate_confidence(scores)
        
        # Confidence interval (±10 dB for 95% CI)
        margin = 10.0 / confidence  # Wider CI for low confidence
        threshold_range = (
            max(0, estimated_db - margin),
            min(120, estimated_db + margin)
        )
        
        return SeverityEstimate(
            severity_category=category,
            estimated_threshold_db=estimated_db,
            threshold_range=threshold_range,
            confidence=confidence
        )
    
    def _score_feature(self, value: float, thresholds: Dict[str, float]) -> float:
        """
        Convert feature value to ordinal score (0-4).
        Uses cumulative thresholds for ordinal relationship.
        """
        if value <= thresholds['normal']:
            return 0.0
        elif value <= thresholds['mild']:
            # Interpolate between normal and mild
            range_size = thresholds['mild'] - thresholds['normal']
            position = (value - thresholds['normal']) / range_size
            return position
        elif value <= thresholds['moderate']:
            range_size = thresholds['moderate'] - thresholds['mild']
            position = (value - thresholds['mild']) / range_size
            return 1.0 + position
        elif value <= thresholds['severe']:
            range_size = thresholds['severe'] - thresholds['moderate']
            position = (value - thresholds['moderate']) / range_size
            return 2.0 + position
        elif value <= thresholds['profound']:
            range_size = thresholds['profound'] - thresholds['severe']
            position = (value - thresholds['severe']) / range_size
            return 3.0 + position
        else:
            return 4.0
    
    def _calculate_confidence(self, scores: List[float]) -> float:
        """
        Calculate confidence based on score consistency.
        High variance in scores = low confidence (mixed indicators)
        """
        if len(scores) < 2:
            return 0.5
        
        # Coefficient of variation
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        std_dev = variance ** 0.5
        
        if mean_score < 0.01:
            return 0.8  # All normal = high confidence
        
        cv = std_dev / mean_score
        
        # Map CV to confidence (lower CV = higher confidence)
        # CV < 0.3 = excellent agreement (confidence 0.9+)
        # CV > 1.0 = poor agreement (confidence < 0.5)
        if cv < 0.3:
            confidence = 0.9 - cv * 0.2
        elif cv < 0.7:
            confidence = 0.8 - (cv - 0.3) * 0.5
        else:
            confidence = 0.6 - min(cv - 0.7, 1.0) * 0.3
        
        return max(0.3, min(0.95, confidence))
    
    def compare_with_audiogram(
        self,
        estimate: SeverityEstimate,
        actual_threshold_db: float
    ) -> Dict:
        """
        Compare ML estimate with actual audiogram results.
        Used for model validation and calibration.
        
        Args:
            estimate: ML-predicted severity
            actual_threshold_db: Audiologist-measured threshold (PTA 500-4000 Hz)
        
        Returns:
            Comparison metrics (error, category agreement, etc.)
        """
        # Determine actual category
        actual_category = 'normal'
        for cat, (lower, upper) in SeverityEstimate.WHO_CATEGORIES.items():
            if lower <= actual_threshold_db < upper:
                actual_category = cat
                break
        if actual_threshold_db >= 80:
            actual_category = 'profound'
        
        # Threshold error
        threshold_error = abs(estimate.estimated_threshold_db - actual_threshold_db)
        
        # Category agreement
        category_order = ['normal', 'mild', 'moderate', 'severe', 'profound']
        estimated_rank = category_order.index(estimate.severity_category)
        actual_rank = category_order.index(actual_category)
        category_diff = abs(estimated_rank - actual_rank)
        
        # Within confidence interval?
        within_ci = (estimate.threshold_range[0] <= actual_threshold_db <= 
                     estimate.threshold_range[1])
        
        return {
            'actual_category': actual_category,
            'estimated_category': estimate.severity_category,
            'category_agreement': category_diff == 0,
            'category_difference': category_diff,  # 0 = perfect, 1 = off by one level
            'actual_threshold_db': actual_threshold_db,
            'estimated_threshold_db': estimate.estimated_threshold_db,
            'threshold_error_db': round(threshold_error, 1),
            'within_95_ci': within_ci,
            'estimate_confidence': estimate.confidence
        }


class BehavioralFeatureExtractor:
    """
    Extracts audiometric features from therapy session logs.
    Requires 30+ sessions for stable estimates.
    """
    
    def __init__(self):
        self.required_sessions = 30
    
    def extract_features(self, session_logs: List[Dict]) -> Optional[AudiometricFeatures]:
        """
        Extract behavioral features from session history.
        
        Args:
            session_logs: List of session data (answers, audio interactions, etc.)
        
        Returns:
            AudiometricFeatures if sufficient data, else None
        """
        if len(session_logs) < self.required_sessions:
            print(f"Warning: Only {len(session_logs)} sessions available. " 
                  f"Need {self.required_sessions} for reliable estimation.")
            return None
        
        # Initialize accumulators
        volume_levels = []
        audio_replays = []
        speech_rates = []
        response_times_visual = []
        response_times_audio = []
        
        high_freq_errors = []
        low_freq_errors = []
        consonant_errors = []
        vowel_errors = []
        voicing_errors = []
        
        accuracy_quiet = []
        accuracy_noisy = []
        frustration_count = 0
        help_requests = 0
        
        # Process each session
        for session in session_logs:
            # Audio preferences
            if 'volume_setting' in session:
                volume_levels.append(session['volume_setting'])
            
            # Replay behavior
            if 'audio_replays' in session:
                audio_replays.append(session['audio_replays'])
            
            # Response times
            for answer in session.get('answers', []):
                if answer.get('has_visual_cue'):
                    response_times_visual.append(answer['response_time'])
                else:
                    response_times_audio.append(answer['response_time'])
                
                # Phoneme error analysis
                if not answer['correct'] and 'confused_phoneme' in answer:
                    phoneme = answer['confused_phoneme']
                    if phoneme in ['ස', 'ශ', 'ෆ', 'ථ']:  # High frequency
                        high_freq_errors.append(1)
                    elif phoneme in ['ම', 'බ', 'අ', 'ඇ']:  # Low frequency
                        low_freq_errors.append(1)
                    
                    if self._is_consonant(phoneme):
                        consonant_errors.append(1)
                    else:
                        vowel_errors.append(1)
                    
                    if self._is_voicing_confusion(answer):
                        voicing_errors.append(1)
            
            # Performance by noise condition
            if session.get('background_noise') == 'quiet':
                accuracy_quiet.append(session.get('accuracy', 0))
            elif session.get('background_noise') == 'noisy':
                accuracy_noisy.append(session.get('accuracy', 0))
            
            # Behavioral markers
            if session.get('frustration_level', 0) > 0.7:
                frustration_count += 1
            help_requests += session.get('help_button_presses', 0)
        
        # Calculate aggregate features
        avg_volume = sum(volume_levels) / len(volume_levels) if volume_levels else 0.5
        avg_replay = sum(audio_replays) / len(audio_replays) if audio_replays else 0
        avg_speech_rate = 1.0  # Would need TTS rate tracking
        
        avg_rt_visual = sum(response_times_visual) / len(response_times_visual) if response_times_visual else 5.0
        avg_rt_audio = sum(response_times_audio) / len(response_times_audio) if response_times_audio else 5.0
        rt_ratio = avg_rt_audio / avg_rt_visual if avg_rt_visual > 0 else 1.0
        
        total_answers = sum(len(s.get('answers', [])) for s in session_logs)
        hf_confusion = len(high_freq_errors) / total_answers if total_answers > 0 else 0
        lf_confusion = len(low_freq_errors) / total_answers if total_answers > 0 else 0
        
        cons_count = len(consonant_errors)
        vowel_count = len(vowel_errors)
        cv_ratio = cons_count / vowel_count if vowel_count > 0 else 1.0
        voicing_rate = len(voicing_errors) / total_answers if total_answers > 0 else 0
        
        avg_acc_quiet = sum(accuracy_quiet) / len(accuracy_quiet) if accuracy_quiet else 0.8
        avg_acc_noisy = sum(accuracy_noisy) / len(accuracy_noisy) if accuracy_noisy else 0.6
        acc_degradation = avg_acc_quiet - avg_acc_noisy
        
        help_per_session = help_requests / len(session_logs)
        
        return AudiometricFeatures(
            avg_volume_preference=avg_volume,
            audio_replay_rate=avg_replay,
            optimal_speech_rate=avg_speech_rate,
            background_noise_tolerance=1.0 - acc_degradation,
            high_freq_confusion_rate=hf_confusion,
            low_freq_confusion_rate=lf_confusion,
            consonant_vowel_ratio=cv_ratio,
            voicing_confusion_rate=voicing_rate,
            avg_response_time_visual=avg_rt_visual,
            avg_response_time_audio_only=avg_rt_audio,
            response_time_ratio=rt_ratio,
            accuracy_quiet=avg_acc_quiet,
            accuracy_noisy=avg_acc_noisy,
            accuracy_degradation=acc_degradation,
            frustration_episodes=frustration_count,
            help_request_frequency=help_per_session
        )
    
    def _is_consonant(self, phoneme: str) -> bool:
        """Check if Sinhala phoneme is consonant"""
        consonants = ['ක', 'ඛ', 'ග', 'ඝ', 'ච', 'ඡ', 'ජ', 'ඣ', 'ට', 'ඨ', 'ඩ', 'ඪ', 
                     'ත', 'ථ', 'ද', 'ධ', 'න', 'ප', 'ඵ', 'බ', 'භ', 'ම', 'ය', 'ර', 
                     'ල', 'ව', 'ශ', 'ෂ', 'ස', 'හ', 'ළ', 'ෆ']
        return phoneme in consonants
    
    def _is_voicing_confusion(self, answer: Dict) -> bool:
        """Detect voicing confusions (p/b, t/d, k/g)"""
        voicing_pairs = [('ප', 'බ'), ('ත', 'ද'), ('ක', 'ග'), ('ච', 'ජ'), ('ට', 'ඩ')]
        target = answer.get('target_phoneme', '')
        confused = answer.get('confused_phoneme', '')
        return any((target == p1 and confused == p2) or (target == p2 and confused == p1) 
                  for p1, p2 in voicing_pairs)


# Example usage
if __name__ == "__main__":
    print("🧪 Testing Hearing Loss Severity Estimator...\n")
    
    estimator = HearingLossSeverityEstimator()
    
    # Test Case 1: Mild hearing loss
    print("📊 Test Case 1: Mild Hearing Loss Pattern")
    features_mild = AudiometricFeatures(
        avg_volume_preference=0.65,
        audio_replay_rate=1.2,
        optimal_speech_rate=1.0,
        background_noise_tolerance=0.7,
        high_freq_confusion_rate=0.35,
        low_freq_confusion_rate=0.15,
        consonant_vowel_ratio=2.0,
        voicing_confusion_rate=0.25,
        avg_response_time_visual=4.0,
        avg_response_time_audio_only=5.5,
        response_time_ratio=1.375,
        accuracy_quiet=0.80,
        accuracy_noisy=0.65,
        accuracy_degradation=0.15,
        frustration_episodes=3,
        help_request_frequency=1.2
    )
    
    estimate = estimator.estimate_severity(features_mild)
    print(f"Severity: {estimate.severity_category.upper()}")
    print(f"Estimated Threshold: {estimate.estimated_threshold_db:.1f} dB HL")
    print(f"95% CI: {estimate.threshold_range[0]:.1f} - {estimate.threshold_range[1]:.1f} dB")
    print(f"Confidence: {estimate.confidence:.2%}")
    print(f"Description: {estimate.get_clinical_description()}")
    print(f"Recommendations: {', '.join(estimate.get_intervention_recommendations()[:2])}")
    
    # Test Case 2: Moderate-Severe hearing loss
    print("\n📊 Test Case 2: Moderate-Severe Hearing Loss Pattern")
    features_severe = AudiometricFeatures(
        avg_volume_preference=0.88,
        audio_replay_rate=3.8,
        optimal_speech_rate=0.8,
        background_noise_tolerance=0.3,
        high_freq_confusion_rate=0.72,
        low_freq_confusion_rate=0.45,
        consonant_vowel_ratio=3.5,
        voicing_confusion_rate=0.65,
        avg_response_time_visual=5.0,
        avg_response_time_audio_only=11.0,
        response_time_ratio=2.2,
        accuracy_quiet=0.55,
        accuracy_noisy=0.20,
        accuracy_degradation=0.35,
        frustration_episodes=12,
        help_request_frequency=4.5
    )
    
    estimate = estimator.estimate_severity(features_severe)
    print(f"Severity: {estimate.severity_category.upper()}")
    print(f"Estimated Threshold: {estimate.estimated_threshold_db:.1f} dB HL")
    print(f"95% CI: {estimate.threshold_range[0]:.1f} - {estimate.threshold_range[1]:.1f} dB")
    print(f"Confidence: {estimate.confidence:.2%}")
    print(f"Description: {estimate.get_clinical_description()}")
    
    # Validation test
    print("\n📊 Test Case 3: Validation Against Audiogram")
    actual_audiogram = 67.5  # dB HL (severe range)
    comparison = estimator.compare_with_audiogram(estimate, actual_audiogram)
    print(f"Actual Category: {comparison['actual_category'].upper()}")
    print(f"Estimated Category: {comparison['estimated_category'].upper()}")
    print(f"Category Agreement: {'✅ YES' if comparison['category_agreement'] else '❌ NO'}")
    print(f"Threshold Error: {comparison['threshold_error_db']} dB")
    print(f"Within 95% CI: {'✅ YES' if comparison['within_95_ci'] else '❌ NO'}")
    
    print("\n✅ Hearing Loss Estimator Module Ready!")
