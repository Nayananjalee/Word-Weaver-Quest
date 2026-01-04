"""
Sinhala Phonetics and Linguistic Analysis Utilities
For Hearing Therapy Application

References:
- Sinhala Phonology (Gair & Paolillo, 1997)
- WHO Hearing Loss Classification
"""

import re
from typing import List, Dict, Tuple
from collections import defaultdict

# Sinhala Unicode Ranges
SINHALA_VOWELS = [
    'අ', 'ආ', 'ඇ', 'ඈ', 'ඉ', 'ඊ', 'උ', 'ඌ', 'ඍ', 'ඎ', 
    'ඏ', 'ඐ', 'එ', 'ඒ', 'ඓ', 'ඔ', 'ඕ', 'ඖ'
]

SINHALA_CONSONANTS = [
    'ක', 'ඛ', 'ග', 'ඝ', 'ඞ', 'ඟ',  # Velars
    'ච', 'ඡ', 'ජ', 'ඣ', 'ඤ', 'ඥ',  # Palatals
    'ට', 'ඨ', 'ඩ', 'ඪ', 'ණ', 'ඬ',  # Retroflexes
    'ත', 'ථ', 'ද', 'ධ', 'න', 'ඳ',  # Dentals
    'ප', 'ඵ', 'බ', 'භ', 'ම', 'ඹ',  # Labials
    'ය', 'ර', 'ල', 'ව', 'ශ', 'ෂ', 'ස', 'හ', 'ළ', 'ෆ'  # Sonorants/Fricatives
]

# Phonetically Confusable Pairs (for Hearing-Impaired Children)
# Based on acoustic similarity and clinical observations
CONFUSABLE_PAIRS = [
    # Voicing Contrasts (most confusing for hearing-impaired)
    ("ප", "බ"), ("ක", "ග"), ("ත", "ද"), ("ච", "ජ"), ("ට", "ඩ"),
    
    # Place of Articulation
    ("ස", "ශ"), ("ස", "ෂ"), ("ශ", "ෂ"),  # Sibilants
    ("න", "ණ"), ("ල", "ළ"),  # Retroflexes
    
    # Nasals
    ("ම", "න"), ("ඤ", "ඥ"),
    
    # Aspirated vs Unaspirated
    ("ක", "ඛ"), ("ප", "ඵ"), ("ත", "ථ"), ("ච", "ඡ"),
    
    # Vowels (length contrasts)
    ("අ", "ආ"), ("ඉ", "ඊ"), ("උ", "ඌ"), ("එ", "ඒ"), ("ඔ", "ඕ")
]

# Phoneme Feature Classification
PHONEME_FEATURES = {
    # Voiceless stops
    "ක": {"type": "stop", "voicing": "voiceless", "place": "velar", "freq_range": "mid"},
    "ට": {"type": "stop", "voicing": "voiceless", "place": "retroflex", "freq_range": "mid"},
    "ත": {"type": "stop", "voicing": "voiceless", "place": "dental", "freq_range": "mid"},
    "ප": {"type": "stop", "voicing": "voiceless", "place": "labial", "freq_range": "low"},
    "ච": {"type": "affricate", "voicing": "voiceless", "place": "palatal", "freq_range": "high"},
    
    # Voiced stops
    "ග": {"type": "stop", "voicing": "voiced", "place": "velar", "freq_range": "mid"},
    "ඩ": {"type": "stop", "voicing": "voiced", "place": "retroflex", "freq_range": "mid"},
    "ද": {"type": "stop", "voicing": "voiced", "place": "dental", "freq_range": "mid"},
    "බ": {"type": "stop", "voicing": "voiced", "place": "labial", "freq_range": "low"},
    "ජ": {"type": "affricate", "voicing": "voiced", "place": "palatal", "freq_range": "high"},
    
    # Fricatives (high frequency - difficult for hearing-impaired)
    "ස": {"type": "fricative", "voicing": "voiceless", "place": "alveolar", "freq_range": "high"},
    "ශ": {"type": "fricative", "voicing": "voiceless", "place": "palatal", "freq_range": "high"},
    "ෂ": {"type": "fricative", "voicing": "voiceless", "place": "retroflex", "freq_range": "high"},
    "ෆ": {"type": "fricative", "voicing": "voiceless", "place": "labial", "freq_range": "high"},
    
    # Nasals (low frequency - easier to hear)
    "ම": {"type": "nasal", "voicing": "voiced", "place": "labial", "freq_range": "low"},
    "න": {"type": "nasal", "voicing": "voiced", "place": "dental", "freq_range": "low"},
    "ණ": {"type": "nasal", "voicing": "voiced", "place": "retroflex", "freq_range": "low"},
    "ඤ": {"type": "nasal", "voicing": "voiced", "place": "palatal", "freq_range": "low"},
    
    # Liquids/Approximants
    "ර": {"type": "trill", "voicing": "voiced", "place": "alveolar", "freq_range": "mid"},
    "ල": {"type": "lateral", "voicing": "voiced", "place": "alveolar", "freq_range": "mid"},
    "ළ": {"type": "lateral", "voicing": "voiced", "place": "retroflex", "freq_range": "mid"},
    "ව": {"type": "approximant", "voicing": "voiced", "place": "labial", "freq_range": "low"},
    "ය": {"type": "approximant", "voicing": "voiced", "place": "palatal", "freq_range": "mid"},
}

class SinhalaPhoneticAnalyzer:
    """Analyzes phonetic complexity and acoustic features of Sinhala words."""
    
    def __init__(self):
        self.confusable_map = defaultdict(list)
        for pair in CONFUSABLE_PAIRS:
            self.confusable_map[pair[0]].append(pair[1])
            self.confusable_map[pair[1]].append(pair[0])
    
    def count_syllables(self, word: str) -> int:
        """
        Estimates syllable count (simplified algorithm).
        In Sinhala, typically one syllable per vowel sound.
        """
        vowel_count = sum(1 for char in word if char in SINHALA_VOWELS)
        # Account for vowel diacritics (්)
        vowel_signs = ['ා', 'ි', 'ී', 'ු', 'ූ', 'ෘ', 'ෙ', 'ේ', 'ෛ', 'ො', 'ෝ', 'ෞ', 'ෟ']
        vowel_count += sum(1 for char in word if char in vowel_signs)
        return max(1, vowel_count)
    
    def get_phonetic_complexity(self, word: str) -> Dict[str, float]:
        """
        Calculates multiple complexity metrics for a word.
        Higher scores = harder for hearing-impaired children.
        
        Returns:
            {
                'overall_complexity': 0.0-1.0,
                'consonant_cluster_count': int,
                'high_freq_phoneme_count': int,
                'confusable_phoneme_count': int,
                'syllable_count': int
            }
        """
        syllables = self.count_syllables(word)
        
        # Count consonant clusters (consecutive consonants)
        consonant_clusters = 0
        prev_consonant = False
        for char in word:
            if char in SINHALA_CONSONANTS:
                if prev_consonant:
                    consonant_clusters += 1
                prev_consonant = True
            else:
                prev_consonant = False
        
        # Count high-frequency phonemes (fricatives - hard to hear)
        high_freq_count = sum(
            1 for char in word 
            if char in PHONEME_FEATURES and 
            PHONEME_FEATURES[char].get('freq_range') == 'high'
        )
        
        # Count phonemes that have confusable pairs
        confusable_count = sum(1 for char in word if char in self.confusable_map)
        
        # Overall complexity score (0.0 to 1.0)
        # Weighted combination of factors
        complexity = (
            (syllables / 8.0) * 0.3 +  # Normalize by max expected syllables
            (consonant_clusters / 3.0) * 0.3 +
            (high_freq_count / len(word)) * 0.2 +
            (confusable_count / len(word)) * 0.2
        )
        complexity = min(1.0, complexity)  # Cap at 1.0
        
        return {
            'overall_complexity': round(complexity, 3),
            'consonant_cluster_count': consonant_clusters,
            'high_freq_phoneme_count': high_freq_count,
            'confusable_phoneme_count': confusable_count,
            'syllable_count': syllables
        }
    
    def get_confusable_phonemes(self, word: str) -> List[str]:
        """Returns list of phonemes in this word that have confusable pairs."""
        return [char for char in word if char in self.confusable_map]
    
    def generate_phonetically_similar_words(self, word: str, count: int = 3) -> List[str]:
        """
        Generates distractor words by substituting confusable phonemes.
        Used for creating quiz options.
        
        Args:
            word: Target Sinhala word
            count: Number of distractors to generate
            
        Returns:
            List of phonetically similar distractor words
        """
        distractors = []
        confusable_positions = [
            (i, char) for i, char in enumerate(word) 
            if char in self.confusable_map
        ]
        
        if not confusable_positions:
            return []
        
        for i, (pos, char) in enumerate(confusable_positions[:count]):
            # Replace with confusable phoneme
            similar_phonemes = self.confusable_map[char]
            if similar_phonemes:
                replacement = similar_phonemes[i % len(similar_phonemes)]
                distractor = word[:pos] + replacement + word[pos+1:]
                distractors.append(distractor)
        
        return distractors[:count]
    
    def calculate_acoustic_distance(self, phoneme1: str, phoneme2: str) -> float:
        """
        Estimates acoustic distance between two phonemes.
        Lower distance = more confusable.
        
        Returns: 0.0 (identical) to 1.0 (very different)
        """
        if phoneme1 == phoneme2:
            return 0.0
        
        if phoneme1 not in PHONEME_FEATURES or phoneme2 not in PHONEME_FEATURES:
            return 0.5  # Unknown, assume moderate distance
        
        f1 = PHONEME_FEATURES[phoneme1]
        f2 = PHONEME_FEATURES[phoneme2]
        
        distance = 0.0
        
        # Different type (stop vs fricative) = big difference
        if f1['type'] != f2['type']:
            distance += 0.4
        
        # Different voicing = moderate difference (but confusable for hearing-impaired!)
        if f1['voicing'] != f2['voicing']:
            distance += 0.2
        
        # Different place = moderate difference
        if f1['place'] != f2['place']:
            distance += 0.3
        
        # Different frequency range = big difference
        if f1['freq_range'] != f2['freq_range']:
            distance += 0.1
        
        return min(1.0, distance)


# Word Difficulty Classification
def classify_word_difficulty(word: str) -> str:
    """
    Classifies a Sinhala word into difficulty levels.
    
    Returns: 'easy', 'medium', 'hard', 'very_hard'
    """
    analyzer = SinhalaPhoneticAnalyzer()
    metrics = analyzer.get_phonetic_complexity(word)
    
    complexity = metrics['overall_complexity']
    
    if complexity < 0.25:
        return 'easy'
    elif complexity < 0.5:
        return 'medium'
    elif complexity < 0.75:
        return 'hard'
    else:
        return 'very_hard'


# Example word lists for testing
SAMPLE_WORDS_BY_DIFFICULTY = {
    'easy': ['අම්මා', 'තාත්තා', 'මල', 'ගස', 'පොත'],  # Simple, common words
    'medium': ['පාසල', 'කෑම', 'වතුර', 'හිරු', 'රිය'],
    'hard': ['පුස්තකාලය', 'ආහාර', 'විශ්වාසය', 'පාඨශාලාව'],
    'very_hard': ['ප්‍රශ්නාර්ථය', 'ස්ථාවරත්වය', 'ශ්‍රීමත්']  # Complex clusters
}


if __name__ == "__main__":
    # Testing
    analyzer = SinhalaPhoneticAnalyzer()
    
    test_words = ["හාවා", "පුස්තකාලය", "ස", "ශ", "අම්මා"]
    
    for word in test_words:
        print(f"\n📝 Word: {word}")
        complexity = analyzer.get_phonetic_complexity(word)
        print(f"   Complexity: {complexity}")
        print(f"   Difficulty: {classify_word_difficulty(word)}")
        
        distractors = analyzer.generate_phonetically_similar_words(word, 3)
        print(f"   Distractors: {distractors}")
