"""
Visualization and Reporting Tools for Phoneme Confusion Analysis

Generates:
- Confusion matrix heatmaps (JSON for frontend visualization)
- Progress charts
- Therapy reports
"""

import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict

try:
    from ml_model.phoneme_analyzer import PhonemeConfusionAnalyzer
    from ml_model.utils.sinhala_phonetics import CONFUSABLE_PAIRS, PHONEME_FEATURES
except ImportError:
    from phoneme_analyzer import PhonemeConfusionAnalyzer
    from utils.sinhala_phonetics import CONFUSABLE_PAIRS, PHONEME_FEATURES


class PhonemeVisualizationGenerator:
    """Generates visualization-ready data for phoneme confusion analysis."""
    
    @staticmethod
    def generate_heatmap_data(analyzer: PhonemeConfusionAnalyzer) -> Dict:
        """
        Generate confusion matrix heatmap data for frontend visualization.
        
        Returns:
            {
                'phonemes': ['ප', 'බ', 'ක', 'ග', ...],
                'matrix': [[0.0, 0.5, 0.1, ...], ...],
                'labels': {
                    'x': 'Target Phoneme',
                    'y': 'Confused With',
                    'title': 'Phoneme Confusion Matrix'
                }
            }
        """
        confusion_matrix = analyzer.get_confusion_matrix()
        
        if not confusion_matrix:
            return {
                'phonemes': [],
                'matrix': [],
                'message': 'No confusion data available yet'
            }
        
        # Get all phonemes (sorted for consistency)
        all_phonemes = sorted(confusion_matrix.keys())
        
        # Build 2D matrix
        matrix = []
        for p1 in all_phonemes:
            row = []
            for p2 in all_phonemes:
                if p1 == p2:
                    row.append(0.0)  # No self-confusion
                else:
                    row.append(confusion_matrix.get(p1, {}).get(p2, 0.0))
            matrix.append(row)
        
        return {
            'phonemes': all_phonemes,
            'matrix': matrix,
            'labels': {
                'x': 'Target Phoneme',
                'y': 'Confused With',
                'title': 'Phoneme Confusion Heatmap',
                'colorbar': 'Confusion Rate'
            },
            'metadata': {
                'total_phonemes': len(all_phonemes),
                'generated_at': datetime.now().isoformat()
            }
        }
    
    @staticmethod
    def generate_progress_chart(analyzer: PhonemeConfusionAnalyzer, days: int = 30) -> Dict:
        """
        Generate time-series data for progress visualization.
        
        Returns:
            {
                'dates': ['2025-12-01', '2025-12-08', ...],
                'accuracy': [0.65, 0.72, 0.78, ...],
                'total_attempts': [20, 25, 30, ...]
            }
        """
        progress_data = analyzer.get_progress_over_time(days=days)
        
        if 'weekly_progress' not in progress_data:
            return {'message': 'No progress data available'}
        
        dates = []
        accuracy = []
        attempts = []
        
        for week_data in progress_data['weekly_progress']:
            dates.append(week_data['week'])
            accuracy.append(week_data['accuracy'])
            attempts.append(week_data['total_attempts'])
        
        return {
            'dates': dates,
            'accuracy': accuracy,
            'total_attempts': attempts,
            'labels': {
                'x': 'Week',
                'y_accuracy': 'Accuracy Rate',
                'y_attempts': 'Total Attempts',
                'title': f'Progress Over Last {days} Days'
            },
            'overall_accuracy': progress_data.get('overall_accuracy', 0)
        }
    
    @staticmethod
    def generate_priority_breakdown(analyzer: PhonemeConfusionAnalyzer) -> Dict:
        """
        Generate pie/bar chart data for priority breakdown.
        
        Returns:
            {
                'labels': ['High Priority', 'Medium Priority', 'Low Priority'],
                'values': [3, 5, 2],
                'colors': ['#ff4444', '#ffaa44', '#44ff44']
            }
        """
        stats = analyzer.get_summary_statistics()
        priority_breakdown = stats.get('priority_breakdown', {})
        
        return {
            'labels': ['High Priority', 'Medium Priority', 'Low Priority'],
            'values': [
                priority_breakdown.get('high', 0),
                priority_breakdown.get('medium', 0),
                priority_breakdown.get('low', 0)
            ],
            'colors': ['#ff4444', '#ffaa44', '#44ff44'],
            'title': 'Phoneme Confusion Priority Distribution'
        }
    
    @staticmethod
    def generate_top_confusions_chart(analyzer: PhonemeConfusionAnalyzer, limit: int = 10) -> Dict:
        """
        Generate horizontal bar chart data for top confusions.
        
        Returns:
            {
                'labels': ['ප/බ', 'ක/ග', 'ස/ශ', ...],
                'values': [0.75, 0.65, 0.55, ...],
                'colors': ['#ff4444', '#ff5555', '#ff6666', ...]
            }
        """
        top_confusions = analyzer.get_top_confusions(limit=limit)
        
        labels = []
        values = []
        colors = []
        
        for confusion in top_confusions:
            labels.append(f"{confusion.phoneme1}/{confusion.phoneme2}")
            values.append(confusion.confusion_rate)
            
            # Color based on confusion rate
            if confusion.confusion_rate >= 0.5:
                colors.append('#ff4444')  # Red (high)
            elif confusion.confusion_rate >= 0.3:
                colors.append('#ffaa44')  # Orange (medium)
            else:
                colors.append('#ffdd44')  # Yellow (low)
        
        return {
            'labels': labels,
            'values': values,
            'colors': colors,
            'title': f'Top {limit} Most Confused Phoneme Pairs',
            'x_label': 'Confusion Rate',
            'y_label': 'Phoneme Pair'
        }


class TherapyReportGenerator:
    """Generates comprehensive therapy reports in various formats."""
    
    @staticmethod
    def generate_text_report(
        analyzer: PhonemeConfusionAnalyzer,
        language: str = 'english'
    ) -> str:
        """
        Generate human-readable text report.
        
        Args:
            analyzer: PhonemeConfusionAnalyzer instance
            language: 'english' or 'sinhala'
            
        Returns:
            Formatted text report
        """
        stats = analyzer.get_summary_statistics()
        recommendations = analyzer.get_therapy_recommendations()
        
        if language == 'sinhala':
            return TherapyReportGenerator._generate_sinhala_report(stats, recommendations)
        else:
            return TherapyReportGenerator._generate_english_report(stats, recommendations)
    
    @staticmethod
    def _generate_english_report(stats: Dict, recommendations: List) -> str:
        """Generate English language report."""
        report_lines = [
            "=" * 70,
            "PHONEME CONFUSION ANALYSIS REPORT",
            "=" * 70,
            "",
            f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "SUMMARY STATISTICS",
            "-" * 70,
            f"Total Phoneme Pairs Tracked: {stats.get('total_phoneme_pairs_tracked', 0)}",
            f"Total Errors Recorded: {stats.get('total_errors', 0)}",
            f"Total Correct Responses: {stats.get('total_correct', 0)}",
            f"Overall Accuracy: {stats.get('overall_accuracy', 0):.1%}",
            "",
            "PRIORITY BREAKDOWN",
            "-" * 70,
        ]
        
        priority_breakdown = stats.get('priority_breakdown', {})
        report_lines.extend([
            f"High Priority (>50% confusion): {priority_breakdown.get('high', 0)} pairs",
            f"Medium Priority (30-50%): {priority_breakdown.get('medium', 0)} pairs",
            f"Low Priority (<30%): {priority_breakdown.get('low', 0)} pairs",
            "",
            "TOP THERAPY RECOMMENDATIONS",
            "-" * 70,
        ])
        
        for i, rec in enumerate(recommendations[:5], 1):
            p1, p2 = rec.phoneme_pair
            report_lines.extend([
                "",
                f"{i}. Phoneme Pair: {p1} / {p2}",
                f"   Priority: {rec.priority.upper()}",
                f"   Confusion Rate: {rec.confusion_rate:.1%}",
                f"   Explanation: {rec.acoustic_explanation}",
                f"   Recommended Practice Words: {', '.join(rec.recommended_words[:5])}",
                f"   Exercises:",
            ])
            for exercise in rec.practice_exercises[:3]:
                report_lines.append(f"      • {exercise}")
        
        report_lines.extend([
            "",
            "=" * 70,
            "END OF REPORT",
            "=" * 70
        ])
        
        return "\n".join(report_lines)
    
    @staticmethod
    def _generate_sinhala_report(stats: Dict, recommendations: List) -> str:
        """Generate Sinhala language report (simplified)."""
        report_lines = [
            "=" * 70,
            "ශ්‍රවණ චිකිත්සක වාර්තාව - ෆෝනීම් අවුල් විශ්ලේෂණය",
            "=" * 70,
            "",
            f"වාර්තාව ජනනය කළ දිනය: {datetime.now().strftime('%Y-%m-%d')}",
            "",
            "සාරාංශ සංඛ්‍යාලේඛන",
            "-" * 70,
            f"නිරීක්ෂණය කළ ශබ්ද යුගල: {stats.get('total_phoneme_pairs_tracked', 0)}",
            f"දෝෂ ගණන: {stats.get('total_errors', 0)}",
            f"නිවැරදි පිළිතුරු: {stats.get('total_correct', 0)}",
            f"සමස්ත නිරවද්‍යතාව: {stats.get('overall_accuracy', 0):.1%}",
            "",
            "ප්‍රමුඛතා බෙදීම",
            "-" * 70,
        ]
        
        priority_breakdown = stats.get('priority_breakdown', {})
        report_lines.extend([
            f"ඉහළ ප්‍රමුඛතාවය: {priority_breakdown.get('high', 0)} යුගල",
            f"මධ්‍යම ප්‍රමුඛතාවය: {priority_breakdown.get('medium', 0)} යුගල",
            f"අඩු ප්‍රමුඛතාවය: {priority_breakdown.get('low', 0)} යුගල",
            "",
            "චිකිත්සක නිර්දේශ",
            "-" * 70,
        ])
        
        for i, rec in enumerate(recommendations[:5], 1):
            p1, p2 = rec.phoneme_pair
            report_lines.extend([
                "",
                f"{i}. ශබ්ද යුගලය: {p1} / {p2}",
                f"   ප්‍රමුඛතාවය: {rec.priority}",
                f"   අවුල් අනුපාතය: {rec.confusion_rate:.1%}",
                f"   පුහුණු වචන: {', '.join(rec.recommended_words[:5])}",
            ])
        
        report_lines.extend([
            "",
            "=" * 70,
            "වාර්තාවේ අවසානය",
            "=" * 70
        ])
        
        return "\n".join(report_lines)
    
    @staticmethod
    def generate_json_report(analyzer: PhonemeConfusionAnalyzer) -> Dict:
        """
        Generate comprehensive JSON report for API consumption.
        
        Returns:
            Complete report data as nested dictionary
        """
        stats = analyzer.get_summary_statistics()
        recommendations = analyzer.get_therapy_recommendations()
        
        viz_gen = PhonemeVisualizationGenerator()
        
        return {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'user_id': analyzer.user_id,
                'report_type': 'phoneme_confusion_analysis'
            },
            'summary_statistics': stats,
            'therapy_recommendations': [rec.to_dict() for rec in recommendations],
            'visualizations': {
                'heatmap': viz_gen.generate_heatmap_data(analyzer),
                'progress_chart': viz_gen.generate_progress_chart(analyzer),
                'priority_breakdown': viz_gen.generate_priority_breakdown(analyzer),
                'top_confusions': viz_gen.generate_top_confusions_chart(analyzer)
            },
            'top_confusions': [
                c.to_dict() for c in analyzer.get_top_confusions(limit=10)
            ]
        }


# Testing
if __name__ == "__main__":
    print("Testing Visualization and Reporting Tools\n")
    
    # Create analyzer with sample data
    from ml_model.phoneme_analyzer import PhonemeConfusionAnalyzer
    
    analyzer = PhonemeConfusionAnalyzer("test_user")
    
    # Add sample confusion data
    test_data = [
        ("පල", "බල", False),
        ("පල", "බල", False),
        ("පල", "පල", True),
        ("කත", "ගත", False),
        ("සර", "ශර", False),
        ("සර", "සර", True),
    ]
    
    for target, selected, correct in test_data:
        analyzer.record_answer(target, selected, correct)
    
    # Test visualizations
    viz_gen = PhonemeVisualizationGenerator()
    
    print("1. Heatmap Data:")
    heatmap = viz_gen.generate_heatmap_data(analyzer)
    print(json.dumps(heatmap, indent=2, ensure_ascii=False))
    
    print("\n2. Top Confusions Chart:")
    top_chart = viz_gen.generate_top_confusions_chart(analyzer, limit=5)
    print(json.dumps(top_chart, indent=2, ensure_ascii=False))
    
    print("\n3. Text Report (English):")
    report_gen = TherapyReportGenerator()
    text_report = report_gen.generate_text_report(analyzer, language='english')
    print(text_report)
    
    print("\n4. JSON Report:")
    json_report = report_gen.generate_json_report(analyzer)
    print(f"Generated JSON report with {len(json_report)} top-level keys")
