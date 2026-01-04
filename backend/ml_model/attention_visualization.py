"""
Feature 4: Attention Heatmap Visualization Generator
=====================================================

Generates visual representations of attention data:
- Heatmap overlays for screen regions
- Attention timeline charts
- Gaze path visualizations
- Focus quality dashboards
- Comparative attention analysis

Author: AI Research Team
Date: 2025-01-01
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import math


class AttentionVisualizationGenerator:
    """
    Generates visualization-ready data for attention heatmaps and analytics.
    Designed for frontend rendering with Chart.js, D3.js, or Canvas.
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id
    
    def generate_heatmap_overlay(
        self,
        heatmap_data: List[List[float]],
        grid_size: Tuple[int, int],
        color_scheme: str = 'hot'  # 'hot', 'cool', 'rainbow'
    ) -> Dict:
        """
        Generate heatmap overlay data for frontend rendering.
        
        Args:
            heatmap_data: 2D array of attention scores
            grid_size: (rows, cols) dimensions
            color_scheme: Color gradient scheme
        
        Returns:
            Dictionary with canvas/SVG rendering data
        """
        rows, cols = grid_size
        
        # Normalize scores to 0-1 range
        max_score = max(max(row) for row in heatmap_data) if heatmap_data else 1.0
        normalized = [
            [score / max_score if max_score > 0 else 0 for score in row]
            for row in heatmap_data
        ]
        
        # Generate color mapping
        color_map = self._get_color_map(color_scheme)
        
        # Create cell data for rendering
        cells = []
        cell_width = 100.0 / cols  # Percentage
        cell_height = 100.0 / rows
        
        for row_idx, row in enumerate(normalized):
            for col_idx, intensity in enumerate(row):
                cells.append({
                    'x': col_idx * cell_width,
                    'y': row_idx * cell_height,
                    'width': cell_width,
                    'height': cell_height,
                    'intensity': intensity,
                    'color': self._intensity_to_color(intensity, color_map),
                    'alpha': intensity * 0.7,  # Transparency based on intensity
                    'attention_score': heatmap_data[row_idx][col_idx]
                })
        
        return {
            'type': 'heatmap_overlay',
            'cells': cells,
            'grid_size': grid_size,
            'color_scheme': color_scheme,
            'max_attention_score': max_score
        }
    
    def generate_gaze_path_visualization(
        self,
        gaze_history: List[Dict],
        max_points: int = 100
    ) -> Dict:
        """
        Generate gaze path visualization (spaghetti plot).
        
        Shows the path of eye movements over time.
        """
        # Take last N points for clarity
        recent_gazes = gaze_history[-max_points:] if len(gaze_history) > max_points else gaze_history
        
        if not recent_gazes:
            return {'type': 'gaze_path', 'points': [], 'lines': []}
        
        # Create point data
        points = []
        for idx, gaze in enumerate(recent_gazes):
            points.append({
                'x': gaze['x'] * 100,  # Convert to percentage
                'y': gaze['y'] * 100,
                'timestamp': gaze['timestamp'],
                'sequence': idx,
                'opacity': 0.3 + (idx / len(recent_gazes)) * 0.7  # Fade older points
            })
        
        # Create line segments connecting points
        lines = []
        for i in range(len(points) - 1):
            lines.append({
                'from': points[i],
                'to': points[i + 1],
                'opacity': points[i]['opacity']
            })
        
        return {
            'type': 'gaze_path',
            'points': points,
            'lines': lines,
            'total_points': len(gaze_history)
        }
    
    def generate_attention_timeline(
        self,
        attention_history: List[Dict],
        time_window_minutes: int = 30
    ) -> Dict:
        """
        Generate timeline chart showing attention over time.
        
        Returns Chart.js compatible data structure.
        """
        if not attention_history:
            return {
                'type': 'timeline',
                'labels': [],
                'datasets': []
            }
        
        # Group by time buckets (1-minute intervals)
        time_buckets = {}
        
        for entry in attention_history:
            timestamp = datetime.fromisoformat(entry['timestamp'])
            bucket_key = timestamp.strftime('%Y-%m-%d %H:%M')
            
            if bucket_key not in time_buckets:
                time_buckets[bucket_key] = {
                    'attention_scores': [],
                    'confidence_scores': []
                }
            
            time_buckets[bucket_key]['attention_scores'].append(entry.get('attention_score', 0))
            time_buckets[bucket_key]['confidence_scores'].append(entry.get('confidence', 1.0))
        
        # Calculate averages for each bucket
        labels = sorted(time_buckets.keys())
        attention_data = []
        confidence_data = []
        
        for label in labels:
            bucket = time_buckets[label]
            avg_attention = sum(bucket['attention_scores']) / len(bucket['attention_scores'])
            avg_confidence = sum(bucket['confidence_scores']) / len(bucket['confidence_scores'])
            
            attention_data.append(round(avg_attention, 2))
            confidence_data.append(round(avg_confidence * 100, 2))
        
        return {
            'type': 'timeline',
            'labels': [self._format_time_label(l) for l in labels],
            'datasets': [
                {
                    'label': 'Attention Score',
                    'data': attention_data,
                    'borderColor': 'rgb(75, 192, 192)',
                    'backgroundColor': 'rgba(75, 192, 192, 0.2)',
                    'tension': 0.4
                },
                {
                    'label': 'Gaze Confidence',
                    'data': confidence_data,
                    'borderColor': 'rgb(153, 102, 255)',
                    'backgroundColor': 'rgba(153, 102, 255, 0.2)',
                    'tension': 0.4
                }
            ]
        }
    
    def generate_focus_quality_dashboard(
        self,
        focus_quality: float,
        drift_count: int,
        avg_fixation_duration: float,
        attention_distribution: Dict
    ) -> Dict:
        """
        Generate dashboard summary for focus quality metrics.
        """
        # Determine quality level
        if focus_quality >= 70:
            quality_level = 'Excellent'
            quality_color = 'green'
        elif focus_quality >= 50:
            quality_level = 'Good'
            quality_color = 'yellow'
        elif focus_quality >= 30:
            quality_level = 'Fair'
            quality_color = 'orange'
        else:
            quality_level = 'Poor'
            quality_color = 'red'
        
        # Create gauge data
        gauge_data = {
            'current_score': focus_quality,
            'label': quality_level,
            'color': quality_color,
            'max_score': 100
        }
        
        # Drift analysis
        drift_severity = 'Low' if drift_count < 3 else 'Medium' if drift_count < 6 else 'High'
        
        # Top attention zones
        top_zones = sorted(
            attention_distribution.items(),
            key=lambda x: x[1].get('attention_score', 0),
            reverse=True
        )[:5]
        
        return {
            'gauge': gauge_data,
            'metrics': {
                'focus_quality': focus_quality,
                'drift_events': drift_count,
                'drift_severity': drift_severity,
                'avg_fixation_duration': round(avg_fixation_duration, 1),
                'quality_level': quality_level
            },
            'top_attention_zones': [
                {
                    'zone': zone_id,
                    'score': data['attention_score'],
                    'visits': data.get('visit_count', 0)
                }
                for zone_id, data in top_zones
            ],
            'recommendations': self._generate_focus_recommendations(
                focus_quality,
                drift_count,
                avg_fixation_duration
            )
        }
    
    def generate_comparative_heatmap(
        self,
        before_heatmap: List[List[float]],
        after_heatmap: List[List[float]],
        grid_size: Tuple[int, int]
    ) -> Dict:
        """
        Generate comparative heatmap (before vs after intervention/change).
        
        Useful for A/B testing UI layouts or measuring therapy progress.
        """
        rows, cols = grid_size
        
        # Calculate difference map
        difference_map = []
        for row_idx in range(rows):
            diff_row = []
            for col_idx in range(cols):
                before_val = before_heatmap[row_idx][col_idx] if row_idx < len(before_heatmap) and col_idx < len(before_heatmap[row_idx]) else 0
                after_val = after_heatmap[row_idx][col_idx] if row_idx < len(after_heatmap) and col_idx < len(after_heatmap[row_idx]) else 0
                diff_row.append(after_val - before_val)
            difference_map.append(diff_row)
        
        # Find improvements and regressions
        improvements = []
        regressions = []
        
        for row_idx, row in enumerate(difference_map):
            for col_idx, diff in enumerate(row):
                if diff > 5:  # Significant improvement
                    improvements.append({
                        'row': row_idx,
                        'col': col_idx,
                        'improvement': diff
                    })
                elif diff < -5:  # Significant regression
                    regressions.append({
                        'row': row_idx,
                        'col': col_idx,
                        'regression': abs(diff)
                    })
        
        return {
            'type': 'comparative_heatmap',
            'before': before_heatmap,
            'after': after_heatmap,
            'difference': difference_map,
            'improvements': improvements,
            'regressions': regressions,
            'overall_change': sum(sum(row) for row in difference_map)
        }
    
    def _get_color_map(self, scheme: str) -> List[Tuple[int, int, int]]:
        """Get color gradient for heatmap."""
        if scheme == 'hot':
            return [
                (0, 0, 128),      # Dark blue (cold)
                (0, 128, 255),    # Blue
                (0, 255, 255),    # Cyan
                (255, 255, 0),    # Yellow
                (255, 128, 0),    # Orange
                (255, 0, 0)       # Red (hot)
            ]
        elif scheme == 'cool':
            return [
                (255, 255, 255),  # White (cold)
                (200, 200, 255),  # Light blue
                (100, 100, 255),  # Blue
                (0, 0, 200)       # Dark blue (hot)
            ]
        else:  # rainbow
            return [
                (148, 0, 211),    # Violet
                (75, 0, 130),     # Indigo
                (0, 0, 255),      # Blue
                (0, 255, 0),      # Green
                (255, 255, 0),    # Yellow
                (255, 127, 0),    # Orange
                (255, 0, 0)       # Red
            ]
    
    def _intensity_to_color(
        self,
        intensity: float,
        color_map: List[Tuple[int, int, int]]
    ) -> str:
        """Convert intensity (0-1) to RGB color string."""
        if intensity == 0:
            return 'rgba(0, 0, 0, 0)'  # Transparent
        
        # Map intensity to color gradient
        segment_count = len(color_map) - 1
        scaled_intensity = intensity * segment_count
        segment_index = min(int(scaled_intensity), segment_count - 1)
        
        # Interpolate between two colors
        color1 = color_map[segment_index]
        color2 = color_map[segment_index + 1]
        
        t = scaled_intensity - segment_index  # Interpolation factor
        
        r = int(color1[0] + (color2[0] - color1[0]) * t)
        g = int(color1[1] + (color2[1] - color1[1]) * t)
        b = int(color1[2] + (color2[2] - color1[2]) * t)
        
        return f'rgb({r}, {g}, {b})'
    
    def _format_time_label(self, timestamp_str: str) -> str:
        """Format timestamp for chart labels."""
        try:
            dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M')
            return dt.strftime('%H:%M')
        except:
            return timestamp_str
    
    def _generate_focus_recommendations(
        self,
        focus_quality: float,
        drift_count: int,
        avg_fixation_duration: float
    ) -> List[str]:
        """Generate recommendations based on focus metrics."""
        recommendations = []
        
        if focus_quality < 50:
            recommendations.append('Consider adding more visual cues to guide attention')
            recommendations.append('Reduce session duration to prevent fatigue')
        
        if drift_count > 5:
            recommendations.append('Frequent attention drift detected - add engaging visual elements')
            recommendations.append('Consider scheduled micro-breaks every 5 minutes')
        
        if avg_fixation_duration < 200:
            recommendations.append('Short fixation duration - content may be too complex')
            recommendations.append('Simplify visual elements and reduce on-screen clutter')
        elif avg_fixation_duration > 1000:
            recommendations.append('Very long fixations - child may be struggling')
            recommendations.append('Provide more scaffolding or hints')
        
        if not recommendations:
            recommendations.append('Focus quality is good - maintain current approach')
        
        return recommendations


class AttentionReportGenerator:
    """Generate comprehensive attention analysis reports."""
    
    def __init__(self, user_id: str, child_name: str):
        self.user_id = user_id
        self.child_name = child_name
    
    def generate_therapist_report(
        self,
        attention_stats: Dict,
        heatmap_data: Dict,
        recommendations: List[Dict],
        language: str = 'english'
    ) -> str:
        """
        Generate detailed attention report for therapists.
        """
        if language == 'sinhala':
            return self._generate_sinhala_therapist_report(
                attention_stats,
                heatmap_data,
                recommendations
            )
        
        # English report
        report = f"""
VISUAL ATTENTION ANALYSIS REPORT
Child: {self.child_name}
User ID: {self.user_id}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

═══════════════════════════════════════════════════════════════

ATTENTION METRICS SUMMARY
-------------------------
Total Gaze Points Recorded: {attention_stats.get('total_gaze_points', 0)}
Total Fixation Time: {attention_stats.get('total_fixation_time', 0)/1000:.1f} seconds
Average Fixation Duration: {attention_stats.get('average_fixation_duration', 0):.1f} ms
Attention Drift Events: {attention_stats.get('drift_events', 0)}
Focus Quality Score: {attention_stats.get('focus_quality', 0):.1f}/100

FOCUS QUALITY INTERPRETATION:
{self._interpret_focus_quality(attention_stats.get('focus_quality', 0))}

ATTENTION DISTRIBUTION
----------------------
Top 5 Most-Attended Screen Zones:
"""
        
        for idx, zone in enumerate(attention_stats.get('top_attention_zones', [])[:5], 1):
            report += f"\n{idx}. Zone {zone['zone_id']}: {zone['attention_score']:.1f} points ({zone['visit_count']} visits)"
        
        report += f"""

ATTENTION DRIFT ANALYSIS
------------------------
Total Drift Events: {attention_stats.get('drift_events', 0)}
Impact: {self._interpret_drift_impact(attention_stats.get('drift_events', 0))}

CLINICAL RECOMMENDATIONS
------------------------
"""
        
        for idx, rec in enumerate(recommendations, 1):
            report += f"\n{idx}. [{rec['severity'].upper()}] {rec['message']}"
        
        report += f"""

UI/CONTENT OPTIMIZATION SUGGESTIONS
-----------------------------------
Based on heatmap analysis:
• High-attention zones detected: {len(heatmap_data.get('hotspots', []))} zones
• Max attention score: {heatmap_data.get('max_attention_score', 0):.1f}

Consider placing critical content in high-attention zones for maximum effectiveness.

═══════════════════════════════════════════════════════════════
Report generated by Multimodal Attention Tracking System v1.0
Medical Framework: Eye-tracking research (Duchowski, 2007)
"""
        
        return report
    
    def _interpret_focus_quality(self, score: float) -> str:
        """Interpret focus quality score."""
        if score >= 70:
            return "Excellent - Child maintains strong visual attention with minimal drift."
        elif score >= 50:
            return "Good - Generally attentive with some minor distractions."
        elif score >= 30:
            return "Fair - Moderate attention issues. Consider shorter sessions or breaks."
        else:
            return "Poor - Significant attention difficulties. Recommend clinical evaluation and modified intervention."
    
    def _interpret_drift_impact(self, drift_count: int) -> str:
        """Interpret drift event count."""
        if drift_count < 3:
            return "Minimal - Within normal range for age group."
        elif drift_count < 6:
            return "Moderate - Some attention regulation challenges."
        else:
            return "High - Significant attention maintenance difficulties. Intervention recommended."
    
    def _generate_sinhala_therapist_report(
        self,
        attention_stats: Dict,
        heatmap_data: Dict,
        recommendations: List[Dict]
    ) -> str:
        """Generate Sinhala language report."""
        return f"""
දෘශ්‍ය අවධානය විශ්ලේෂණ වාර්තාව
දරුවා: {self.child_name}
පරිශීලක හැඳුනුම්පත: {self.user_id}
දිනය: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

═══════════════════════════════════════════════════════════════

අවධානය මිනුම් සාරාංශය
---------------------
සම්පූර්ණ ඇස් ලකුණු: {attention_stats.get('total_gaze_points', 0)}
සම්පූර්ණ නිවැරදි කාලය: {attention_stats.get('total_fixation_time', 0)/1000:.1f} තත්පර
සාමාන්‍ය නිවැරදි කාල සීමාව: {attention_stats.get('average_fixation_duration', 0):.1f} ms
අවධානය නැති වීම්: {attention_stats.get('drift_events', 0)}
අවධාන ගුණාත්මක ලකුණු: {attention_stats.get('focus_quality', 0):.1f}/100

වෛද්‍ය නිර්දේශ
--------------
"""
        
        for idx, rec in enumerate(recommendations, 1):
            report += f"\n{idx}. {rec['message']}"
        
        return report
