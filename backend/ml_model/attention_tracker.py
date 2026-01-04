"""
Feature 4: Visual Attention Heatmap
====================================

Tracks eye gaze and attention patterns during therapy sessions.
Uses spatial clustering and time-weighted aggregation to identify:
- High-attention zones (where child looks most)
- Low-attention zones (ignored areas)
- Attention drift patterns (loss of focus over time)
- Optimal UI placement recommendations

Medical/Research Basis:
- Eye-tracking research (Duchowski, 2007)
- Visual attention in learning disabilities (Franceschini et al., 2012)
- Gaze-based attention assessment (Just & Carpenter, 1976)

Author: AI Research Team
Date: 2025-01-01
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from collections import deque
from datetime import datetime
import json
import math


@dataclass
class GazePoint:
    """Single gaze data point with screen coordinates and timestamp."""
    x: float  # Screen x coordinate (0-1, normalized)
    y: float  # Screen y coordinate (0-1, normalized)
    timestamp: datetime
    confidence: float = 1.0  # Gaze detection confidence (0-1)
    fixation_duration: float = 0.0  # Duration of fixation in ms
    
    def to_dict(self):
        return {
            'x': self.x,
            'y': self.y,
            'timestamp': self.timestamp.isoformat(),
            'confidence': self.confidence,
            'fixation_duration': self.fixation_duration
        }


@dataclass
class AttentionZone:
    """Spatial zone with aggregated attention metrics."""
    zone_id: str  # e.g., "top_left", "center", "question_area"
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    total_fixation_time: float  # Total ms spent in this zone
    visit_count: int  # Number of times gaze entered zone
    average_confidence: float
    attention_score: float  # 0-100 weighted attention score
    
    def to_dict(self):
        return asdict(self)


@dataclass
class AttentionDrift:
    """Detected attention drift event (loss of focus)."""
    start_time: datetime
    duration: float  # Duration in seconds
    drift_type: str  # 'wandering', 'distraction', 'fatigue'
    severity: float  # 0-1 (1 = severe drift)
    recovery_time: Optional[float] = None  # Time to refocus (seconds)
    
    def to_dict(self):
        return {
            'start_time': self.start_time.isoformat(),
            'duration': self.duration,
            'drift_type': self.drift_type,
            'severity': self.severity,
            'recovery_time': self.recovery_time
        }


class AttentionHeatmapTracker:
    """
    Tracks visual attention patterns and generates heatmap data.
    
    Features:
    - Real-time gaze point recording
    - Spatial clustering into attention zones
    - Temporal attention drift detection
    - Heatmap generation for visualization
    - UI optimization recommendations
    """
    
    def __init__(
        self,
        user_id: str,
        grid_size: Tuple[int, int] = (10, 10),
        min_fixation_duration: float = 100.0,  # ms
        drift_threshold: float = 5.0  # seconds without fixation
    ):
        self.user_id = user_id
        self.grid_size = grid_size  # Divide screen into grid cells
        self.min_fixation_duration = min_fixation_duration
        self.drift_threshold = drift_threshold
        
        # Data storage
        self.gaze_history: deque = deque(maxlen=5000)  # Last 5000 gaze points
        self.attention_zones: Dict[str, AttentionZone] = {}
        self.drift_events: List[AttentionDrift] = []
        
        # State tracking
        self.current_fixation_start: Optional[datetime] = None
        self.current_fixation_point: Optional[Tuple[float, float]] = None
        self.last_gaze_time: Optional[datetime] = None
        
        # Initialize grid zones
        self._initialize_grid_zones()
    
    def _initialize_grid_zones(self):
        """Create grid-based attention zones."""
        rows, cols = self.grid_size
        cell_width = 1.0 / cols
        cell_height = 1.0 / rows
        
        for row in range(rows):
            for col in range(cols):
                zone_id = f"grid_{row}_{col}"
                self.attention_zones[zone_id] = AttentionZone(
                    zone_id=zone_id,
                    x_min=col * cell_width,
                    y_min=row * cell_height,
                    x_max=(col + 1) * cell_width,
                    y_max=(row + 1) * cell_height,
                    total_fixation_time=0.0,
                    visit_count=0,
                    average_confidence=0.0,
                    attention_score=0.0
                )
    
    def record_gaze(
        self,
        x: float,
        y: float,
        confidence: float = 1.0,
        timestamp: Optional[datetime] = None
    ) -> Dict:
        """
        Record a gaze point and update attention metrics.
        
        Args:
            x: Normalized x coordinate (0-1)
            y: Normalized y coordinate (0-1)
            confidence: Gaze detection confidence (0-1)
            timestamp: Time of gaze (default: now)
        
        Returns:
            Dictionary with current attention state
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Detect fixation (gaze staying in same area)
        fixation_duration = 0.0
        is_fixation = False
        
        if self.current_fixation_point:
            dist = self._euclidean_distance(
                (x, y),
                self.current_fixation_point
            )
            
            # If gaze is within fixation radius (0.05 = ~5% of screen)
            if dist < 0.05:
                fixation_duration = (timestamp - self.current_fixation_start).total_seconds() * 1000
                is_fixation = True
            else:
                # Fixation ended, start new one
                self.current_fixation_start = timestamp
                self.current_fixation_point = (x, y)
        else:
            # First fixation
            self.current_fixation_start = timestamp
            self.current_fixation_point = (x, y)
        
        # Create gaze point
        gaze_point = GazePoint(
            x=x,
            y=y,
            timestamp=timestamp,
            confidence=confidence,
            fixation_duration=fixation_duration
        )
        
        self.gaze_history.append(gaze_point)
        
        # Update attention zones
        self._update_attention_zones(gaze_point)
        
        # Detect attention drift
        drift_detected = self._detect_attention_drift(timestamp)
        
        self.last_gaze_time = timestamp
        
        return {
            'gaze_recorded': True,
            'is_fixation': is_fixation,
            'fixation_duration': fixation_duration,
            'drift_detected': drift_detected,
            'total_gaze_points': len(self.gaze_history)
        }
    
    def _update_attention_zones(self, gaze_point: GazePoint):
        """Update attention metrics for relevant zones."""
        for zone_id, zone in self.attention_zones.items():
            if self._point_in_zone(gaze_point, zone):
                # Update zone metrics
                zone.total_fixation_time += gaze_point.fixation_duration
                zone.visit_count += 1
                
                # Update average confidence (exponential moving average)
                alpha = 0.1
                zone.average_confidence = (
                    alpha * gaze_point.confidence +
                    (1 - alpha) * zone.average_confidence
                )
                
                # Calculate attention score (weighted by fixation time and confidence)
                zone.attention_score = min(
                    100.0,
                    (zone.total_fixation_time / 1000.0) * zone.average_confidence
                )
    
    def _point_in_zone(self, gaze_point: GazePoint, zone: AttentionZone) -> bool:
        """Check if gaze point falls within zone boundaries."""
        return (
            zone.x_min <= gaze_point.x <= zone.x_max and
            zone.y_min <= gaze_point.y <= zone.y_max
        )
    
    def _euclidean_distance(
        self,
        point1: Tuple[float, float],
        point2: Tuple[float, float]
    ) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt(
            (point1[0] - point2[0]) ** 2 +
            (point1[1] - point2[1]) ** 2
        )
    
    def _detect_attention_drift(self, current_time: datetime) -> bool:
        """
        Detect if attention has drifted (no fixations for threshold time).
        
        Returns:
            True if drift detected, False otherwise
        """
        if not self.last_gaze_time:
            return False
        
        time_since_last_gaze = (current_time - self.last_gaze_time).total_seconds()
        
        if time_since_last_gaze > self.drift_threshold:
            # Determine drift type based on duration
            if time_since_last_gaze < 10:
                drift_type = 'wandering'
                severity = 0.3
            elif time_since_last_gaze < 30:
                drift_type = 'distraction'
                severity = 0.6
            else:
                drift_type = 'fatigue'
                severity = 0.9
            
            # Record drift event
            drift_event = AttentionDrift(
                start_time=self.last_gaze_time,
                duration=time_since_last_gaze,
                drift_type=drift_type,
                severity=severity
            )
            
            self.drift_events.append(drift_event)
            return True
        
        return False
    
    def generate_heatmap_data(self) -> Dict:
        """
        Generate heatmap visualization data.
        
        Returns:
            Dictionary with heatmap grid and statistics
        """
        # Create 2D heatmap array
        rows, cols = self.grid_size
        heatmap = [[0.0 for _ in range(cols)] for _ in range(rows)]
        
        # Fill heatmap with attention scores
        for zone_id, zone in self.attention_zones.items():
            if zone_id.startswith('grid_'):
                _, row, col = zone_id.split('_')
                row, col = int(row), int(col)
                heatmap[row][col] = zone.attention_score
        
        # Find hotspots (zones with high attention)
        max_score = max(
            zone.attention_score
            for zone in self.attention_zones.values()
        ) if self.attention_zones else 1.0
        
        hotspots = [
            {
                'zone_id': zone.zone_id,
                'x_center': (zone.x_min + zone.x_max) / 2,
                'y_center': (zone.y_min + zone.y_max) / 2,
                'attention_score': zone.attention_score,
                'intensity': zone.attention_score / max_score if max_score > 0 else 0
            }
            for zone in self.attention_zones.values()
            if zone.attention_score > max_score * 0.5  # Top 50% attention zones
        ]
        
        return {
            'heatmap': heatmap,
            'grid_size': self.grid_size,
            'hotspots': hotspots,
            'max_attention_score': max_score,
            'total_gaze_points': len(self.gaze_history)
        }
    
    def get_attention_statistics(self) -> Dict:
        """
        Get comprehensive attention statistics.
        
        Returns:
            Dictionary with attention metrics
        """
        if not self.gaze_history:
            return {
                'total_gaze_points': 0,
                'total_fixation_time': 0.0,
                'average_fixation_duration': 0.0,
                'attention_distribution': {},
                'drift_events': 0,
                'focus_quality': 0.0
            }
        
        # Calculate statistics
        total_fixation_time = sum(
            gaze.fixation_duration for gaze in self.gaze_history
        )
        
        fixations = [
            gaze for gaze in self.gaze_history
            if gaze.fixation_duration >= self.min_fixation_duration
        ]
        
        avg_fixation_duration = (
            sum(f.fixation_duration for f in fixations) / len(fixations)
            if fixations else 0.0
        )
        
        # Attention distribution (which zones get most attention)
        sorted_zones = sorted(
            self.attention_zones.values(),
            key=lambda z: z.attention_score,
            reverse=True
        )
        
        attention_distribution = {
            zone.zone_id: {
                'attention_score': zone.attention_score,
                'visit_count': zone.visit_count,
                'fixation_time': zone.total_fixation_time
            }
            for zone in sorted_zones[:10]  # Top 10 zones
        }
        
        # Focus quality score (0-100)
        # Based on: fixation stability, drift frequency, confidence
        focus_quality = self._calculate_focus_quality()
        
        return {
            'total_gaze_points': len(self.gaze_history),
            'total_fixation_time': total_fixation_time,
            'average_fixation_duration': avg_fixation_duration,
            'attention_distribution': attention_distribution,
            'drift_events': len(self.drift_events),
            'focus_quality': focus_quality,
            'top_attention_zones': [z.to_dict() for z in sorted_zones[:5]]
        }
    
    def _calculate_focus_quality(self) -> float:
        """
        Calculate overall focus quality score (0-100).
        
        Factors:
        - Fixation stability (fewer saccades = better)
        - Low drift events
        - High gaze confidence
        """
        if not self.gaze_history:
            return 0.0
        
        # Fixation stability (ratio of fixations to total gazes)
        fixations = [
            g for g in self.gaze_history
            if g.fixation_duration >= self.min_fixation_duration
        ]
        stability_score = len(fixations) / len(self.gaze_history) * 100
        
        # Drift penalty
        drift_penalty = min(30, len(self.drift_events) * 5)
        
        # Average confidence
        avg_confidence = sum(g.confidence for g in self.gaze_history) / len(self.gaze_history)
        confidence_score = avg_confidence * 100
        
        # Weighted combination
        focus_quality = (
            stability_score * 0.4 +
            confidence_score * 0.4 +
            (100 - drift_penalty) * 0.2
        )
        
        return min(100.0, max(0.0, focus_quality))
    
    def get_ui_recommendations(self) -> List[Dict]:
        """
        Generate UI optimization recommendations based on attention patterns.
        
        Returns:
            List of recommendations for improving UI/content placement
        """
        recommendations = []
        
        # Find low-attention zones
        low_attention_zones = [
            zone for zone in self.attention_zones.values()
            if zone.attention_score < 10.0 and zone.visit_count < 3
        ]
        
        if low_attention_zones:
            recommendations.append({
                'type': 'low_attention_area',
                'severity': 'medium',
                'message': f'Detected {len(low_attention_zones)} low-attention zones. Consider moving important content to center or top-center.',
                'affected_zones': [z.zone_id for z in low_attention_zones[:5]]
            })
        
        # Check for attention drift
        if len(self.drift_events) > 5:
            recommendations.append({
                'type': 'frequent_drift',
                'severity': 'high',
                'message': f'Detected {len(self.drift_events)} attention drift events. Consider adding more visual engagement elements or reducing session duration.',
                'drift_count': len(self.drift_events)
            })
        
        # Check focus quality
        focus_quality = self._calculate_focus_quality()
        if focus_quality < 40:
            recommendations.append({
                'type': 'low_focus_quality',
                'severity': 'high',
                'message': f'Focus quality is low ({focus_quality:.1f}/100). Child may need breaks or content simplification.',
                'focus_score': focus_quality
            })
        
        return recommendations
    
    def save_state(self) -> Dict:
        """Save tracker state for persistence."""
        return {
            'user_id': self.user_id,
            'grid_size': self.grid_size,
            'min_fixation_duration': self.min_fixation_duration,
            'drift_threshold': self.drift_threshold,
            'gaze_history': [g.to_dict() for g in list(self.gaze_history)[-100:]],  # Last 100
            'attention_zones': {k: v.to_dict() for k, v in self.attention_zones.items()},
            'drift_events': [d.to_dict() for d in self.drift_events],
            'statistics': self.get_attention_statistics()
        }
    
    @classmethod
    def load_state(cls, state: Dict) -> 'AttentionHeatmapTracker':
        """Load tracker from saved state."""
        tracker = cls(
            user_id=state['user_id'],
            grid_size=tuple(state['grid_size']),
            min_fixation_duration=state['min_fixation_duration'],
            drift_threshold=state['drift_threshold']
        )
        
        # Restore gaze history
        for gaze_dict in state.get('gaze_history', []):
            gaze = GazePoint(
                x=gaze_dict['x'],
                y=gaze_dict['y'],
                timestamp=datetime.fromisoformat(gaze_dict['timestamp']),
                confidence=gaze_dict['confidence'],
                fixation_duration=gaze_dict['fixation_duration']
            )
            tracker.gaze_history.append(gaze)
        
        # Restore attention zones
        for zone_id, zone_dict in state.get('attention_zones', {}).items():
            tracker.attention_zones[zone_id] = AttentionZone(**zone_dict)
        
        # Restore drift events
        for drift_dict in state.get('drift_events', []):
            drift = AttentionDrift(
                start_time=datetime.fromisoformat(drift_dict['start_time']),
                duration=drift_dict['duration'],
                drift_type=drift_dict['drift_type'],
                severity=drift_dict['severity'],
                recovery_time=drift_dict.get('recovery_time')
            )
            tracker.drift_events.append(drift)
        
        return tracker
