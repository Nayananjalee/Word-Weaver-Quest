"""
Engagement Visualization and Reporting

Generates charts, dashboards, and reports for engagement analysis.

Visualizations:
1. Real-time engagement gauge (0-100 score)
2. Engagement timeline (score over sessions)
3. Component breakdown (emotion, gesture, response time, attention)
4. Risk alert dashboard
5. Intervention history timeline

Output Format:
- JSON for frontend (Chart.js, D3.js compatible)
- PDF reports for therapists
- Sinhala/English text summaries for parents
"""

import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass


class EngagementVisualizationGenerator:
    """Generate visualization data for engagement monitoring."""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
    
    
    def generate_realtime_gauge(
        self,
        current_score: float,
        trend: str,
        risk_level: str
    ) -> Dict:
        """
        Generate real-time engagement gauge data.
        
        Returns JSON for gauge chart:
        - Score: 0-100
        - Color: green (70-100), yellow (40-70), red (0-40)
        - Trend indicator: ↑ ↓ →
        """
        # Determine color zones
        color = "green" if current_score >= 70 else "yellow" if current_score >= 40 else "red"
        
        # Trend arrow
        trend_arrow = "↑" if trend == "increasing" else "↓" if trend == "declining" else "→"
        
        return {
            'type': 'gauge',
            'score': current_score,
            'max_score': 100,
            'color': color,
            'trend_arrow': trend_arrow,
            'risk_level': risk_level,
            'zones': [
                {'min': 0, 'max': 40, 'color': '#FF4444', 'label': 'Low'},
                {'min': 40, 'max': 70, 'color': '#FFBB33', 'label': 'Medium'},
                {'min': 70, 'max': 100, 'color': '#00C851', 'label': 'High'}
            ]
        }
    
    
    def generate_timeline_chart(
        self,
        engagement_history: List[float],
        timestamps: Optional[List[datetime]] = None
    ) -> Dict:
        """
        Generate engagement timeline chart data.
        
        Returns JSON for line chart showing engagement over time.
        """
        if not timestamps:
            # Generate mock timestamps
            timestamps = [
                datetime.now() - timedelta(hours=i)
                for i in range(len(engagement_history)-1, -1, -1)
            ]
        
        # Calculate moving average for trend line
        window = 3
        moving_avg = []
        for i in range(len(engagement_history)):
            start_idx = max(0, i - window + 1)
            avg = np.mean(engagement_history[start_idx:i+1])
            moving_avg.append(round(avg, 2))
        
        return {
            'type': 'line',
            'labels': [ts.strftime('%Y-%m-%d %H:%M') for ts in timestamps],
            'datasets': [
                {
                    'label': 'Engagement Score',
                    'data': engagement_history,
                    'borderColor': '#4285F4',
                    'backgroundColor': 'rgba(66, 133, 244, 0.1)',
                    'tension': 0.4
                },
                {
                    'label': 'Trend',
                    'data': moving_avg,
                    'borderColor': '#FF6384',
                    'borderDash': [5, 5],
                    'fill': False
                }
            ],
            'threshold_lines': [
                {'value': 70, 'label': 'Optimal', 'color': '#00C851'},
                {'value': 40, 'label': 'At-Risk', 'color': '#FF4444'}
            ]
        }
    
    
    def generate_component_breakdown(
        self,
        component_scores: Dict[str, float]
    ) -> Dict:
        """
        Generate component breakdown chart (radar/spider chart).
        
        Shows: emotion, gesture, response_time, attention
        """
        return {
            'type': 'radar',
            'labels': [
                'Emotion',
                'Gesture Quality',
                'Response Speed',
                'Attention'
            ],
            'datasets': [{
                'label': 'Current Session',
                'data': [
                    component_scores.get('emotion', 0),
                    component_scores.get('gesture', 0),
                    component_scores.get('response_time', 0),
                    component_scores.get('attention', 0)
                ],
                'backgroundColor': 'rgba(66, 133, 244, 0.2)',
                'borderColor': '#4285F4',
                'pointBackgroundColor': '#4285F4'
            }],
            'max_value': 100
        }
    
    
    def generate_risk_dashboard(
        self,
        dropout_risk: float,
        risk_factors: List[str],
        consecutive_low_sessions: int
    ) -> Dict:
        """
        Generate risk alert dashboard data.
        
        Shows:
        - Dropout risk score
        - Risk factors
        - Alert status
        """
        # Map risk factors to user-friendly labels
        factor_labels = {
            'low_average_engagement': 'Low average engagement',
            'declining_trend': 'Engagement is declining',
            'consecutive_low_sessions': f'{consecutive_low_sessions} consecutive low sessions',
            'long_absence': 'Long absence from sessions',
            'no_improvement': 'No improvement over time',
            'insufficient_data': 'Not enough data yet'
        }
        
        formatted_factors = [
            factor_labels.get(factor, factor)
            for factor in risk_factors
        ]
        
        # Determine alert level
        if dropout_risk >= 80:
            alert_level = "critical"
            alert_message = "Immediate intervention required! 🚨"
        elif dropout_risk >= 60:
            alert_level = "high"
            alert_message = "High dropout risk - schedule consultation"
        elif dropout_risk >= 30:
            alert_level = "medium"
            alert_message = "Monitor closely for engagement issues"
        else:
            alert_level = "low"
            alert_message = "Child is engaged and progressing well ✅"
        
        return {
            'type': 'risk_dashboard',
            'dropout_risk': dropout_risk,
            'alert_level': alert_level,
            'alert_message': alert_message,
            'risk_factors': formatted_factors,
            'recommendations': self._generate_risk_recommendations(risk_factors)
        }
    
    
    def generate_intervention_timeline(
        self,
        interventions: List[Dict]
    ) -> Dict:
        """
        Generate intervention history timeline.
        
        Shows when interventions were triggered and their types.
        """
        # Group interventions by type
        intervention_counts = {}
        for intervention in interventions:
            itype = intervention.get('intervention_type', 'unknown')
            intervention_counts[itype] = intervention_counts.get(itype, 0) + 1
        
        return {
            'type': 'timeline',
            'interventions': [
                {
                    'timestamp': intervention.get('timestamp'),
                    'type': intervention.get('intervention_type'),
                    'engagement_before': intervention.get('engagement_score'),
                    'icon': self._get_intervention_icon(intervention.get('intervention_type'))
                }
                for intervention in interventions
            ],
            'summary': intervention_counts,
            'total_interventions': len(interventions)
        }
    
    
    def generate_progress_summary(
        self,
        engagement_history: List[float],
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """
        Generate progress summary statistics.
        
        Returns:
        - Average engagement
        - Improvement rate
        - Best/worst sessions
        - Consistency score
        """
        if not engagement_history:
            return {
                'average_engagement': 0.0,
                'improvement_rate': 0.0,
                'best_session': 0.0,
                'worst_session': 0.0,
                'consistency_score': 0.0,
                'total_sessions': 0
            }
        
        avg_engagement = np.mean(engagement_history)
        best_session = np.max(engagement_history)
        worst_session = np.min(engagement_history)
        
        # Compute improvement rate (slope of linear regression)
        if len(engagement_history) >= 5:
            x = np.arange(len(engagement_history))
            slope = np.cov(x, engagement_history)[0, 1] / np.var(x) if np.var(x) > 0 else 0
            improvement_rate = slope
        else:
            improvement_rate = 0.0
        
        # Consistency score (inverse of coefficient of variation)
        std_dev = np.std(engagement_history)
        consistency_score = 100 * (1 - min(std_dev / avg_engagement, 1.0)) if avg_engagement > 0 else 0
        
        # Session count and duration
        total_sessions = len(engagement_history)
        days_elapsed = (end_date - start_date).days + 1
        sessions_per_week = (total_sessions / days_elapsed) * 7 if days_elapsed > 0 else 0
        
        return {
            'average_engagement': round(avg_engagement, 2),
            'improvement_rate': round(improvement_rate, 3),
            'best_session': round(best_session, 2),
            'worst_session': round(worst_session, 2),
            'consistency_score': round(consistency_score, 2),
            'total_sessions': total_sessions,
            'sessions_per_week': round(sessions_per_week, 2),
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d')
        }
    
    
    def _generate_risk_recommendations(self, risk_factors: List[str]) -> List[str]:
        """Generate recommendations based on risk factors."""
        recommendations = []
        
        if 'low_average_engagement' in risk_factors:
            recommendations.append("Schedule 1-on-1 session to understand child's challenges")
        
        if 'declining_trend' in risk_factors:
            recommendations.append("Reduce difficulty level to rebuild confidence")
        
        if 'consecutive_low_sessions' in risk_factors:
            recommendations.append("Take a break - suggest fun activity instead of therapy")
        
        if 'long_absence' in risk_factors:
            recommendations.append("Welcome back session with easier tasks")
        
        if 'no_improvement' in risk_factors:
            recommendations.append("Consult speech therapist to adjust therapy plan")
        
        if not recommendations:
            recommendations.append("Continue current therapy approach - child is progressing well")
        
        return recommendations
    
    
    def _get_intervention_icon(self, intervention_type: str) -> str:
        """Get emoji icon for intervention type."""
        icons = {
            'break': '☕',
            'reward': '⭐',
            'difficulty_adjust': '🎯',
            'encouragement': '💪'
        }
        return icons.get(intervention_type, '📌')


class EngagementReportGenerator:
    """Generate text and PDF reports for therapists and parents."""
    
    def __init__(self, user_id: str, child_name: str):
        self.user_id = user_id
        self.child_name = child_name
    
    
    def generate_therapist_report(
        self,
        engagement_stats: Dict,
        risk_analysis: Dict,
        recommendations: List[str],
        language: str = 'english'
    ) -> str:
        """
        Generate comprehensive report for speech therapist.
        
        Includes:
        - Executive summary
        - Engagement trends
        - Risk assessment
        - Intervention history
        - Clinical recommendations
        """
        if language == 'sinhala':
            return self._generate_sinhala_therapist_report(
                engagement_stats, risk_analysis, recommendations
            )
        
        report = f"""
======================================================================
ENGAGEMENT ANALYSIS REPORT - SPEECH THERAPIST
======================================================================

Child: {self.child_name}
Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

EXECUTIVE SUMMARY
----------------------------------------------------------------------
Average Engagement: {engagement_stats.get('average_engagement', 0)}/100
Trend: {engagement_stats.get('improvement_rate', 0) > 0 and 'Improving' or 'Declining'}
Dropout Risk: {risk_analysis.get('dropout_risk', 0)}/100 ({risk_analysis.get('alert_level', 'unknown').upper()})

ENGAGEMENT STATISTICS
----------------------------------------------------------------------
Total Sessions: {engagement_stats.get('total_sessions', 0)}
Sessions per Week: {engagement_stats.get('sessions_per_week', 0)}
Best Session Score: {engagement_stats.get('best_session', 0)}/100
Worst Session Score: {engagement_stats.get('worst_session', 0)}/100
Consistency Score: {engagement_stats.get('consistency_score', 0)}/100

MULTIMODAL ANALYSIS
----------------------------------------------------------------------
Emotion Component: Strong indicator of emotional state during therapy
Gesture Component: Measures physical engagement and motor skills
Response Time: Cognitive processing speed and task difficulty match
Attention Component: Eye contact and focus on learning materials

RISK ASSESSMENT
----------------------------------------------------------------------
Dropout Risk Level: {risk_analysis.get('alert_level', 'unknown').upper()}
Risk Score: {risk_analysis.get('dropout_risk', 0)}/100

Identified Risk Factors:
"""
        for i, factor in enumerate(risk_analysis.get('risk_factors', []), 1):
            report += f"{i}. {factor}\n"
        
        report += f"""

CLINICAL RECOMMENDATIONS
----------------------------------------------------------------------
"""
        for i, rec in enumerate(recommendations, 1):
            report += f"{i}. {rec}\n"
        
        report += f"""

INTERVENTION SUMMARY
----------------------------------------------------------------------
Total Interventions: {risk_analysis.get('total_interventions', 0)}
Most Common Type: {risk_analysis.get('most_common_intervention', 'None')}

NEXT STEPS
----------------------------------------------------------------------
1. Review engagement patterns with multidisciplinary team
2. Adjust therapy plan based on component analysis
3. Schedule parent consultation if dropout risk is high
4. Continue monitoring for 2 more weeks before major changes

======================================================================
Report generated by Multimodal Engagement Analysis System
Medical basis: Flow State Theory (Csikszentmihalyi, 1990)
======================================================================
"""
        return report
    
    
    def generate_parent_report(
        self,
        engagement_stats: Dict,
        language: str = 'sinhala'
    ) -> str:
        """
        Generate simplified report for parents.
        
        Focus on:
        - How child is doing
        - Positive highlights
        - Areas needing support
        - Home practice suggestions
        """
        if language == 'sinhala':
            return self._generate_sinhala_parent_report(engagement_stats)
        
        avg_score = engagement_stats.get('average_engagement', 0)
        
        # Determine overall status
        if avg_score >= 70:
            status = "Excellent"
            emoji = "🌟"
            message = "Your child is highly engaged and making great progress!"
        elif avg_score >= 50:
            status = "Good"
            emoji = "😊"
            message = "Your child is doing well. Keep encouraging them!"
        else:
            status = "Needs Support"
            emoji = "💪"
            message = "Your child needs extra encouragement. Let's work together!"
        
        report = f"""
======================================================================
YOUR CHILD'S PROGRESS REPORT {emoji}
======================================================================

Child: {self.child_name}
Date: {datetime.now().strftime('%Y-%m-%d')}

OVERALL STATUS: {status}
----------------------------------------------------------------------
{message}

ENGAGEMENT SCORE: {avg_score}/100
----------------------------------------------------------------------
Your child's average engagement during therapy sessions.

Total Sessions Completed: {engagement_stats.get('total_sessions', 0)}
Best Session: {engagement_stats.get('best_session', 0)}/100

WHAT YOUR CHILD IS DOING WELL
----------------------------------------------------------------------
✅ Attending regular therapy sessions
✅ Showing interest in learning activities
✅ Improving communication skills

HOW YOU CAN HELP AT HOME
----------------------------------------------------------------------
1. Practice 10-15 minutes daily with fun activities
2. Praise effort, not just correct answers
3. Keep sessions short and enjoyable
4. Celebrate small victories

NEXT SESSION
----------------------------------------------------------------------
Continue with current therapy plan.
Your therapist will contact you if any concerns arise.

======================================================================
Questions? Contact your speech therapist.
======================================================================
"""
        return report
    
    
    def _generate_sinhala_therapist_report(
        self,
        engagement_stats: Dict,
        risk_analysis: Dict,
        recommendations: List[str]
    ) -> str:
        """Generate therapist report in Sinhala."""
        return f"""
======================================================================
නිරතුරු විශ්ලේෂණ වාර්තාව - කථන චිකිත්සක
======================================================================

දරුවා: {self.child_name}
වාර්තා දිනය: {datetime.now().strftime('%Y-%m-%d %H:%M')}

සාරාංශය
----------------------------------------------------------------------
සාමාන්‍ය නිරතුරුව: {engagement_stats.get('average_engagement', 0)}/100
අවදානම් මට්ටම: {risk_analysis.get('alert_level', 'unknown')}

සංඛ්‍යාලේඛන
----------------------------------------------------------------------
සැසි ගණන: {engagement_stats.get('total_sessions', 0)}
හොඳම සැසිය: {engagement_stats.get('best_session', 0)}/100

නිර්දේශ
----------------------------------------------------------------------
"""
        for i, rec in enumerate(recommendations, 1):
            report += f"{i}. {rec}\n"
        
        return report
    
    
    def _generate_sinhala_parent_report(self, engagement_stats: Dict) -> str:
        """Generate parent report in Sinhala."""
        avg_score = engagement_stats.get('average_engagement', 0)
        
        if avg_score >= 70:
            status = "විශිෂ්ටයි"
            emoji = "🌟"
        elif avg_score >= 50:
            status = "හොඳයි"
            emoji = "😊"
        else:
            status = "සහය අවශ්‍යයි"
            emoji = "💪"
        
        return f"""
======================================================================
ඔබේ දරුවාගේ ප්‍රගති වාර්තාව {emoji}
======================================================================

දරුවා: {self.child_name}
දිනය: {datetime.now().strftime('%Y-%m-%d')}

සමස්ත තත්ත්වය: {status}
----------------------------------------------------------------------

නිරතුරු ලකුණු: {avg_score}/100

සම්පූර්ණ කළ සැසි: {engagement_stats.get('total_sessions', 0)}

ගෙදර උදවු කරන ආකාරය
----------------------------------------------------------------------
1. දිනපතා මිනිත්තු 10-15 ක් පුහුණු වන්න
2. උත්සාහය අගය කරන්න
3. සැසි කෙටි හා විනෝදජනක කරන්න
4. කුඩා ජයග්‍රහණ සමරන්න

======================================================================
ප්‍රශ්න? ඔබේ කථන චිකිත්සක අමතන්න.
======================================================================
"""
        return report
    
    
    def generate_json_report(
        self,
        engagement_stats: Dict,
        risk_analysis: Dict,
        visualizations: Dict
    ) -> Dict:
        """
        Generate structured JSON report for API consumption.
        
        Used by frontend dashboard to display all metrics.
        """
        return {
            'report_metadata': {
                'user_id': self.user_id,
                'child_name': self.child_name,
                'generated_at': datetime.now().isoformat(),
                'report_type': 'engagement_analysis'
            },
            'engagement_statistics': engagement_stats,
            'risk_analysis': risk_analysis,
            'visualizations': visualizations,
            'recommendations': self._generate_risk_recommendations(
                risk_analysis.get('risk_factors', [])
            )
        }
    
    
    def _generate_risk_recommendations(self, risk_factors: List[str]) -> List[str]:
        """Same as visualization generator."""
        recommendations = []
        
        if 'low_average_engagement' in risk_factors:
            recommendations.append("Schedule 1-on-1 session to understand child's challenges")
        
        if 'declining_trend' in risk_factors:
            recommendations.append("Reduce difficulty level to rebuild confidence")
        
        if 'consecutive_low_sessions' in risk_factors:
            recommendations.append("Take a break - suggest fun activity instead of therapy")
        
        if 'long_absence' in risk_factors:
            recommendations.append("Welcome back session with easier tasks")
        
        if 'no_improvement' in risk_factors:
            recommendations.append("Consult speech therapist to adjust therapy plan")
        
        if not recommendations:
            recommendations.append("Continue current therapy approach - child is progressing well")
        
        return recommendations
