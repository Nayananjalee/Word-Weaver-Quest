/**
 * CognitiveLoadIndicator - Real-Time Cognitive Load Monitor Component
 * 
 * Displays real-time cognitive load estimation based on Sweller's Cognitive Load Theory.
 * Shows intrinsic, extraneous, and germane load with zone classification.
 * Designed as a compact overlay for the game view.
 * 
 * Research basis:
 * - Sweller (2020): Cognitive Load Theory and Educational Technology
 * - Klepsch & Seufert (2020): Understanding Instructional Design Effects on CLT
 * - Chen et al. (2023): Adaptive Learning Systems for Special Needs
 * - Kalyuga (2021): Expertise Reversal Effect and CLT
 */

import React, { useState, useEffect, useCallback } from 'react';
import API_BASE_URL from '../config';

const CognitiveLoadIndicator = ({ userId, compact = false }) => {
  const [loadReport, setLoadReport] = useState(null);
  const [timeline, setTimeline] = useState([]);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState(false);

  const fetchDashboard = useCallback(async () => {
    if (!userId) return;
    try {
      const res = await fetch(`${API_BASE_URL}/cognitive-load/dashboard/${userId}`);
      const data = await res.json();
      if (data) {
        // Backend returns data.report with average_loads.{intrinsic,extraneous,germane,total}
        // Normalise into flat shape the UI expects
        const raw = data.report || {};
        const loads = raw.average_loads || {};
        const total = loads.total || 0;
        const derivedZone =
          total > 0.85 ? 'overload' :
          total > 0.70 ? 'high' :
          total > 0.30 ? 'optimal' : 'underload';
        setLoadReport({
          zone: raw.current_zone || derivedZone,
          total_load: total,
          intrinsic_load: loads.intrinsic || 0,
          extraneous_load: loads.extraneous || 0,
          germane_load: loads.germane || 0,
          difficulty_adjustment: raw.difficulty_adjustment || 'maintain',
          recommendations: raw.recommendations || []
        });
        setTimeline(data.timeline || []);
      }
    } catch (err) {
      console.error('Cognitive load fetch error:', err);
    } finally {
      setLoading(false);
    }
  }, [userId]);

  useEffect(() => {
    fetchDashboard();
    const interval = setInterval(fetchDashboard, 10000); // Update every 10s
    return () => clearInterval(interval);
  }, [fetchDashboard]);

  // Zone colors and labels
  const zoneConfig = {
    underload: { color: 'from-blue-400 to-blue-500', text: 'Too Easy', emoji: '😴', border: 'border-blue-400', bg: 'bg-blue-500/20' },
    optimal: { color: 'from-green-400 to-emerald-500', text: 'Optimal', emoji: '🎯', border: 'border-green-400', bg: 'bg-green-500/20' },
    high: { color: 'from-yellow-400 to-orange-500', text: 'Challenging', emoji: '🤔', border: 'border-yellow-400', bg: 'bg-yellow-500/20' },
    overload: { color: 'from-red-400 to-red-600', text: 'Overloaded', emoji: '😰', border: 'border-red-400', bg: 'bg-red-500/20' },
  };

  // Load bar component
  const LoadBar = ({ label, value, color, icon }) => {
    const pct = Math.round((value || 0) * 100);
    return (
      <div className="space-y-0.5">
        <div className="flex justify-between items-center">
          <span className="text-white/80 text-xs font-medium">{icon} {label}</span>
          <span className="text-white text-xs font-bold">{pct}%</span>
        </div>
        <div className="w-full bg-gray-700/50 rounded-full h-2">
          <div
            className={`${color} h-2 rounded-full transition-all duration-700`}
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>
    );
  };

  if (loading && !loadReport) {
    return compact ? null : (
      <div className="flex items-center justify-center p-4">
        <div className="animate-spin text-2xl">🧠</div>
      </div>
    );
  }

  const zone = loadReport?.zone || 'optimal';
  const config = zoneConfig[zone] || zoneConfig.optimal;
  const totalLoad = loadReport?.total_load || 0;

  // Compact mode - small inline indicator for game view
  if (compact) {
    return (
      <div
        onClick={() => setExpanded(!expanded)}
        className={`cursor-pointer transition-all duration-300 ${expanded ? 'w-64' : 'w-auto'}`}
      >
        {/* Compact badge */}
        <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full ${config.bg} backdrop-blur border ${config.border} shadow-lg`}>
          <span className="text-sm">{config.emoji}</span>
          <span className="text-white text-xs font-bold">{config.text}</span>
          <div className="w-16 bg-gray-700/50 rounded-full h-1.5">
            <div
              className={`bg-gradient-to-r ${config.color} h-1.5 rounded-full transition-all duration-700`}
              style={{ width: `${Math.round(totalLoad * 100)}%` }}
            />
          </div>
        </div>

        {/* Expanded details */}
        {expanded && (
          <div className={`mt-2 p-3 rounded-2xl ${config.bg} backdrop-blur border ${config.border} shadow-xl`}>
            <LoadBar label="Intrinsic" value={loadReport?.intrinsic_load} color="bg-blue-400" icon="🔤" />
            <LoadBar label="Extraneous" value={loadReport?.extraneous_load} color="bg-red-400" icon="🔧" />
            <LoadBar label="Germane" value={loadReport?.germane_load} color="bg-green-400" icon="🌱" />
            
            {loadReport?.difficulty_adjustment && (
              <div className="mt-2 text-center">
                <span className={`text-xs font-bold ${
                  loadReport.difficulty_adjustment === 'decrease' ? 'text-red-300' :
                  loadReport.difficulty_adjustment === 'increase' ? 'text-blue-300' :
                  'text-green-300'
                }`}>
                  {loadReport.difficulty_adjustment === 'decrease' ? '⬇️ Reduce difficulty' :
                   loadReport.difficulty_adjustment === 'increase' ? '⬆️ Increase difficulty' :
                   '✅ Difficulty OK'}
                </span>
              </div>
            )}
          </div>
        )}
      </div>
    );
  }

  // Full dashboard mode
  return (
    <div className="p-4 space-y-4">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-2xl font-bold text-white mb-1">🧠 සංජානනීය බර නිරීක්ෂණය</h2>
        <p className="text-white/80 text-sm">Cognitive Load Monitor (Sweller's CLT)</p>
      </div>

      {/* Current Zone */}
      <div className={`p-6 rounded-3xl ${config.bg} backdrop-blur-lg border-2 ${config.border} shadow-2xl text-center`}>
        <div className="text-4xl mb-2">{config.emoji}</div>
        <div className="text-2xl font-bold text-white mb-1">{config.text}</div>
        <div className="text-white/70 text-sm mb-4">
          {zone === 'underload' && 'Child can handle more challenge / දරුවාට තව අභියෝගයක් දිය හැක'}
          {zone === 'optimal' && 'Perfect learning zone! / පරිපූර්ණ ඉගෙනුම් කලාපය!'}
          {zone === 'high' && 'Getting difficult, monitor closely / අමාරු වෙනවා, මොනිටර් කරන්න'}
          {zone === 'overload' && 'Too much! Reduce difficulty / බොහොම වැඩියි! අමාරුකම අඩු කරන්න'}
        </div>
        
        {/* Total load gauge */}
        <div className="relative w-32 h-32 mx-auto">
          <svg viewBox="0 0 100 100" className="transform -rotate-90">
            <circle cx="50" cy="50" r="40" fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="10" />
            <circle
              cx="50" cy="50" r="40" fill="none"
              stroke="url(#loadGradient)" strokeWidth="10"
              strokeDasharray={`${totalLoad * 251.2} 251.2`}
              strokeLinecap="round"
              className="transition-all duration-700"
            />
            <defs>
              <linearGradient id="loadGradient">
                <stop offset="0%" stopColor="#4ade80" />
                <stop offset="50%" stopColor="#facc15" />
                <stop offset="100%" stopColor="#ef4444" />
              </linearGradient>
            </defs>
          </svg>
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-2xl font-bold text-white">{Math.round(totalLoad * 100)}%</span>
          </div>
        </div>
      </div>

      {/* Three Load Types */}
      <div className="grid grid-cols-3 gap-3">
        <div className="bg-blue-500/20 backdrop-blur rounded-2xl p-4 border border-blue-400/30">
          <div className="text-center">
            <div className="text-2xl mb-1">🔤</div>
            <div className="text-white font-bold text-lg">
              {Math.round((loadReport?.intrinsic_load || 0) * 100)}%
            </div>
            <div className="text-blue-200 text-xs">Intrinsic</div>
            <div className="text-blue-300/70 text-[10px]">Task Difficulty</div>
          </div>
        </div>
        <div className="bg-red-500/20 backdrop-blur rounded-2xl p-4 border border-red-400/30">
          <div className="text-center">
            <div className="text-2xl mb-1">🔧</div>
            <div className="text-white font-bold text-lg">
              {Math.round((loadReport?.extraneous_load || 0) * 100)}%
            </div>
            <div className="text-red-200 text-xs">Extraneous</div>
            <div className="text-red-300/70 text-[10px]">Interface Load</div>
          </div>
        </div>
        <div className="bg-green-500/20 backdrop-blur rounded-2xl p-4 border border-green-400/30">
          <div className="text-center">
            <div className="text-2xl mb-1">🌱</div>
            <div className="text-white font-bold text-lg">
              {Math.round((loadReport?.germane_load || 0) * 100)}%
            </div>
            <div className="text-green-200 text-xs">Germane</div>
            <div className="text-green-300/70 text-[10px]">Learning Load</div>
          </div>
        </div>
      </div>

      {/* Load Timeline */}
      {timeline.length > 0 && (
        <div className="bg-white/10 backdrop-blur rounded-2xl p-4">
          <h3 className="text-white font-bold mb-3">📈 Load Timeline</h3>
          <div className="flex items-end gap-0.5 h-24">
            {timeline.slice(-30).map((point, i) => {
              const total = point.total_load || 0;
              const barColor = total > 0.85 ? 'bg-red-400' : total > 0.7 ? 'bg-yellow-400' : total > 0.3 ? 'bg-green-400' : 'bg-blue-400';
              return (
                <div
                  key={i}
                  className={`flex-1 ${barColor} rounded-t transition-all duration-300 hover:opacity-80`}
                  style={{ height: `${total * 100}%`, minWidth: '3px' }}
                  title={`Load: ${Math.round(total * 100)}%`}
                />
              );
            })}
          </div>
          <div className="flex justify-between text-white/50 text-[10px] mt-1">
            <span>Earlier</span>
            <span>Now</span>
          </div>
        </div>
      )}

      {/* Difficulty Recommendation */}
      {loadReport?.difficulty_adjustment && (
        <div className={`p-4 rounded-2xl text-center ${
          loadReport.difficulty_adjustment === 'decrease'
            ? 'bg-red-500/20 border border-red-400/30'
            : loadReport.difficulty_adjustment === 'increase'
            ? 'bg-blue-500/20 border border-blue-400/30'
            : 'bg-green-500/20 border border-green-400/30'
        }`}>
          <span className="text-white font-bold">
            {loadReport.difficulty_adjustment === 'decrease' && '⬇️ Recommendation: Reduce difficulty to prevent cognitive overload'}
            {loadReport.difficulty_adjustment === 'increase' && '⬆️ Recommendation: Increase difficulty for optimal learning'}
            {loadReport.difficulty_adjustment === 'maintain' && '✅ Current difficulty level is optimal'}
          </span>
        </div>
      )}
    </div>
  );
};

export default CognitiveLoadIndicator;
