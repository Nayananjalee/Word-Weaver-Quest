/**
 * ParentTherapistDashboard - Clinical Report & Analytics Dashboard
 * 
 * Provides three types of AI-generated clinical reports:
 * 1. Therapist Report - Clinical metrics, IEP goals, WHO classification
 * 2. Parent Report - Simplified bilingual (Sinhala + English), encouraging
 * 3. Research Report - Statistical analysis with effect sizes & suggested tests
 * 
 * Research basis:
 * - Khosravi et al. (2022): Explainable AI for Education
 * - Holstein et al. (2021): Designing AI Dashboards for Teachers
 * - Lim et al. (2023): Game-Based Learning Analytics
 * - WHO (2021): World Report on Hearing Loss Classification
 */

import React, { useState, useCallback } from 'react';
import API_BASE_URL from '../config';

const ParentTherapistDashboard = ({ userId }) => {
  const [reportType, setReportType] = useState('therapist');
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [childName, setChildName] = useState('');
  const [childAge, setChildAge] = useState('72');
  const [language, setLanguage] = useState('english');
  const [error, setError] = useState(null);

  const generateReport = useCallback(async () => {
    if (!userId) return;
    setLoading(true);
    setError(null);
    setReport(null);

    try {
      const res = await fetch(`${API_BASE_URL}/generate-clinical-report`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          child_name: childName || 'Child',
          child_age_months: parseInt(childAge) || 72,
          report_type: reportType,
          language: language,
          report_period_days: 30
        })
      });

      if (!res.ok) {
        throw new Error(`Report generation failed: ${res.statusText}`);
      }

      const data = await res.json();
      setReport(data);
    } catch (err) {
      console.error('Report generation error:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [userId, reportType, childName, childAge, language]);

  // Report type configurations
  const reportTypes = [
    {
      key: 'therapist',
      label: '👨‍⚕️ Therapist',
      labelSi: 'චිකිත්සක',
      desc: 'Clinical metrics, IEP goals, WHO classification',
      color: 'blue'
    },
    {
      key: 'parent',
      label: '👨‍👩‍👧 Parent',
      labelSi: 'දෙමව්පියන්',
      desc: 'Simplified, encouraging, bilingual report',
      color: 'green'
    },
    {
      key: 'research',
      label: '📊 Research',
      labelSi: 'පර්යේෂණ',
      desc: 'Statistical analysis with effect sizes',
      color: 'purple'
    }
  ];

  // Section renderer for structured report data
  const ReportSection = ({ title, icon, children, color = 'blue' }) => (
    <div className={`bg-white/10 backdrop-blur rounded-2xl p-4 border border-${color}-400/20 mb-4`}>
      <h3 className={`text-lg font-bold text-${color}-300 mb-3 flex items-center gap-2`}>
        <span>{icon}</span> {title}
      </h3>
      {children}
    </div>
  );

  // Metric display
  const MetricRow = ({ label, value, unit = '', highlight = false }) => (
    <div className={`flex justify-between items-center py-1.5 px-3 rounded-lg ${highlight ? 'bg-white/10' : ''}`}>
      <span className="text-white/80 text-sm">{label}</span>
      <span className={`font-bold text-sm ${highlight ? 'text-yellow-300' : 'text-white'}`}>
        {value}{unit}
      </span>
    </div>
  );

  // Render therapist report
  const renderTherapistReport = (data) => (
    <div className="space-y-4">
      {/* Header */}
      <div className="bg-blue-500/20 backdrop-blur rounded-2xl p-6 border border-blue-400/30 text-center">
        <h2 className="text-xl font-bold text-white mb-1">Clinical Report</h2>
        <p className="text-blue-200 text-sm">Word-Weaver-Quest Speech Therapy Platform</p>
        {data.patient_profile && (
          <p className="text-white/60 text-xs mt-2">
            Patient: {data.patient_profile.name || 'N/A'} | Age: {data.patient_profile.age || 'N/A'}
          </p>
        )}
        <p className="text-white/60 text-xs">Report Period: {data.report_period || 'Last 30 days'}</p>
      </div>

      {/* Session Overview */}
      {data.session_overview && (
        <ReportSection title="Session Overview" icon="📊" color="blue">
          <MetricRow label="Total Sessions" value={data.session_overview.total_sessions ?? 0} highlight />
          <MetricRow label="Questions Attempted" value={data.session_overview.total_questions_attempted ?? 0} />
          <MetricRow label="Overall Accuracy" value={data.session_overview.overall_accuracy || '0%'} highlight />
          <MetricRow label="Sessions/Week" value={data.session_overview.sessions_per_week ?? 0} />
          <MetricRow label="Recommended" value={data.session_overview.recommended_sessions_per_week || '4-5/week'} />
          {data.session_overview.accuracy_interpretation && (
            <p className="text-white/60 text-xs px-3 mt-2 italic">{data.session_overview.accuracy_interpretation}</p>
          )}
        </ReportSection>
      )}

      {/* Hearing Classification */}
      {data.patient_profile?.who_classification && (
        <ReportSection title="Hearing Loss Assessment (WHO)" icon="👂" color="yellow">
          <MetricRow label="Severity" value={data.patient_profile.hearing_severity || 'N/A'} highlight />
          <MetricRow label="WHO Grade" value={data.patient_profile.who_classification.grade || 'N/A'} />
          <MetricRow label="Threshold Range" value={data.patient_profile.who_classification.threshold_range || 'N/A'} />
          <MetricRow label="Description" value={data.patient_profile.who_classification.description || 'N/A'} />
        </ReportSection>
      )}

      {/* Phoneme Analysis */}
      {data.phoneme_analysis && (
        <ReportSection title="Phoneme Analysis" icon="🗣️" color="purple">
          <MetricRow label="Phoneme Pairs Tracked" value={data.phoneme_analysis.total_phoneme_pairs_tracked ?? 0} />
          {data.phoneme_analysis.most_confused_pairs?.length > 0 && (
            <div className="mt-2 space-y-1">
              <p className="text-white/70 text-xs px-3">Most Confused Pairs:</p>
              {data.phoneme_analysis.most_confused_pairs.map((pair, i) => (
                <div key={i} className="flex justify-between bg-white/5 rounded-lg px-3 py-1.5">
                  <span className="text-white text-sm">{typeof pair === 'string' ? pair : JSON.stringify(pair)}</span>
                </div>
              ))}
            </div>
          )}
          {data.phoneme_analysis.clinical_notes && (
            <p className="text-white/60 text-xs px-3 mt-2 italic">{data.phoneme_analysis.clinical_notes}</p>
          )}
        </ReportSection>
      )}

      {/* Vocabulary Acquisition */}
      {data.vocabulary_acquisition && (
        <ReportSection title="Vocabulary Acquisition" icon="📚" color="cyan">
          <MetricRow label="Words in Deck" value={data.vocabulary_acquisition.total_words_in_deck ?? 0} highlight />
          <MetricRow label="Words Mastered" value={data.vocabulary_acquisition.words_mastered ?? 0} />
          <MetricRow label="Words Learning" value={data.vocabulary_acquisition.words_learning ?? 0} />
          <MetricRow label="Average Retention" value={data.vocabulary_acquisition.average_retention || '0%'} />
          <MetricRow label="Mastery Rate" value={data.vocabulary_acquisition.mastery_rate || '0%'} />
        </ReportSection>
      )}

      {/* Cognitive Load */}
      {data.cognitive_load_analysis && (
        <ReportSection title="Cognitive Load Analysis (CLT)" icon="🧠" color="blue">
          <MetricRow label="Intrinsic Load" value={data.cognitive_load_analysis.average_intrinsic_load ?? 0} highlight />
          <MetricRow label="Extraneous Load" value={data.cognitive_load_analysis.average_extraneous_load ?? 0} />
          <MetricRow label="Germane Load" value={data.cognitive_load_analysis.average_germane_load ?? 0} />
          <MetricRow label="Optimal Zone Time" value={data.cognitive_load_analysis.optimal_zone_time || '0%'} />
          {data.cognitive_load_analysis.cognitive_load_interpretation && (
            <p className="text-white/60 text-xs px-3 mt-2 italic">{data.cognitive_load_analysis.cognitive_load_interpretation}</p>
          )}
        </ReportSection>
      )}

      {/* Engagement & Attention */}
      {data.engagement_attention && (
        <ReportSection title="Engagement & Attention" icon="👁️" color="orange">
          <MetricRow label="Avg Engagement Score" value={data.engagement_attention.average_engagement_score ?? 0} highlight />
          <MetricRow label="Engagement Trend" value={data.engagement_attention.engagement_trend || 'stable'} />
          <MetricRow label="Focus Quality" value={data.engagement_attention.focus_quality || 'N/A'} />
          <MetricRow label="Dropout Risk" value={data.engagement_attention.dropout_risk || '0%'} />
        </ReportSection>
      )}

      {/* IEP Goals */}
      {data.iep_goals && data.iep_goals.length > 0 && (
        <ReportSection title="IEP Goals" icon="🎯" color="green">
          {data.iep_goals.map((goal, i) => (
            <div key={i} className="bg-white/5 rounded-xl p-3 mb-2">
              <div className="flex items-center justify-between mb-1">
                <span className="text-green-300 font-bold text-sm">{goal.area || `Goal ${i + 1}`}</span>
                <span className={`text-xs px-2 py-0.5 rounded-full ${
                  goal.status === 'on_track' ? 'bg-green-500/30 text-green-300' :
                  goal.status === 'needs_attention' ? 'bg-yellow-500/30 text-yellow-300' :
                  'bg-red-500/30 text-red-300'
                }`}>{(goal.status || 'pending').replace(/_/g, ' ')}</span>
              </div>
              <div className="text-xs text-white/70 space-y-0.5">
                <p><span className="text-white/50">Current:</span> {goal.current_level || 'N/A'}</p>
                <p><span className="text-white/50">Target:</span> {goal.target || 'N/A'}</p>
                <p><span className="text-white/50">Timeline:</span> {goal.timeline || 'N/A'}</p>
                <p><span className="text-white/50">Measure:</span> {goal.measurement || 'N/A'}</p>
              </div>
            </div>
          ))}
        </ReportSection>
      )}

      {/* Clinical Recommendations */}
      {data.clinical_recommendations && data.clinical_recommendations.length > 0 && (
        <ReportSection title="Clinical Recommendations" icon="💡" color="orange">
          {data.clinical_recommendations.map((rec, i) => (
            <div key={i} className="flex gap-2 items-start bg-white/5 rounded-lg p-2 mb-1">
              <span className={`text-xs font-bold px-1.5 py-0.5 rounded ${
                rec.priority === 'high' ? 'bg-red-500/30 text-red-300' :
                rec.priority === 'medium' ? 'bg-yellow-500/30 text-yellow-300' :
                'bg-blue-500/30 text-blue-300'
              }`}>{rec.priority || '—'}</span>
              <div>
                <p className="text-white/90 text-sm font-semibold">{rec.area || ''}</p>
                <p className="text-white/70 text-xs">{rec.recommendation_en || String(rec)}</p>
                {rec.recommendation_si && language === 'sinhala' && (
                  <p className="text-white/50 text-xs mt-0.5">{rec.recommendation_si}</p>
                )}
              </div>
            </div>
          ))}
        </ReportSection>
      )}
    </div>
  );

  // Render parent report
  const renderParentReport = (data) => {
    const summary = data.summary || {};
    const progress = data.progress || {};
    // tips can be {en:[...], si:[...]} or a plain array
    const tipsRaw = data.tips || {};
    const tipsArr = Array.isArray(tipsRaw)
      ? tipsRaw
      : (language === 'sinhala' ? (tipsRaw.si || tipsRaw.en || []) : (tipsRaw.en || []));

    return (
    <div className="space-y-4">
      {/* Friendly Header */}
      <div className="bg-green-500/20 backdrop-blur rounded-3xl p-6 border border-green-400/30 text-center">
        <div className="text-4xl mb-2">🌟</div>
        <h2 className="text-xl font-bold text-white mb-1">
          {language === 'sinhala' ? 'ඔබේ දරුවාගේ ප්‍රගතිය' : `${summary.child_name || 'Your Child'}'s Progress`}
        </h2>
        <p className="text-green-200 text-sm">
          {language === 'sinhala' ? 'වචන-වියන්නා ඉගෙනුම් වාර්තාව' : 'Word-Weaver-Quest Learning Report'}
        </p>
        <p className="text-white/50 text-xs mt-1">{summary.period || 'Last 30 days'}</p>
      </div>

      {/* Celebration message */}
      {(progress.celebration_en || progress.celebration_si) && (
        <div className="bg-yellow-500/20 backdrop-blur rounded-2xl p-4 border border-yellow-400/30 text-center">
          <div className="text-3xl mb-2">🎉</div>
          <p className="text-white font-semibold">
            {language === 'sinhala' ? (progress.celebration_si || progress.celebration_en) : (progress.celebration_en || progress.celebration_si)}
          </p>
        </div>
      )}

      {/* Simple Progress */}
      <div className="bg-white/10 backdrop-blur rounded-2xl p-4">
        <h3 className="text-white font-bold mb-3">📈 {language === 'sinhala' ? 'ප්‍රගති සාරාංශය' : 'Progress Summary'}</h3>
        <div className="space-y-3">
          <div className="flex justify-between text-white text-sm">
            <span>{language === 'sinhala' ? 'සම්පූර්ණ කළ පාඩම්' : 'Sessions Completed'}</span>
            <span className="font-bold">{summary.sessions_completed ?? 0}</span>
          </div>
          <div className="flex justify-between text-white text-sm">
            <span>{language === 'sinhala' ? 'ඉගෙන ගත් වචන' : 'Words Practiced'}</span>
            <span className="font-bold text-green-300">{summary.words_practiced ?? 0}</span>
          </div>
          <div className="flex justify-between text-white text-sm">
            <span>{language === 'sinhala' ? 'ශ්‍රේණිය' : 'Accuracy Stars'}</span>
            <span className="font-bold text-yellow-300">{'⭐'.repeat(Math.max(0, summary.accuracy_stars ?? 1))}</span>
          </div>
          {progress.accuracy_percentage !== undefined && (
            <div>
              <div className="flex justify-between text-white text-sm mb-1">
                <span>{language === 'sinhala' ? 'නිවැරදි අනුපාතය' : 'Accuracy Rate'}</span>
                <span className="font-bold text-yellow-300">{Math.round(progress.accuracy_percentage)}%</span>
              </div>
              <div className="w-full bg-gray-600 rounded-full h-3">
                <div
                  className="bg-gradient-to-r from-green-400 to-yellow-400 h-3 rounded-full"
                  style={{ width: `${Math.min(100, progress.accuracy_percentage)}%` }}
                />
              </div>
            </div>
          )}
          {progress.words_mastered !== undefined && (
            <div className="flex justify-between text-white text-sm">
              <span>{language === 'sinhala' ? 'ප්‍රගුණ කළ වචන' : 'Words Mastered'}</span>
              <span className="font-bold">{progress.words_mastered}</span>
            </div>
          )}
          {progress.engagement_level && (
            <div className="flex justify-between text-white text-sm">
              <span>{language === 'sinhala' ? 'නිරත මට්ටම' : 'Engagement Level'}</span>
              <span className="font-bold">{progress.engagement_level}</span>
            </div>
          )}
        </div>
      </div>

      {/* Message from system */}
      {(summary.message_en || summary.message_si) && (
        <div className="bg-pink-500/20 backdrop-blur rounded-2xl p-4 border border-pink-400/30">
          <div className="text-2xl mb-2 text-center">💪</div>
          <p className="text-white text-sm text-center">
            {language === 'sinhala' ? (summary.message_si || summary.message_en) : (summary.message_en || summary.message_si)}
          </p>
        </div>
      )}

      {/* Improvement areas */}
      {(progress.improvement_areas_en?.length > 0 || progress.improvement_areas_si?.length > 0) && (
        <div className="bg-white/10 backdrop-blur rounded-2xl p-4">
          <h3 className="text-white font-bold mb-3">💡 {language === 'sinhala' ? 'වැඩිදියුණු කළ හැකි ක්ෂේත්‍ර' : 'Areas to Improve'}</h3>
          {(language === 'sinhala' ? (progress.improvement_areas_si || progress.improvement_areas_en) : (progress.improvement_areas_en || [])).map((area, i) => (
            <div key={i} className="flex gap-2 items-start mb-2">
              <span className="text-yellow-400">•</span>
              <span className="text-white/90 text-sm">{area}</span>
            </div>
          ))}
        </div>
      )}

      {/* Tips */}
      {tipsArr.length > 0 && (
        <div className="bg-white/10 backdrop-blur rounded-2xl p-4">
          <h3 className="text-white font-bold mb-3">
            {language === 'sinhala' ? '💡 ගෙදර දී කළ හැකි දේ' : '💡 Things to do at Home'}
          </h3>
          {tipsArr.map((tip, i) => (
            <div key={i} className="flex gap-2 items-start mb-2">
              <span className="text-green-400 text-lg">✨</span>
              <span className="text-white/90 text-sm">{tip}</span>
            </div>
          ))}
        </div>
      )}
    </div>
    );
  };

  // Render research report
  const renderResearchReport = (data) => (
    <div className="space-y-4">
      {/* Research Header */}
      <div className="bg-purple-500/20 backdrop-blur rounded-2xl p-6 border border-purple-400/30">
        <h2 className="text-xl font-bold text-white mb-1">Research Analysis Report</h2>
        <p className="text-purple-200 text-sm">Word-Weaver-Quest: AI-Powered Sinhala Speech Therapy</p>
        <p className="text-white/60 text-xs mt-1">Generated for academic/research purposes</p>
      </div>

      {/* Descriptive Statistics */}
      {data.descriptive_stats && (
        <ReportSection title="Descriptive Statistics" icon="📊" color="purple">
          {Object.entries(data.descriptive_stats).map(([key, val]) => (
            <MetricRow key={key} label={key.replace(/_/g, ' ')} value={
              typeof val === 'number' ? val.toFixed(3) : String(val)
            } />
          ))}
        </ReportSection>
      )}

      {/* Effect Sizes */}
      {data.effect_sizes && (
        <ReportSection title="Effect Sizes (Cohen's d)" icon="📐" color="blue">
          {Object.entries(data.effect_sizes).map(([key, val]) => {
            const d = typeof val === 'number' ? val : 0;
            const magnitude = Math.abs(d) < 0.2 ? 'Negligible' : Math.abs(d) < 0.5 ? 'Small' : Math.abs(d) < 0.8 ? 'Medium' : 'Large';
            return (
              <div key={key} className="flex justify-between items-center py-1.5 px-3 rounded-lg bg-white/5 mb-1">
                <span className="text-white/80 text-sm">{key.replace(/_/g, ' ')}</span>
                <div className="flex items-center gap-2">
                  <span className="text-white font-mono text-sm">{d.toFixed(3)}</span>
                  <span className={`text-xs px-2 py-0.5 rounded ${
                    magnitude === 'Large' ? 'bg-green-500/30 text-green-300' :
                    magnitude === 'Medium' ? 'bg-yellow-500/30 text-yellow-300' :
                    'bg-gray-500/30 text-gray-300'
                  }`}>{magnitude}</span>
                </div>
              </div>
            );
          })}
        </ReportSection>
      )}

      {/* Suggested Statistical Tests */}
      {data.suggested_tests && data.suggested_tests.length > 0 && (
        <ReportSection title="Suggested Statistical Tests" icon="🧪" color="green">
          {data.suggested_tests.map((test, i) => (
            <div key={i} className="bg-white/5 rounded-lg p-2 mb-1">
              <span className="text-white text-sm">{test}</span>
            </div>
          ))}
        </ReportSection>
      )}

      {/* Research Variables */}
      {data.variables && (
        <ReportSection title="Study Variables" icon="🔬" color="orange">
          {data.variables.independent && (
            <div className="mb-2">
              <p className="text-orange-300 text-xs font-bold mb-1">Independent Variables:</p>
              {data.variables.independent.map((v, i) => (
                <span key={i} className="inline-block bg-orange-500/20 text-orange-200 text-xs px-2 py-1 rounded mr-1 mb-1">{v}</span>
              ))}
            </div>
          )}
          {data.variables.dependent && (
            <div>
              <p className="text-blue-300 text-xs font-bold mb-1">Dependent Variables:</p>
              {data.variables.dependent.map((v, i) => (
                <span key={i} className="inline-block bg-blue-500/20 text-blue-200 text-xs px-2 py-1 rounded mr-1 mb-1">{v}</span>
              ))}
            </div>
          )}
        </ReportSection>
      )}
    </div>
  );

  return (
    <div className="p-4 space-y-4 max-w-4xl mx-auto">
      {/* Dashboard Header */}
      <div className="text-center">
        <h2 className="text-2xl font-bold text-white mb-1">📋 වාර්තා උත්පාදකය</h2>
        <p className="text-white/80 text-sm">Clinical Report Generator</p>
      </div>

      {/* Report Type Selector */}
      <div className="flex gap-2 justify-center">
        {reportTypes.map(rt => (
          <button
            key={rt.key}
            onClick={() => { setReportType(rt.key); setReport(null); }}
            className={`px-4 py-2 rounded-full text-sm font-bold transition-all ${
              reportType === rt.key
                ? `bg-gradient-to-r from-${rt.color}-400 to-${rt.color}-500 text-white shadow-lg scale-105`
                : 'bg-white/20 text-white hover:bg-white/30'
            }`}
          >
            {rt.label}
          </button>
        ))}
      </div>

      {/* Configuration */}
      <div className="bg-white/10 backdrop-blur rounded-2xl p-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div>
            <label className="text-white/70 text-xs font-bold block mb-1">
              Child Name / දරුවාගේ නම
            </label>
            <input
              type="text"
              value={childName}
              onChange={e => setChildName(e.target.value)}
              placeholder="Enter name..."
              className="w-full px-3 py-2 rounded-xl bg-white/10 border border-white/20 text-white text-sm placeholder-white/40 focus:outline-none focus:border-white/50"
            />
          </div>
          <div>
            <label className="text-white/70 text-xs font-bold block mb-1">
              Age (months) / වයස (මාස)
            </label>
            <input
              type="number"
              value={childAge}
              onChange={e => setChildAge(e.target.value)}
              className="w-full px-3 py-2 rounded-xl bg-white/10 border border-white/20 text-white text-sm focus:outline-none focus:border-white/50"
            />
          </div>
          <div>
            <label className="text-white/70 text-xs font-bold block mb-1">
              Language / භාෂාව
            </label>
            <select
              value={language}
              onChange={e => setLanguage(e.target.value)}
              className="w-full px-3 py-2 rounded-xl bg-white/10 border border-white/20 text-white text-sm focus:outline-none focus:border-white/50"
            >
              <option value="english" className="text-black">English</option>
              <option value="sinhala" className="text-black">සිංහල</option>
              <option value="bilingual" className="text-black">Both / දෙකම</option>
            </select>
          </div>
        </div>

        <button
          onClick={generateReport}
          disabled={loading}
          className="w-full mt-4 py-3 bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 text-white font-bold rounded-full text-lg shadow-lg transform hover:scale-105 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <span className="animate-spin">⏳</span>
              Generating Report...
            </span>
          ) : (
            `📄 Generate ${reportTypes.find(r => r.key === reportType)?.label || ''} Report`
          )}
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-500/20 border border-red-400/30 rounded-2xl p-4 text-center">
          <span className="text-red-300 text-sm">❌ {error}</span>
        </div>
      )}

      {/* Report Output */}
      {report && (
        <div className="mt-4">
          {reportType === 'therapist' && renderTherapistReport(report)}
          {reportType === 'parent' && renderParentReport(report)}
          {reportType === 'research' && renderResearchReport(report)}
          
          {/* Print/Export Button */}
          <div className="mt-4 text-center">
            <button
              onClick={() => window.print()}
              className="px-6 py-2 bg-white/20 text-white rounded-full font-bold hover:bg-white/30 transition-all"
            >
              🖨️ Print Report
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default ParentTherapistDashboard;
