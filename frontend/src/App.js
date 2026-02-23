import React, { useState, useEffect } from 'react';
import API_BASE_URL from './config';
import SentenceBySentenceStory from './components/SentenceBySentenceStory';
import RewardDashboard from './components/RewardDashboard';
import EngagementDashboard from './components/EngagementDashboard';
import AttentionDashboard from './components/AttentionDashboard';
import ProgressDashboard from './components/ProgressDashboard';
import SpacedRepetitionReview from './components/SpacedRepetitionReview';
import CognitiveLoadIndicator from './components/CognitiveLoadIndicator';
import ParentTherapistDashboard from './components/ParentTherapistDashboard';
import SharedCameraProvider from './components/SharedCameraProvider';
// import HandGestureReader from './components/HandGestureReader'; // Temporarily disabled
import './App.css';

function App() {
  const [storyData, setStoryData] = useState(null);
  const [score, setScore] = useState(0);
  const [userId, setUserId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showConfetti, setShowConfetti] = useState(false);
  const [activeTab, setActiveTab] = useState('game'); // 'game', 'rewards', 'engagement', 'attention', 'progress', 'review', 'brain', 'reports'
  const [selectedTopic, setSelectedTopic] = useState('a friendly animal');

  // Topic options for story generation - child-friendly, culturally relevant
  const storyTopics = [
    { id: 'animal', label: '🐘 සතුන්', labelEn: 'Animals', value: 'a friendly animal in Sri Lanka' },
    { id: 'school', label: '🏫 පාසල', labelEn: 'School', value: 'a fun day at school' },
    { id: 'food', label: '🍛 ආහාර', labelEn: 'Food', value: 'delicious Sri Lankan food' },
    { id: 'nature', label: '🌳 ස්වභාවය', labelEn: 'Nature', value: 'beautiful nature and trees' },
    { id: 'family', label: '👨‍👩‍👧 පවුල', labelEn: 'Family', value: 'a happy family activity' },
    { id: 'festival', label: '🎊 උත්සව', labelEn: 'Festivals', value: 'a Sri Lankan festival celebration' },
  ];

  // --- Speak Function using Gemini Audio Generation (Free!) ---
  const speak = async (text) => {
    try {
      const response = await fetch(`${API_BASE_URL}/text-to-speech`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text }),
      });
      
      if (response.ok) {
        const data = await response.json();
        // Decode base64 audio and play it
        const audioData = atob(data.audio);
        const arrayBuffer = new ArrayBuffer(audioData.length);
        const view = new Uint8Array(arrayBuffer);
        for (let i = 0; i < audioData.length; i++) {
          view[i] = audioData.charCodeAt(i);
        }
        const blob = new Blob([arrayBuffer], { type: 'audio/mp3' });
        const audioUrl = URL.createObjectURL(blob);
        const audio = new Audio(audioUrl);
        audio.play();
      } else {
        throw new Error('TTS audio generation failed');
      }
    } catch (error) {
      console.error('Error with Gemini TTS, falling back to browser TTS:', error);
      // Fallback to browser TTS if Gemini fails
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = 'si-LK';
      utterance.rate = 0.9;
      utterance.pitch = 1.2;
      window.speechSynthesis.speak(utterance);
    }
  };

  // --- User and Profile Management ---
  useEffect(() => {
    const currentUserId = '123e4567-e89b-12d3-a456-426614174000'; // Placeholder UUID
    setUserId(currentUserId);

    const setupUserProfile = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/profile/${currentUserId}`);
        const profile = await response.json();
        setScore(profile.score != null && !isNaN(profile.score) ? profile.score : 0);
      } catch (error) {
        console.error('Error fetching profile:', error);
        setScore(0);
      }
    };
    
    setupUserProfile();
  }, []);

  // --- Story and Answer Logic ---
  const fetchStory = async () => {
    if (!userId) return;
    setLoading(true);
    setStoryData(null);

    try {
      const response = await fetch(`${API_BASE_URL}/generate-story`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, topic: selectedTopic }),
      });
      const data = await response.json();

      if (response.ok) {
        setStoryData(data.story);
        // Don't auto-speak here - the SentenceBySentenceStory component will handle it
      } else {
        throw new Error(data.detail || "Failed to fetch story");
      }
    } catch (error) {
      console.error("Error fetching story:", error);
      alert(`Error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

    const handleAnswer = async (earnedStars) => {
    // If it's a number, it's the star count from the story completion
    const stars = typeof earnedStars === 'number' ? earnedStars : 0;
    
    if (stars > 0) {
      setShowConfetti(true);
      setTimeout(() => setShowConfetti(false), 3000);
      
      // Update score via backend API
      try {
        const response = await fetch(`${API_BASE_URL}/update-score`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ user_id: userId, stars: stars })
        });
        const data = await response.json();
        const newScore = data.new_score || 0;
        setScore(newScore);
        speak(`ඔබ තරු ${stars} ක් ලබා ගත්තා! මුළු ලකුණු ${newScore}`);
      } catch (error) {
        console.error('Error updating score:', error);
      }
    }
    
    // Reset for next story
    setStoryData(null);
  };
  
  // const handleGesture = (gesture) => {
  //   if (gesture === 'wave' && storyData?.options?.length > 0) {
  //     handleAnswer(storyData.options[0]);
  //   }
  // };

  return (
    <SharedCameraProvider>
      <div className="min-h-screen app-container relative overflow-hidden p-4">
        {/* Floating decorative elements */}
        <div className="floating-clouds">
          <div className="cloud" style={{top: '10%', left: '10%', animationDelay: '0s'}}>☁️</div>
          <div className="cloud" style={{top: '15%', right: '15%', animationDelay: '2s'}}>☁️</div>
          <div className="cloud" style={{top: '70%', left: '5%', animationDelay: '4s'}}>☁️</div>
        </div>

        <div className="stars">
          <div className="star" style={{top: '5%', left: '20%', animationDelay: '0s'}}>⭐</div>
          <div className="star" style={{top: '10%', right: '25%', animationDelay: '1s'}}>✨</div>
        <div className="star" style={{top: '80%', left: '15%', animationDelay: '2s'}}>🌟</div>
        <div className="star" style={{top: '75%', right: '10%', animationDelay: '1.5s'}}>⭐</div>
      </div>

      {/* Confetti overlay */}
      {showConfetti && (
        <div className="confetti-container">
          {[...Array(30)].map((_, i) => (
            <div key={i} className="confetti" style={{
              left: `${Math.random() * 100}%`,
              animationDelay: `${Math.random() * 0.5}s`,
              backgroundColor: ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff'][Math.floor(Math.random() * 5)]
            }}></div>
          ))}
        </div>
      )}

      {/* Main content container */}
      <div className="max-w-7xl mx-auto h-screen flex flex-col py-2">
        {/* Header with Score Badge */}
        <div className="text-center mb-2 relative">
          {/* Score Badge - Floating Top Left */}
          <div className="absolute left-0 top-0 bg-gradient-to-r from-yellow-400 to-orange-400 rounded-full px-4 py-2 shadow-2xl border-4 border-white flex items-center gap-2 transform hover:scale-110 transition-transform z-10">
            <span className="text-xl">⭐</span>
            <span className="text-lg font-bold text-white">{isNaN(score) || score == null ? 0 : score}</span>
          </div>

          <h1 className="text-3xl md:text-4xl font-bold text-white mb-0.5 bg-gradient-to-r from-yellow-300 via-pink-300 to-purple-300 bg-clip-text text-transparent animate-gradient drop-shadow-lg">
            සිංහල කතා ලෝකය
          </h1>
          <p className="text-base text-white drop-shadow-md">🌈 Sinhala Story World 📚</p>
        </div>

        {/* Tab Navigation */}
        <div className="flex gap-2 mb-3 justify-center">
          <button
            onClick={() => setActiveTab('game')}
            className={`px-6 py-2 rounded-full font-bold transition-all ${
              activeTab === 'game'
                ? 'bg-gradient-to-r from-green-400 to-blue-500 text-white shadow-lg scale-105'
                : 'bg-white/30 text-white hover:bg-white/40'
            }`}
          >
            🎮 Game
          </button>
          <button
            onClick={() => setActiveTab('rewards')}
            className={`px-6 py-2 rounded-full font-bold transition-all ${
              activeTab === 'rewards'
                ? 'bg-gradient-to-r from-yellow-400 to-orange-400 text-white shadow-lg scale-105'
                : 'bg-white/30 text-white hover:bg-white/40'
            }`}
          >
            🏆 Rewards
          </button>
          <button
            onClick={() => setActiveTab('engagement')}
            className={`px-6 py-2 rounded-full font-bold transition-all ${
              activeTab === 'engagement'
                ? 'bg-gradient-to-r from-purple-400 to-pink-400 text-white shadow-lg scale-105'
                : 'bg-white/30 text-white hover:bg-white/40'
            }`}
          >
            📊 Engagement
          </button>
          <button
            onClick={() => setActiveTab('attention')}
            className={`px-6 py-2 rounded-full font-bold transition-all ${
              activeTab === 'attention'
                ? 'bg-gradient-to-r from-blue-400 to-indigo-400 text-white shadow-lg scale-105'
                : 'bg-white/30 text-white hover:bg-white/40'
            }`}
          >
            👁️ Attention
          </button>
          <button
            onClick={() => setActiveTab('progress')}
            className={`px-6 py-2 rounded-full font-bold transition-all ${
              activeTab === 'progress'
                ? 'bg-gradient-to-r from-teal-400 to-cyan-400 text-white shadow-lg scale-105'
                : 'bg-white/30 text-white hover:bg-white/40'
            }`}
          >
            📈 Progress
          </button>
          <button
            onClick={() => setActiveTab('review')}
            className={`px-6 py-2 rounded-full font-bold transition-all ${
              activeTab === 'review'
                ? 'bg-gradient-to-r from-pink-400 to-rose-400 text-white shadow-lg scale-105'
                : 'bg-white/30 text-white hover:bg-white/40'
            }`}
          >
            🧠 Review
          </button>
          <button
            onClick={() => setActiveTab('brain')}
            className={`px-6 py-2 rounded-full font-bold transition-all ${
              activeTab === 'brain'
                ? 'bg-gradient-to-r from-cyan-400 to-blue-400 text-white shadow-lg scale-105'
                : 'bg-white/30 text-white hover:bg-white/40'
            }`}
          >
            🔬 Brain Load
          </button>
          <button
            onClick={() => setActiveTab('reports')}
            className={`px-6 py-2 rounded-full font-bold transition-all ${
              activeTab === 'reports'
                ? 'bg-gradient-to-r from-amber-400 to-orange-500 text-white shadow-lg scale-105'
                : 'bg-white/30 text-white hover:bg-white/40'
            }`}
          >
            📋 Reports
          </button>
        </div>

        {/* Tab Content */}
        <div className="flex-1 flex flex-col overflow-hidden min-h-0">
          {activeTab === 'game' && (
            <>
              {/* Topic Selector */}
              {!storyData && !loading && (
                <div className="mb-2">
                  <p className="text-white text-sm font-semibold text-center mb-1">🎨 මාතෘකාවක් තෝරන්න (Pick a topic):</p>
                  <div className="flex gap-1.5 justify-center flex-wrap">
                    {storyTopics.map(topic => (
                      <button
                        key={topic.id}
                        onClick={() => setSelectedTopic(topic.value)}
                        className={`px-3 py-1.5 rounded-full text-xs font-bold transition-all ${
                          selectedTopic === topic.value
                            ? 'bg-white text-purple-700 shadow-lg scale-105 ring-2 ring-purple-300'
                            : 'bg-white/20 text-white hover:bg-white/40'
                        }`}
                      >
                        {topic.label}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              <button
                onClick={fetchStory}
                disabled={loading || storyData}
                className={`w-full mb-2 bg-gradient-to-r from-green-400 to-blue-500 hover:from-green-500 hover:to-blue-600 text-white font-bold py-2 px-6 rounded-full text-lg shadow-lg transform hover:scale-105 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed ${loading ? 'animate-pulse' : ''}`}
              >
                {loading ? (
                  <span className="flex items-center justify-center gap-2">
                    <span className="animate-spin">⏳</span> 
                    නව කතාවක් ගෙන එනවා...
                  </span>
                ) : storyData ? (
                  '📚 නව කතාවක් 🎉'
                ) : (
                  '📖 නව කතාවක් ගන්න 🔥'
                )}
              </button>

              <div className="flex-1 overflow-hidden min-h-0">
                {storyData && (
                  <div className="h-full">
                    <SentenceBySentenceStory
                      storyData={storyData}
                      onComplete={handleAnswer}
                      onScoreUpdate={(newScore) => setScore(newScore)}
                      userId={userId}
                    />
                  </div>
                )}
              </div>
            </>
          )}

          {activeTab === 'rewards' && (
            <div className="flex-1 overflow-auto">
              <RewardDashboard userId={userId} score={score} />
            </div>
          )}

          {activeTab === 'engagement' && userId && (
            <div className="flex-1 overflow-auto">
              <EngagementDashboard userId={userId} />
            </div>
          )}

          {activeTab === 'attention' && userId && (
            <div className="flex-1 overflow-auto">
              <AttentionDashboard userId={userId} />
            </div>
          )}

          {activeTab === 'progress' && userId && (
            <div className="flex-1 overflow-auto">
              <ProgressDashboard userId={userId} />
            </div>
          )}

          {activeTab === 'review' && userId && (
            <div className="flex-1 overflow-auto">
              <SpacedRepetitionReview userId={userId} />
            </div>
          )}

          {activeTab === 'brain' && userId && (
            <div className="flex-1 overflow-auto">
              <CognitiveLoadIndicator userId={userId} />
            </div>
          )}

          {activeTab === 'reports' && userId && (
            <div className="flex-1 overflow-auto">
              <ParentTherapistDashboard userId={userId} />
            </div>
          )}
        </div>
      </div>
    </div>
    </SharedCameraProvider>
  );
}

export default App;
