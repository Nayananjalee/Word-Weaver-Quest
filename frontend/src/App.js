import React, { useState, useEffect } from 'react';
import API_BASE_URL from './config';

const RESEARCH_APP_URL = process.env.REACT_APP_RESEARCH_URL || 'http://localhost:5173';
import SentenceBySentenceStory from './components/SentenceBySentenceStory';
import WordManager from './components/WordManager';
import SharedCameraProvider from './components/SharedCameraProvider';
import StoryLoadingAnimation from './components/StoryLoadingAnimation';
import gestureService from './components/GestureRecognizerService';
// import HandGestureReader from './components/HandGestureReader'; // Temporarily disabled
import './App.css';

// 🚀 Start preloading MediaPipe WASM + model immediately at import time
// This runs while the child sees the home screen / picks a topic
gestureService.preload();

function App() {
  const [storyData, setStoryData] = useState(null);
  const [score, setScore] = useState(0);
  const [userId, setUserId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showConfetti, setShowConfetti] = useState(false);
  const [activeTab, setActiveTab] = useState('game'); // 'game', 'words'
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
        const blob = new Blob([arrayBuffer], { type: 'audio/wav' });
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
  // Read userId and username from URL query params (passed from SilentSpark research app)
  // Falls back to default UUID if not provided (standalone mode)
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const researchUserId = params.get('userId');
    const researchUsername = params.get('username');
    const currentUserId = researchUserId || '123e4567-e89b-12d3-a456-426614174000';
    const currentUsername = researchUsername || 'default_child';
    if (researchUserId) {
      console.log('\ud83d\udd17 Research app user detected:', researchUsername, '(ID:', researchUserId, ')');
    }
    setUserId(currentUserId);

    const setupUserProfile = async () => {
      try {
        // Pass username so the backend can create/update the profile with the correct name
        const response = await fetch(`${API_BASE_URL}/profile/${currentUserId}?username=${encodeURIComponent(currentUsername)}`);
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

          {/* Back to SilentSpark - Floating Top Right */}
          <a
            href={RESEARCH_APP_URL}
            className="absolute right-0 top-0 z-10"
            style={{
              background: 'rgba(255,255,255,0.25)',
              backdropFilter: 'blur(8px)',
              border: '2px solid rgba(255,255,255,0.5)',
              borderRadius: '9999px',
              padding: '8px 18px',
              color: 'white',
              fontWeight: 700,
              fontSize: '0.9rem',
              textDecoration: 'none',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              boxShadow: '0 2px 12px rgba(0,0,0,0.2)',
              transition: 'all 0.2s'
            }}
          >
            ← SilentSpark
          </a>

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
            onClick={() => setActiveTab('words')}
            className={`px-6 py-2 rounded-full font-bold transition-all ${
              activeTab === 'words'
                ? 'bg-gradient-to-r from-purple-400 to-pink-400 text-white shadow-lg scale-105'
                : 'bg-white/30 text-white hover:bg-white/40'
            }`}
          >
            📝 Words
          </button>

        </div>

        {/* Tab Content */}
        <div className="flex-1 flex flex-col overflow-hidden min-h-0">
          {activeTab === 'game' && (
            <>
              {/* Magical Home Screen when no story is active */}
              {!storyData && !loading && (
                <div className="flex-1 flex flex-col items-center justify-center gap-4 animate-fade-in">
                  {/* Angel welcome illustration */}
                  <div className="angel-welcome-scene">
                    <div className="welcome-divine-glow"></div>
                    <div className="text-6xl mb-2 animate-bounce" style={{animationDuration: '2s'}}>👼</div>
                    <div className="welcome-speech-bubble">
                      <p className="text-lg font-bold text-amber-800 mb-1">දේවදූතයාගේ මායා කතා උයනට සාදරයෙන් පිළිගනිමු!</p>
                      <p className="text-sm text-gray-600">Welcome to the Angel's Enchanted Story Garden!</p>
                    </div>
                  </div>

                  {/* Topic Selector - Magical Cards */}
                  <div className="topic-selector-container">
                    <p className="text-white text-sm font-bold text-center mb-2 drop-shadow-lg">
                      🎨 කතාවක් තෝරන්න (Pick a topic)
                    </p>
                    <div className="grid grid-cols-3 gap-3 max-w-lg mx-auto">
                      {storyTopics.map(topic => (
                        <button
                          key={topic.id}
                          onClick={() => setSelectedTopic(topic.value)}
                          className={`topic-card group ${
                            selectedTopic === topic.value ? 'topic-card-selected' : ''
                          }`}
                        >
                          <span className="text-3xl block mb-1 group-hover:scale-125 transition-transform duration-300">
                            {topic.label.split(' ')[0]}
                          </span>
                          <span className="text-xs font-bold block text-gray-700">{topic.labelEn}</span>
                          <span className="text-xs block text-purple-600 mt-0.5">{topic.label.split(' ').slice(1).join(' ')}</span>
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Start Story Button - Big and magical */}
                  <button
                    onClick={fetchStory}
                    className="start-story-btn group"
                  >
                    <span className="start-story-btn-shine"></span>
                    <span className="relative z-10 flex items-center gap-3">
                      <span className="text-3xl group-hover:animate-bounce">📖</span>
                      <span>
                        <span className="block text-lg font-bold">කතාවක් පටන් ගමු!</span>
                        <span className="block text-xs opacity-80">Start a New Story</span>
                      </span>
                      <span className="text-3xl group-hover:animate-bounce">🌟</span>
                    </span>
                  </button>
                </div>
              )}

              {/* Loading state - Rich magical animation to keep children engaged */}
              {loading && (
                <div className="flex-1 animate-fade-in">
                  <StoryLoadingAnimation topic={selectedTopic} />
                </div>
              )}

              {/* Story Scene */}
              <div className="flex-1 overflow-hidden min-h-0">
                {storyData && (
                  <div className="h-full rounded-2xl overflow-hidden shadow-2xl">
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

          {activeTab === 'words' && userId && (
            <div className="flex-1 overflow-auto">
              <WordManager userId={userId} />
            </div>
          )}


        </div>
      </div>
    </div>
    </SharedCameraProvider>
  );
}

export default App;
