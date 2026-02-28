import React, { useState, useEffect, useRef, useCallback } from 'react';
import HandGestureDetector from './HandGestureDetector';
import EngagementTracker from './EngagementTracker';
import GazeTracker from './GazeTracker';
import AttentionHeatmapOverlay from './AttentionHeatmapOverlay';
import API_BASE_URL from '../config';

/**
 * SentenceBySentenceStory - Core Game Component
 * 
 * Implements the listen-then-answer therapy game flow with:
 * - Sentence-by-sentence story playback with Gemini TTS
 * - Hand gesture answer selection (1-4 fingers, 2s confirm timer)
 * - Real-time engagement tracking and session analytics
 * - Visual/audio feedback with encourage/reward system
 * - Gaze tracking and attention heatmap overlay
 * 
 * Research basis:
 * - Gamification in speech therapy (Deterding et al., 2021)
 * - Multimodal interaction for hearing-impaired children (Knoors & Marschark, 2020)
 * - Adaptive scaffolding (Molenaar, 2022)
 */
function SentenceBySentenceStory({ storyData, onComplete, onScoreUpdate, userId }) {
  const [currentSentenceIndex, setCurrentSentenceIndex] = useState(0);
  const [showQuestion, setShowQuestion] = useState(false);
  const [hasListened, setHasListened] = useState(false);
  const [earnedStars, setEarnedStars] = useState(0);
  const [totalQuestions, setTotalQuestions] = useState(0);
  const [showFinalReward, setShowFinalReward] = useState(false);
  const [answered, setAnswered] = useState(false);
  const [selectedAnswer, setSelectedAnswer] = useState(null);
  const [isCorrect, setIsCorrect] = useState(false);
  const [useGesture, setUseGesture] = useState(true); // Toggle for gesture control
  
  // New states for improved gesture interaction
  const [hoveredOptionIndex, setHoveredOptionIndex] = useState(null); // Which option is currently highlighted
  const [confirmProgress, setConfirmProgress] = useState(0); // 0-100 progress of confirmation
  const confirmTimerRef = useRef(null);
  const confirmStartTimeRef = useRef(null);
  const hoveredOptionRef = useRef(null); // Track current hovered option synchronously to prevent restarts
  const CONFIRM_DURATION = 2000; // 2 seconds to confirm (changed from 2500ms)
  
  // Engagement tracking states
  // eslint-disable-next-line no-unused-vars
  const [gestureAccuracy, setGestureAccuracy] = useState(0);
  // eslint-disable-next-line no-unused-vars
  const [hasEyeContact, setHasEyeContact] = useState(true);
  // eslint-disable-next-line no-unused-vars
  const [currentEmotion, setCurrentEmotion] = useState('neutral'); // Track child's emotion (happy, sad, frustrated, neutral)
  const engagementTrackerRef = useRef(null);

  // Attention tracking states
  const [showHeatmap, setShowHeatmap] = useState(false);
  // eslint-disable-next-line no-unused-vars
  const [isGazeTrackingActive, setIsGazeTrackingActive] = useState(true);

  // Cleanup timer on unmount or when moving to next question
  useEffect(() => {
    return () => {
      if (confirmTimerRef.current) {
        clearInterval(confirmTimerRef.current);
      }
    };
  }, []);
  
  // Reset gesture states when question changes or is answered
  useEffect(() => {
    setHoveredOptionIndex(null);
    setConfirmProgress(0);
    if (confirmTimerRef.current) {
      clearInterval(confirmTimerRef.current);
    }
  }, [currentSentenceIndex, answered]);

  // Record answer to session analytics (must be before early returns)
  const recordSessionAnswer = useCallback(async (word, isCorrect, responseTime) => {
    if (!userId) return;
    try {
      await fetch(`${API_BASE_URL}/session/record-answer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          word: word,
          is_correct: isCorrect,
          response_time: responseTime,
          difficulty: 2,
          engagement_score: 50
        })
      });
    } catch (err) {
      console.warn('Session recording failed:', err);
    }
  }, [userId]);

  // Persist performance to DB (update-performance writes to performance_logs + adaptive state)
  const updatePerformance = useCallback(async (isCorrect, responseTime, storyId = 0) => {
    if (!userId) return;
    try {
      fetch(`${API_BASE_URL}/update-performance`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          story_id: storyId,
          is_correct: isCorrect,
          response_time: responseTime,
          engagement_score: 50,
          difficulty_level: 2
        })
      }).catch(err => console.warn('Performance update failed:', err));
    } catch (e) { /* non-critical */ }
  }, [userId]);

  // Track engagement directly without needing the timer (fallback)
  const trackEngagementDirect = useCallback(async (isCorrect, responseTime) => {
    if (!userId) return;
    try {
      fetch(`${API_BASE_URL}/track-engagement`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          emotion: isCorrect ? 'happy' : 'neutral',
          gesture_accuracy: 0.7,
          response_time_seconds: Math.max(0.5, responseTime),
          has_eye_contact: true
        })
      }).catch(err => console.warn('Engagement tracking failed:', err));
    } catch (e) { /* non-critical */ }
  }, [userId]);

  // Complete session and persist to learning trajectory
  const completeSession = useCallback(async () => {
    if (!userId) return;
    try {
      await fetch(`${API_BASE_URL}/session/complete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          include_trajectory: true
        })
      });
    } catch (err) {
      console.warn('Session complete failed:', err);
    }
  }, [userId]);

  // Record cognitive load signal → populates Brain Load tab
  const recordCognitiveLoad = useCallback((isCorrect, responseTime, difficultyLevel = 2) => {
    if (!userId) return;
    const wordDifficulty = Math.min(1.0, 0.2 + (difficultyLevel - 1) * 0.25);
    fetch(`${API_BASE_URL}/cognitive-load/record`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: userId,
        response_time: responseTime,
        is_correct: isCorrect,
        audio_replayed: false,
        help_requested: false,
        hesitated: responseTime > 5,
        engagement_score: isCorrect ? 75 : 40,
        word_difficulty: wordDifficulty,
        phoneme_count: 3,
        difficulty_level: difficultyLevel
      })
    }).catch(err => console.warn('Cognitive load record failed:', err));
  }, [userId]);

  // Add word to SRS deck and record a review → populates Review tab
  const recordSRSAnswer = useCallback((word, isCorrect, responseTime) => {
    if (!userId || !word) return;
    // SM-2 quality: 5=perfect, 4=correct, 3=correct-hard, 1=wrong
    const quality = isCorrect
      ? (responseTime < 3 ? 5 : responseTime < 6 ? 4 : 3)
      : 1;
    // Fire-and-forget: add word then immediately review it
    fetch(`${API_BASE_URL}/srs/add-word`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: userId,
        word: word,
        phonemes: null,
        difficulty_class: 'medium'
      })
    }).then(() =>
      fetch(`${API_BASE_URL}/srs/review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          word: word,
          quality: quality,
          response_time: responseTime,
          confused_with: null
        })
      })
    ).catch(err => console.warn('SRS tracking failed:', err));
  }, [userId]);

  // Track simulated gaze for attention heatmap — maps option position to screen coordinates
  // Options are arranged horizontally, so we map index 0-3 to x positions across the bottom half
  const trackGaze = useCallback((optionIndex, totalOptions) => {
    if (!userId) return;
    // Map option index to approximate normalized screen coordinates (0-1)
    const positions = [
      { x: 0.15, y: 0.72 },
      { x: 0.38, y: 0.72 },
      { x: 0.62, y: 0.72 },
      { x: 0.85, y: 0.72 },
    ];
    const pos = positions[optionIndex % 4] || { x: 0.5, y: 0.7 };
    // Add small random jitter to simulate realistic gaze movement
    const jitter = () => (Math.random() - 0.5) * 0.05;
    fetch(`${API_BASE_URL}/track-gaze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: userId,
        x: Math.max(0, Math.min(1, pos.x + jitter())),
        y: Math.max(0, Math.min(1, pos.y + jitter())),
        confidence: 0.85
      })
    }).catch(() => {});
    // Also send a center-of-screen gaze for the question word area
    fetch(`${API_BASE_URL}/track-gaze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: userId,
        x: 0.5 + jitter(),
        y: 0.35 + jitter(),
        confidence: 0.9
      })
    }).catch(() => {});
  }, [userId]);

  // Safety check for story_sentences
  if (!storyData || !storyData.story_sentences || storyData.story_sentences.length === 0) {
    return <div className="text-center p-8 text-red-500">කතාවක් නොමැත</div>;
  }

  const currentSentence = storyData.story_sentences[currentSentenceIndex];
  const isLastSentence = currentSentenceIndex === storyData.story_sentences.length - 1;
  
  // Safety check for current sentence
  if (!currentSentence) {
    return <div className="text-center p-8 text-red-500">දෝෂයකි</div>;
  }

  const handleAnswer = (selectedWord) => {
    if (!currentSentence || answered) return;
    
    const correct = selectedWord === currentSentence.target_word;
    setIsCorrect(correct);
    setSelectedAnswer(selectedWord);
    setAnswered(true);
    
    // End response timer for engagement tracking
    if (engagementTrackerRef.current && userId) {
      engagementTrackerRef.current.endResponseTimer();
    }
    
    // Record to session analytics (Feature 7)
    const responseTime = Date.now() / 1000 - (engagementTrackerRef.current?.responseStartTime || Date.now() / 1000);
    const rt = Math.max(0.5, responseTime);
    recordSessionAnswer(currentSentence.target_word, correct, rt);

    // Persist to performance_logs in DB (essential for Progress/Reports tabs)
    updatePerformance(correct, rt, storyData?.id || 0);

    // Ensure engagement is always tracked even if timer wasn't started
    trackEngagementDirect(correct, rt);

    // Record cognitive load signal → Brain Load tab
    recordCognitiveLoad(correct, rt, 2);

    // Add word to SRS and record review → Review tab
    recordSRSAnswer(currentSentence.target_word, correct, rt);

    // Track gaze position → Attention tab (maps selected option to screen coordinates)
    const optIdx = (currentSentence.options || []).indexOf(selectedWord);
    trackGaze(optIdx >= 0 ? optIdx : 0, (currentSentence.options || []).length);
    
    // Track phoneme confusion (Feature 2) 
    if (userId && currentSentence.target_word) {
      try {
        fetch(`${API_BASE_URL}/track-phoneme-confusion`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_id: userId,
            target_word: currentSentence.target_word,
            selected_word: selectedWord,
            is_correct: correct
          })
        }).catch(err => console.warn('Phoneme tracking failed:', err));
      } catch (e) { /* non-critical */ }
    }
    
    // Update stats
    if (correct) {
      setEarnedStars(prev => {
        const newStars = prev + 1;
        // Update parent score (10 points per star)
        if (onScoreUpdate) {
          onScoreUpdate(newStars * 10);
        }
        return newStars;
      });
    }
    setTotalQuestions(prev => prev + 1);
    
    // Speak feedback
    if (correct) {
      speakText("හරි! ඉතා හොඳයි!");
    } else {
      speakText(`වැරදියි. නිවැරදි වචනය ${currentSentence.target_word}`);
    }
    
    // Auto-move to next after 3 seconds
    setTimeout(() => {
      moveToNext();
    }, 3000);
  };

  // Handle real-time gesture updates (for highlighting and progress)
  const handleGestureUpdate = (fingerCount) => {
    console.log('👋 Gesture update - Finger count:', fingerCount);
    
    // Block if already answered
    if (answered) {
      console.log('⛔ Blocked: already answered');
      return;
    }
    
    if (!currentSentence || !currentSentence.options) {
      console.log('⛔ Blocked: no sentence or options');
      return;
    }
    
    // Map finger count (1-4) to option index (0-3)
    const optionIndex = fingerCount - 1;
    
    // Reset if no valid finger or finger count is 0
    if (fingerCount === 0 || optionIndex < 0 || optionIndex >= currentSentence.options.length) {
      console.log('🚫 Resetting - invalid finger count or out of range');
      setHoveredOptionIndex(null);
      hoveredOptionRef.current = null; // Reset ref
      setConfirmProgress(0);
      if (confirmTimerRef.current) {
        clearInterval(confirmTimerRef.current);
        confirmTimerRef.current = null;
      }
      confirmStartTimeRef.current = null;
      return;
    }
    
    // Check if this is actually a NEW finger position using the ref (not state)
    if (hoveredOptionRef.current !== optionIndex) {
      console.log('🎯 NEW finger detected! Option:', optionIndex + 1, '- Starting timer');
      setHoveredOptionIndex(optionIndex);
      hoveredOptionRef.current = optionIndex; // Update ref immediately
      setConfirmProgress(0);
      
      // Clear old timer
      if (confirmTimerRef.current) {
        console.log('🗑️ Clearing previous timer');
        clearInterval(confirmTimerRef.current);
      }
      
      // Start fresh timer
      const startTime = Date.now();
      confirmStartTimeRef.current = startTime;
      console.log('⏱️ Timer started at:', startTime, '| Target duration:', CONFIRM_DURATION, 'ms');
      
      let tickCount = 0;
      confirmTimerRef.current = setInterval(() => {
        tickCount++;
        
        if (!confirmStartTimeRef.current) {
          console.error('❌ ERROR: Start time is null!');
          return;
        }
        
        const now = Date.now();
        const elapsed = now - confirmStartTimeRef.current;
        const progress = Math.min((elapsed / CONFIRM_DURATION) * 100, 100);
        
        console.log(`⏳ Tick #${tickCount}: ${elapsed}ms elapsed → ${Math.round(progress)}%`);
        setConfirmProgress(progress);
        
        // Submit when 100%
        if (progress >= 100) {
          console.log('✅ 100% COMPLETE! Submitting answer...');
          clearInterval(confirmTimerRef.current);
          confirmTimerRef.current = null;
          confirmStartTimeRef.current = null;
          hoveredOptionRef.current = null; // Reset ref
          const selectedWord = currentSentence.options[optionIndex];
          handleAnswer(selectedWord);
        }
      }, 50);
      
      console.log('🔄 Timer ID:', confirmTimerRef.current);
    }
    // If same finger (hoveredOptionRef.current === optionIndex), timer continues without restarting
  };

  const moveToNext = () => {
    if (isLastSentence) {
      setShowFinalReward(true);
    } else {
      const nextIndex = currentSentenceIndex + 1;
      if (nextIndex < storyData.story_sentences.length) {
        setCurrentSentenceIndex(nextIndex);
        setShowQuestion(false);
        setHasListened(false);
        setAnswered(false);
        setSelectedAnswer(null);
        setIsCorrect(false);
      } else {
        setShowFinalReward(true);
      }
    }
  };

  const handleListen = () => {
    if (!currentSentence || !currentSentence.text) return;
    
    // Speak the FULL sentence including the difficult word
    // The child must listen and identify which word they heard
    speakText(currentSentence.text);
    
    setHasListened(true);
  };

  const handleShowQuestion = () => {
    if (!currentSentence) return;
    
    if (currentSentence.has_target_word) {
      setShowQuestion(true);
      // Start response timer for engagement tracking
      if (engagementTrackerRef.current && userId) {
        engagementTrackerRef.current.startResponseTimer();
      }
    } else {
      // If no target word in this sentence, just move to next
      if (isLastSentence) {
        setShowFinalReward(true);
      } else {
        const nextIndex = currentSentenceIndex + 1;
        if (nextIndex < storyData.story_sentences.length) {
          setCurrentSentenceIndex(nextIndex);
          setHasListened(false);
        }
      }
    }
  };

  const speakText = async (text) => {
    // Try Gemini TTS first for better Sinhala pronunciation
    try {
      const response = await fetch(`${API_BASE_URL}/text-to-speech`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      if (response.ok) {
        const data = await response.json();
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
        return;
      }
    } catch (err) {
      console.warn('gTTS failed, falling back to browser TTS:', err);
    }
    
    // Fallback to browser TTS
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'si-LK';
    utterance.rate = 0.8;
    utterance.pitch = 1.1;
    window.speechSynthesis.speak(utterance);
  };

  // Function to display text with gap where difficult word is
  const getDisplayText = () => {
    if (!currentSentence || !currentSentence.text) {
      return '';
    }
    if (!currentSentence.has_target_word || !currentSentence.target_word) {
      return currentSentence.text;
    }
    return currentSentence.text.replace(currentSentence.target_word, '____');
  };

  // Final reward screen
  if (showFinalReward) {
    const percentage = totalQuestions > 0 ? Math.round((earnedStars / totalQuestions) * 100) : 0;
    let message = '';
    let emoji = '';
    
    if (percentage === 100) {
      message = 'පරිපූර්ණයි! සියල්ල නිවැරදියි! 🎉';
      emoji = '🏆';
      speakText('අති විශිෂ්ටයි! ඔබ සියලුම පිළිතුරු නිවැරදිව දුන්නා!');
    } else if (percentage >= 70) {
      message = 'ඉතා හොඳයි! 👏';
      emoji = '🌟';
      speakText('හොඳයි! ඔබ හොඳින් කළා!');
    } else if (percentage >= 50) {
      message = 'හොඳ උත්සාහයක්! 💪';
      emoji = '⭐';
      speakText('හොඳ උත්සාහයක්! තව පුහුණු වන්න!');
    } else {
      message = 'පුහුණු වන්න! 🌈';
      emoji = '✨';
      speakText('තව පුහුණු වන්න! ඔබට පුළුවන්!');
    }

    return (
      <div className="bg-gradient-to-br from-purple-300 via-pink-300 to-yellow-200 p-6 rounded-3xl shadow-2xl border-4 border-yellow-400 animate-bounce-in h-full flex flex-col justify-center">
        <div className="text-center">
          <div className="text-7xl mb-4 animate-pulse">{emoji}</div>
          <h2 className="text-3xl font-bold mb-3 text-purple-800">{message}</h2>
          
          <div className="bg-white bg-opacity-90 p-4 rounded-2xl mb-4">
            <p className="text-2xl font-bold text-gray-700 mb-2">ප්‍රතිඵලය</p>
            
            {/* Stars Display */}
            <div className="flex justify-center gap-1 mb-3">
              {[...Array(totalQuestions)].map((_, index) => (
                <span 
                  key={index} 
                  className={`text-4xl ${index < earnedStars ? 'animate-star-bounce' : ''}`}
                  style={{animationDelay: `${index * 0.1}s`}}
                >
                  {index < earnedStars ? '⭐' : '☆'}
                </span>
              ))}
            </div>
            
            <p className="text-4xl font-bold text-purple-600 mb-1">
              {earnedStars} / {totalQuestions}
            </p>
            <p className="text-xl text-gray-600">{percentage}% නිවැරදියි</p>
          </div>

          <button
            onClick={async () => { await completeSession(); onComplete(earnedStars); }}
            className="bg-gradient-to-r from-green-400 to-blue-500 hover:from-green-500 hover:to-blue-600 text-white font-bold py-4 px-8 rounded-full text-2xl shadow-2xl transform hover:scale-110 transition-all duration-300"
          >
            ✅ අවසන්
          </button>
        </div>
      </div>
    );
  }

  if (showQuestion && currentSentence.has_target_word) {
    return (
      <div className="bg-gradient-to-br from-purple-200 to-pink-200 p-3 rounded-3xl shadow-2xl border-4 border-purple-400 animate-fade-in h-full flex flex-col overflow-hidden">
        {/* Engagement Tracker */}
        {userId && (
          <EngagementTracker
            ref={engagementTrackerRef}
            userId={userId}
            currentEmotion={currentEmotion}
            gestureAccuracy={gestureAccuracy}
            hasEyeContact={hasEyeContact}
          />
        )}
        
        {/* Progress indicator at top - More compact */}
        <div className="flex justify-between items-center mb-2">
          <div className="bg-white bg-opacity-80 px-3 py-1 rounded-full text-xs font-bold text-gray-700">
            ප්‍රශ්නය {totalQuestions + 1}
          </div>
          <div className="bg-gradient-to-r from-yellow-400 to-orange-400 text-white px-3 py-1 rounded-full font-bold flex items-center gap-1 shadow-lg">
            <span className="text-base">⭐</span>
            <span className="text-sm">{earnedStars} / {totalQuestions}</span>
          </div>
        </div>

        <h3 className="text-lg font-bold mb-2 text-purple-800 text-center">
          👂 ඔබට ඇහුණු වචනය කුමක්ද?
        </h3>

        {/* Two-column layout: Camera LEFT, Options RIGHT */}
        <div className="flex-1 grid grid-cols-2 gap-2 min-h-0 overflow-hidden">
          {/* LEFT SIDE - Camera and Controls */}
          <div className="flex flex-col gap-2 min-h-0">
            {/* Listen button at top */}
            <div className="bg-white bg-opacity-90 p-2 rounded-xl shadow-md">
              <button
                onClick={handleListen}
                className="w-full bg-blue-500 hover:bg-blue-600 text-white px-2 py-1.5 rounded-lg text-xs font-bold flex items-center justify-center gap-1"
              >
                🔊 නැවත අහන්න
              </button>
            </div>

            {/* Camera view - takes remaining space */}
            {useGesture && (
              <div className="flex-1 bg-white bg-opacity-90 rounded-xl shadow-md overflow-hidden min-h-0">
                <HandGestureDetector 
                  onGestureDetected={handleGestureUpdate}
                  isActive={showQuestion && !answered}
                />
              </div>
            )}

            {/* Toggle button */}
            <button
              onClick={() => setUseGesture(!useGesture)}
              className="bg-purple-500 hover:bg-purple-600 text-white px-2 py-1.5 rounded-lg text-xs font-bold shadow-lg"
            >
              {useGesture ? '🖱️ බොත්තම්' : '🖐️ ඇඟිලි'}
            </button>
          </div>

          {/* RIGHT SIDE - Answer Options */}
          <div className="flex flex-col gap-2 min-h-0">
            {/* Feedback at top if answered */}
            {answered && (
              <div className={`p-1.5 rounded-xl ${isCorrect ? 'bg-green-200' : 'bg-red-200'} animate-fade-in`}>
                <p className={`text-xs font-bold text-center ${isCorrect ? 'text-green-800' : 'text-red-800'}`}>
                  {isCorrect ? '✅ හරි!' : `❌ ${currentSentence.target_word}`}
                </p>
              </div>
            )}

            {/* Answer options - stacked vertically */}
            <div className="flex-1 flex flex-col gap-1.5 min-h-0">
              {currentSentence.options && currentSentence.options.map((option, index) => {
                const isHovered = hoveredOptionIndex === index;
                const showProgress = isHovered && confirmTimerRef.current !== null;
                
                return (
                  <button
                    key={index}
                    onClick={() => !useGesture && handleAnswer(option)}
                    disabled={answered || useGesture}
                    className={`font-bold py-2 px-2 rounded-lg text-sm shadow-lg transform transition-all duration-200 relative flex-1
                      ${answered 
                        ? option === currentSentence.target_word
                          ? 'bg-green-500 text-white'
                          : option === selectedAnswer
                            ? 'bg-red-400 text-white'
                            : 'bg-gray-300 text-gray-600'
                        : isHovered
                          ? 'bg-yellow-400 text-purple-900 scale-105 ring-4 ring-yellow-300 shadow-2xl'
                          : useGesture
                            ? 'bg-blue-400 text-white'
                            : 'bg-gradient-to-r from-green-400 to-blue-500 hover:from-green-500 hover:to-blue-600 text-white hover:scale-105 active:scale-95 cursor-pointer'
                      }
                      ${(answered || useGesture) && 'cursor-not-allowed'}
                    `}
                  >
                    {/* Background progress fill */}
                    {showProgress && (
                      <div 
                        className="absolute inset-0 bg-gradient-to-r from-green-400 to-blue-500 opacity-70 transition-all duration-100 rounded-lg overflow-hidden"
                        style={{ width: `${confirmProgress}%` }}
                      />
                    )}
                    
                    {/* Circular progress indicator */}
                    {showProgress && (
                      <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 z-30">
                        <svg className="w-16 h-16 transform -rotate-90 drop-shadow-lg" viewBox="0 0 36 36">
                          {/* Background circle */}
                          <circle
                            cx="18"
                            cy="18"
                            r="15"
                            fill="rgba(0,0,0,0.3)"
                            stroke="rgba(255,255,255,0.5)"
                            strokeWidth="2"
                          />
                          {/* Progress circle */}
                          <circle
                            cx="18"
                            cy="18"
                            r="15"
                            fill="none"
                            stroke="#FFEB3B"
                            strokeWidth="4"
                            strokeDasharray={`${(confirmProgress / 100) * 94.2} 94.2`}
                            strokeLinecap="round"
                          />
                        </svg>
                        {/* Progress percentage text */}
                        <div className="absolute inset-0 flex items-center justify-center">
                          <span className="text-white font-bold text-xs drop-shadow-md">
                            {Math.round(confirmProgress)}%
                          </span>
                        </div>
                      </div>
                    )}
                    
                    {/* Finger count badge */}
                    <span className={`absolute top-0.5 left-0.5 rounded-full w-5 h-5 flex items-center justify-center text-xs font-bold shadow z-20 ${
                      isHovered ? 'bg-white text-yellow-600 animate-pulse' : 'bg-white text-purple-700'
                    }`}>
                      {index + 1}
                    </span>
                    <span className="ml-5 block text-left text-xs sm:text-sm relative z-10">{option}</span>
                  </button>
                );
              })}
            </div>

            {/* Next button */}
            {answered && (
              <button
                onClick={moveToNext}
                className="bg-gradient-to-r from-purple-400 to-pink-500 hover:from-purple-500 hover:to-pink-600 text-white font-bold py-1.5 px-3 rounded-lg text-sm shadow-xl transform hover:scale-105 transition-all"
              >
                {isLastSentence ? '🎉 ප්‍රතිඵල' : '➡️ ඊළඟ'}
              </button>
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gradient-to-br from-yellow-100 to-pink-100 p-8 rounded-3xl shadow-2xl border-4 border-purple-400 relative overflow-hidden animate-fade-in">
      {/* Progress and score at top */}
      <div className="flex justify-between items-center mb-6">
        <div className="bg-white bg-opacity-90 px-4 py-2 rounded-full font-bold text-gray-700 shadow-md">
          📖 වාක්‍යය {currentSentenceIndex + 1} / {storyData.story_sentences.length}
        </div>
        <div className="bg-gradient-to-r from-yellow-400 to-orange-400 text-white px-4 py-2 rounded-full font-bold flex items-center gap-2 shadow-lg">
          <span className="text-xl">⭐</span>
          <span>{earnedStars}</span>
        </div>
      </div>
      
      <div className="relative z-10">
        <h2 className="text-3xl font-bold mb-8 text-purple-700 text-center">
          <span>👂</span> අහන්න සහ ඉගෙන ගන්න <span>📚</span>
        </h2>

        {/* Step 1: Listen Button */}
        {!hasListened && (
          <div className="bg-white bg-opacity-90 p-8 rounded-2xl shadow-inner mb-6 text-center">
            <p className="text-2xl text-gray-700 mb-6">මෙම වාක්‍යය අසන්න</p>
            <button
              onClick={handleListen}
              className="bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 text-white px-8 py-4 rounded-full font-bold text-2xl shadow-lg transform hover:scale-110 transition-all"
            >
              🔊 අහන්න
            </button>
          </div>
        )}

        {/* Step 2: Show text with blanks */}
        {hasListened && (
          <div className="bg-white bg-opacity-90 p-8 rounded-2xl shadow-inner mb-6 animate-slide-in-left">
            <p className="text-3xl leading-relaxed text-gray-800 font-semibold text-center mb-6">
              {getDisplayText()}
            </p>
            
            {currentSentence.has_target_word && (
              <p className="text-lg text-orange-600 text-center mb-4">
                ____ තැන කුමක් තිබේ දැයි ඔබට අසන්නේ ද?
              </p>
            )}

            <div className="flex gap-4 justify-center">
              <button
                onClick={handleListen}
                className="bg-blue-500 hover:bg-blue-600 text-white px-6 py-3 rounded-full font-bold"
              >
                🔊 නැවත අහන්න
              </button>
              
              <button
                onClick={handleShowQuestion}
                className="bg-gradient-to-r from-green-400 to-blue-500 hover:from-green-500 hover:to-blue-600 text-white font-bold py-3 px-8 rounded-full text-xl shadow-lg transform hover:scale-105 transition-all"
              >
                {currentSentence.has_target_word ? '✅ පිළිතුරු දෙන්න' : '➡️ ඊළඟ වාක්‍යය'}
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Gaze Tracker (invisible in background) */}
      {userId && isGazeTrackingActive && (
        <GazeTracker 
          userId={userId}
          showVisualization={false}
          isActive={!showFinalReward && !answered}
        />
      )}

      {/* Attention Heatmap Overlay (toggle visibility) */}
      {userId && showHeatmap && (
        <AttentionHeatmapOverlay 
          userId={userId}
          colorScheme="hot"
          showGazePath={true}
          showHotspots={true}
          opacity={0.6}
          isVisible={showHeatmap}
        />
      )}

      {/* Heatmap Toggle Button */}
      {userId && (
        <button
          onClick={() => setShowHeatmap(!showHeatmap)}
          className="fixed bottom-20 right-20 bg-purple-500 hover:bg-purple-600 text-white px-4 py-2 rounded-full shadow-lg z-[1000] transition-all"
          style={{ pointerEvents: 'auto' }}
        >
          {showHeatmap ? '👁️ Hide Heatmap' : '👁️ Show Heatmap'}
        </button>
      )}
    </div>
  );
}

export default SentenceBySentenceStory;
