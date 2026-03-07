import React, { useState, useEffect, useRef, useCallback } from 'react';
import HandGestureDetector from './HandGestureDetector';
import EngagementTracker from './EngagementTracker';
import GazeTracker from './GazeTracker';
import AttentionHeatmapOverlay from './AttentionHeatmapOverlay';
import StorytellingScene from './StorytellingScene';
import API_BASE_URL from '../config';

/**
 * SentenceBySentenceStory - Core Game Component (Immersive Edition)
 * 
 * Now wrapped in the beautiful Grandmother's Magic Story Garden scene.
 * All original logic preserved + enhanced with:
 * - Grandmother character who "tells" the story
 * - Children who react to answers
 * - Speech bubbles for story text
 * - Magical campfire nighttime atmosphere
 * - Celebration animations
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
  const [useGesture, setUseGesture] = useState(true);

  // Speaking state for Grandmother
  const [isSpeaking, setIsSpeaking] = useState(false);
  // Children reaction state
  const [childrenReaction, setChildrenReaction] = useState('listening');
  // Celebration type
  const [celebrationType, setCelebrationType] = useState(null);
  // Feedback display
  const [showFeedback, setShowFeedback] = useState(false);
  
  // Gesture interaction states
  const [hoveredOptionIndex, setHoveredOptionIndex] = useState(null);
  const [confirmProgress, setConfirmProgress] = useState(0);
  const confirmTimerRef = useRef(null);
  const confirmStartTimeRef = useRef(null);
  const hoveredOptionRef = useRef(null);
  const CONFIRM_DURATION = 2000;

  // Engagement tracking states
  // eslint-disable-next-line no-unused-vars
  const [gestureAccuracy, setGestureAccuracy] = useState(0);
  // eslint-disable-next-line no-unused-vars
  const [hasEyeContact, setHasEyeContact] = useState(true);
  // eslint-disable-next-line no-unused-vars
  const [currentEmotion, setCurrentEmotion] = useState('neutral');
  const engagementTrackerRef = useRef(null);

  // Attention tracking
  const [showHeatmap, setShowHeatmap] = useState(false);
  // eslint-disable-next-line no-unused-vars
  const [isGazeTrackingActive, setIsGazeTrackingActive] = useState(true);

  // Track if session has been completed to avoid double-save
  const sessionCompletedRef = useRef(false);
  const hasAnsweredRef = useRef(false);

  // Timer cleanup + auto-save session on unmount
  useEffect(() => {
    return () => {
      if (confirmTimerRef.current) clearInterval(confirmTimerRef.current);
      // Auto-save session if user navigates away without clicking Finish
      if (userId && hasAnsweredRef.current && !sessionCompletedRef.current) {
        try {
          const payload = JSON.stringify({ user_id: userId, include_trajectory: true });
          navigator.sendBeacon(`${API_BASE_URL}/session/complete`, new Blob([payload], { type: 'application/json' }));
        } catch (e) { /* best effort */ }
      }
    };
  }, [userId]);

  useEffect(() => {
    setHoveredOptionIndex(null);
    setConfirmProgress(0);
    if (confirmTimerRef.current) clearInterval(confirmTimerRef.current);
  }, [currentSentenceIndex, answered]);

  // =====================
  // API TRACKING FUNCTIONS (preserved from original)
  // =====================
  const recordSessionAnswer = useCallback(async (word, isCorrect, responseTime) => {
    if (!userId) return;
    try {
      await fetch(`${API_BASE_URL}/session/record-answer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId, word, is_correct: isCorrect,
          response_time: responseTime, difficulty: 2, engagement_score: 50
        })
      });
    } catch (err) { console.warn('Session recording failed:', err); }
  }, [userId]);

  const updatePerformance = useCallback(async (isCorrect, responseTime, storyId = 0) => {
    if (!userId) return;
    try {
      fetch(`${API_BASE_URL}/update-performance`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId, story_id: storyId, is_correct: isCorrect,
          response_time: responseTime, engagement_score: 50, difficulty_level: 2
        })
      }).catch(err => console.warn('Performance update failed:', err));
    } catch (e) { /* non-critical */ }
  }, [userId]);

  const trackEngagementDirect = useCallback(async (isCorrect, responseTime) => {
    if (!userId) return;
    try {
      fetch(`${API_BASE_URL}/track-engagement`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId, emotion: isCorrect ? 'happy' : 'neutral',
          gesture_accuracy: 0.7, response_time_seconds: Math.max(0.5, responseTime),
          has_eye_contact: true
        })
      }).catch(err => console.warn('Engagement tracking failed:', err));
    } catch (e) { /* non-critical */ }
  }, [userId]);

  const completeSession = useCallback(async () => {
    if (!userId) return;
    sessionCompletedRef.current = true;
    try {
      await fetch(`${API_BASE_URL}/session/complete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, include_trajectory: true })
      });
    } catch (err) { console.warn('Session complete failed:', err); }
  }, [userId]);

  const recordCognitiveLoad = useCallback(async (isCorrect, responseTime, difficultyLevel = 2) => {
    if (!userId) return;
    const wordDifficulty = Math.min(1.0, 0.2 + (difficultyLevel - 1) * 0.25);
    try {
      await fetch(`${API_BASE_URL}/cognitive-load/record`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId, response_time: responseTime, is_correct: isCorrect,
          audio_replayed: false, help_requested: false, hesitated: responseTime > 5,
          engagement_score: isCorrect ? 75 : 40, word_difficulty: wordDifficulty,
          phoneme_count: 3, difficulty_level: difficultyLevel
        })
      });
    } catch (err) { console.warn('Cognitive load record failed:', err); }
  }, [userId]);

  const recordSRSAnswer = useCallback((word, isCorrect, responseTime) => {
    if (!userId || !word) return;
    const quality = isCorrect ? (responseTime < 3 ? 5 : responseTime < 6 ? 4 : 3) : 1;
    fetch(`${API_BASE_URL}/srs/add-word`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: userId, word, phonemes: null, difficulty_class: 'medium' })
    }).then(() =>
      fetch(`${API_BASE_URL}/srs/review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, word, quality, response_time: responseTime, confused_with: null })
      })
    ).catch(err => console.warn('SRS tracking failed:', err));
  }, [userId]);

  const trackGaze = useCallback(async (optionIndex) => {
    if (!userId) return;
    const positions = [
      { x: 0.15, y: 0.72 }, { x: 0.38, y: 0.72 },
      { x: 0.62, y: 0.72 }, { x: 0.85, y: 0.72 },
    ];
    const pos = positions[optionIndex % 4] || { x: 0.5, y: 0.7 };
    const jitter = () => (Math.random() - 0.5) * 0.05;
    try {
      await Promise.all([
        fetch(`${API_BASE_URL}/track-gaze`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ user_id: userId, x: Math.max(0, Math.min(1, pos.x + jitter())), y: Math.max(0, Math.min(1, pos.y + jitter())), confidence: 0.85 })
        }),
        fetch(`${API_BASE_URL}/track-gaze`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ user_id: userId, x: 0.5 + jitter(), y: 0.35 + jitter(), confidence: 0.9 })
        })
      ]);
    } catch (err) { console.warn('Gaze tracking failed:', err); }
  }, [userId]);

  // =====================
  // SAFETY CHECKS
  // =====================
  if (!storyData || !storyData.story_sentences || storyData.story_sentences.length === 0) {
    return <div className="text-center p-8 text-red-500">කතාවක් නොමැත</div>;
  }

  const currentSentence = storyData.story_sentences[currentSentenceIndex];
  const isLastSentence = currentSentenceIndex === storyData.story_sentences.length - 1;

  if (!currentSentence) {
    return <div className="text-center p-8 text-red-500">දෝෂයකි</div>;
  }

  // =====================
  // GAME LOGIC
  // =====================
  const handleAnswer = async (selectedWord) => {
    if (!currentSentence || answered) return;

    const correct = selectedWord === currentSentence.target_word;
    setIsCorrect(correct);
    setSelectedAnswer(selectedWord);
    setAnswered(true);
    setShowFeedback(true);
    hasAnsweredRef.current = true;

    // Children reaction
    setChildrenReaction(correct ? 'excited' : 'sad');

    // Celebration
    setCelebrationType(correct ? 'correct' : 'wrong');
    setTimeout(() => setCelebrationType(null), 2500);

    // Engagement tracking
    if (engagementTrackerRef.current && userId) {
      engagementTrackerRef.current.endResponseTimer();
    }

    const responseTime = Date.now() / 1000 - (engagementTrackerRef.current?.responseStartTime || Date.now() / 1000);
    const rt = Math.max(0.5, responseTime);

    // Fire all tracking calls in parallel and await them all
    const trackingPromises = [
      recordSessionAnswer(currentSentence.target_word, correct, rt),
      updatePerformance(correct, rt, storyData?.id || 0),
      trackEngagementDirect(correct, rt),
      recordCognitiveLoad(correct, rt, 2),
      recordSRSAnswer(currentSentence.target_word, correct, rt),
    ];

    const optIdx = (currentSentence.options || []).indexOf(selectedWord);
    trackingPromises.push(trackGaze(optIdx >= 0 ? optIdx : 0));

    // Phoneme tracking
    if (userId && currentSentence.target_word) {
      trackingPromises.push(
        fetch(`${API_BASE_URL}/track-phoneme-confusion`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_id: userId, target_word: currentSentence.target_word,
            selected_word: selectedWord, is_correct: correct
          })
        }).catch(err => console.warn('Phoneme tracking failed:', err))
      );
    }

    // Wait for all tracking to complete (don't block UI feedback though)
    Promise.allSettled(trackingPromises).catch(() => {});

    // Update stars
    if (correct) {
      setEarnedStars(prev => {
        const newStars = prev + 1;
        if (onScoreUpdate) onScoreUpdate(newStars * 10);
        return newStars;
      });
    }
    setTotalQuestions(prev => prev + 1);

    // Speak feedback with grandmother animation
    setIsSpeaking(true);
    if (correct) {
      speakText("හරි! ඉතා හොඳයි!");
    } else {
      speakText(`වැරදියි. නිවැරදි වචනය ${currentSentence.target_word}`);
    }
    setTimeout(() => setIsSpeaking(false), 2000);

    // Auto-move
    setTimeout(() => {
      setShowFeedback(false);
      moveToNext();
    }, 3000);
  };

  // Gesture handling (preserved)
  const handleGestureUpdate = (fingerCount) => {
    if (answered) return;
    if (!currentSentence || !currentSentence.options) return;

    const optionIndex = fingerCount - 1;

    if (fingerCount === 0 || optionIndex < 0 || optionIndex >= currentSentence.options.length) {
      setHoveredOptionIndex(null);
      hoveredOptionRef.current = null;
      setConfirmProgress(0);
      if (confirmTimerRef.current) { clearInterval(confirmTimerRef.current); confirmTimerRef.current = null; }
      confirmStartTimeRef.current = null;
      return;
    }

    if (hoveredOptionRef.current !== optionIndex) {
      setHoveredOptionIndex(optionIndex);
      hoveredOptionRef.current = optionIndex;
      setConfirmProgress(0);

      if (confirmTimerRef.current) clearInterval(confirmTimerRef.current);

      const startTime = Date.now();
      confirmStartTimeRef.current = startTime;

      confirmTimerRef.current = setInterval(() => {
        if (!confirmStartTimeRef.current) return;
        const elapsed = Date.now() - confirmStartTimeRef.current;
        const progress = Math.min((elapsed / CONFIRM_DURATION) * 100, 100);
        setConfirmProgress(progress);

        if (progress >= 100) {
          clearInterval(confirmTimerRef.current);
          confirmTimerRef.current = null;
          confirmStartTimeRef.current = null;
          hoveredOptionRef.current = null;
          handleAnswer(currentSentence.options[optionIndex]);
        }
      }, 50);
    }
  };

  const moveToNext = () => {
    setChildrenReaction('listening');
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
    setIsSpeaking(true);
    speakText(currentSentence.text);
    setHasListened(true);
    // Children look attentive
    setChildrenReaction('listening');
    setTimeout(() => setIsSpeaking(false), 3000);
  };

  const handleShowQuestion = () => {
    if (!currentSentence) return;
    if (currentSentence.has_target_word) {
      setShowQuestion(true);
      setChildrenReaction('thinking');
      if (engagementTrackerRef.current && userId) {
        engagementTrackerRef.current.startResponseTimer();
      }
    } else {
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
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'si-LK';
    utterance.rate = 0.8;
    utterance.pitch = 1.1;
    window.speechSynthesis.speak(utterance);
  };

  const getDisplayText = () => {
    if (!currentSentence || !currentSentence.text) return '';
    if (!currentSentence.has_target_word || !currentSentence.target_word) return currentSentence.text;
    return currentSentence.text.replace(currentSentence.target_word, '____');
  };

  // Calculate progress
  const storyProgress = ((currentSentenceIndex + 1) / storyData.story_sentences.length) * 100;

  // =====================
  // FINAL REWARD SCREEN
  // =====================
  if (showFinalReward) {
    const percentage = totalQuestions > 0 ? Math.round((earnedStars / totalQuestions) * 100) : 0;
    let emoji = '🏆';
    let message = '';

    if (percentage === 100) {
      message = 'පරිපූර්ණයි! සියල්ල නිවැරදියි! 🎉';
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
      <StorytellingScene
        isSpeaking={true}
        childrenReaction="excited"
        celebrationType="complete"
        storyProgress={100}
        earnedStars={earnedStars}
        totalQuestions={totalQuestions}
        currentSentence={storyData.story_sentences.length}
        totalSentences={storyData.story_sentences.length}
      >
        <div className="final-reward-scene">
          <div className="reward-trophy">{emoji}</div>
          <div className="reward-message">{message}</div>

          <div className="reward-stars-display">
            {[...Array(totalQuestions)].map((_, index) => (
              <span
                key={index}
                className="reward-star"
                style={{ '--delay': `${0.5 + index * 0.15}s` }}
              >
                {index < earnedStars ? '⭐' : '☆'}
              </span>
            ))}
          </div>

          <div className="reward-score-text">
            {earnedStars} / {totalQuestions}
          </div>
          <div className="reward-percentage">{percentage}% නිවැරදියි</div>

          <button
            onClick={async () => { await completeSession(); onComplete(earnedStars); }}
            className="reward-finish-btn"
          >
            ✅ අවසන්
          </button>
        </div>
      </StorytellingScene>
    );
  }

  // =====================
  // QUESTION PHASE (answer selection)
  // =====================
  if (showQuestion && currentSentence.has_target_word) {
    return (
      <StorytellingScene
        isSpeaking={isSpeaking}
        childrenReaction={childrenReaction}
        celebrationType={celebrationType}
        storyProgress={storyProgress}
        earnedStars={earnedStars}
        totalQuestions={totalQuestions}
        currentSentence={currentSentenceIndex + 1}
        totalSentences={storyData.story_sentences.length}
      >
        {/* Hidden Engagement Tracker */}
        {userId && (
          <EngagementTracker
            ref={engagementTrackerRef}
            userId={userId}
            currentEmotion={currentEmotion}
            gestureAccuracy={gestureAccuracy}
            hasEyeContact={hasEyeContact}
          />
        )}

        {/* Speech bubble with the sentence (blank) */}
        <div className="story-bubble-container">
          <div className="story-speech-bubble">
            <div className="bubble-sparkle" style={{ top: -8, right: 10 }}>✨</div>
            <div className="bubble-sparkle" style={{ top: -5, left: 15, animationDelay: '1s' }}>💫</div>
            <div className="bubble-story-text">
              {getDisplayText().split('____').map((part, i, arr) => (
                <React.Fragment key={i}>
                  {part}
                  {i < arr.length - 1 && <span className="blank-space" />}
                </React.Fragment>
              ))}
            </div>
          </div>
        </div>

        {/* Question label */}
        <div className="question-text">
          👂 ඔබට ඇහුණු වචනය කුමක්ද?
        </div>

        {/* Re-listen button */}
        <button className="relisten-btn" onClick={handleListen}>
          🔊 නැවත අහන්න
        </button>

        {/* Feedback overlay */}
        {showFeedback && (
          <div className="feedback-toast">
            <span className="feedback-emoji">
              {isCorrect ? '🎉' : '💪'}
            </span>
            <span className="feedback-text">
              {isCorrect ? 'හරි! ඉතා හොඳයි!' : `${currentSentence.target_word}`}
            </span>
          </div>
        )}

        {/* Answer option cards */}
        {!answered && (
          <div className="answer-options-container">
            {currentSentence.options && currentSentence.options.map((option, index) => {
              const isHovered = hoveredOptionIndex === index;
              const showProgress = isHovered && confirmTimerRef.current !== null;

              return (
                <div
                  key={index}
                  className={`answer-option-card ${
                    answered
                      ? option === currentSentence.target_word
                        ? 'correct'
                        : option === selectedAnswer
                          ? 'incorrect'
                          : 'revealed'
                      : ''
                  }`}
                  onClick={() => !useGesture && !answered && handleAnswer(option)}
                  style={{ cursor: useGesture ? 'default' : 'pointer' }}
                >
                  {/* Progress fill for gesture mode */}
                  {showProgress && (
                    <div className="option-progress-bg">
                      <div
                        className="option-progress-fill"
                        style={{ height: `${confirmProgress}%` }}
                      />
                    </div>
                  )}

                  <div className="option-number">{index + 1}</div>
                  <div className="option-text">{option}</div>

                  {/* Circular progress for gesture mode */}
                  {showProgress && (
                    <div className="option-progress-ring">
                      <svg width="50" height="50" viewBox="0 0 36 36" style={{ transform: 'rotate(-90deg)' }}>
                        <circle cx="18" cy="18" r="15" fill="none" stroke="rgba(156,39,176,0.2)" strokeWidth="3" />
                        <circle cx="18" cy="18" r="15" fill="none" stroke="#9c27b0" strokeWidth="3"
                          strokeDasharray={`${(confirmProgress / 100) * 94.2} 94.2`}
                          strokeLinecap="round"
                        />
                      </svg>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* Answered state - show correct answer + next button */}
        {answered && (
          <>
            <div className="answer-options-container">
              {currentSentence.options && currentSentence.options.map((option, index) => (
                <div
                  key={index}
                  className={`answer-option-card ${
                    option === currentSentence.target_word
                      ? 'correct'
                      : option === selectedAnswer
                        ? 'incorrect'
                        : 'revealed'
                  }`}
                >
                  <div className="option-number">{index + 1}</div>
                  <div className="option-text">{option}</div>
                </div>
              ))}
            </div>

            <button className="next-sentence-btn" onClick={moveToNext}>
              {isLastSentence ? '🎉 ප්‍රතිඵල' : '➡️ ඊළඟ'}
            </button>
          </>
        )}

        {/* Gesture camera panel - Large & visible */}
        {useGesture && (
          <div className="gesture-camera-panel">
            <HandGestureDetector
              onGestureDetected={handleGestureUpdate}
              isActive={showQuestion && !answered}
            />
          </div>
        )}

        {/* Gesture toggle + finger count status */}
        <div className="gesture-mode-badge">
          <button className="gesture-toggle-btn" onClick={() => setUseGesture(!useGesture)}>
            {useGesture ? '🖱️ බොත්තම්' : '🖐️ ඇඟිලි'}
          </button>
          {useGesture && hoveredOptionIndex !== null && (
            <span className="gesture-live-indicator">
              ☝️ {hoveredOptionIndex + 1} තෝරාගෙන...
            </span>
          )}
        </div>

        {/* Gaze Tracker */}
        {userId && isGazeTrackingActive && (
          <GazeTracker userId={userId} showVisualization={false} isActive={!showFinalReward && !answered} />
        )}
        {userId && showHeatmap && (
          <AttentionHeatmapOverlay userId={userId} colorScheme="hot" showGazePath={true} showHotspots={true} opacity={0.6} isVisible={showHeatmap} />
        )}
      </StorytellingScene>
    );
  }

  // =====================
  // LISTENING PHASE (story presentation)
  // =====================
  return (
    <StorytellingScene
      isSpeaking={isSpeaking}
      childrenReaction={childrenReaction}
      storyProgress={storyProgress}
      earnedStars={earnedStars}
      totalQuestions={totalQuestions}
      currentSentence={currentSentenceIndex + 1}
      totalSentences={storyData.story_sentences.length}
    >
      {/* Speech bubble with story text */}
      {hasListened && (
        <div className="story-bubble-container">
          <div className="story-speech-bubble">
            <div className="bubble-sparkle" style={{ top: -8, right: 10 }}>✨</div>
            <div className="bubble-sparkle" style={{ top: -5, left: 15, animationDelay: '1s' }}>💫</div>
            <div className="bubble-sparkle" style={{ bottom: -5, right: 30, animationDelay: '2s' }}>🌟</div>
            <div className="bubble-story-text">
              {getDisplayText().split('____').map((part, i, arr) => (
                <React.Fragment key={i}>
                  {part}
                  {i < arr.length - 1 && <span className="blank-space" />}
                </React.Fragment>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Center content - Listen / Answer buttons */}
      <div className="listen-phase-content">
        {!hasListened && (
          <>
            <div className="listen-instruction">
              🎵 අහන්න... දේවදූතයා කතාව කියනවා!
            </div>
            <button className="magical-listen-btn" onClick={handleListen}>
              <span className="btn-shine" />
              🔊 අහන්න - Listen
            </button>
          </>
        )}

        {hasListened && (
          <>
            {currentSentence.has_target_word && (
              <div className="listen-instruction" style={{ marginBottom: 8 }}>
                📝 ____ තැන කුමක් තිබේ දැයි ඔබට අසන්නේ ද?
              </div>
            )}
            <div style={{ display: 'flex', gap: 12, justifyContent: 'center', flexWrap: 'wrap' }}>
              <button className="magical-listen-btn" onClick={handleListen} style={{ padding: '12px 24px', fontSize: '1rem' }}>
                <span className="btn-shine" />
                🔊 නැවත අහන්න
              </button>
              <button
                className="magical-listen-btn"
                onClick={handleShowQuestion}
                style={{
                  padding: '12px 24px',
                  fontSize: '1rem',
                  background: 'linear-gradient(135deg, #00e676, #00c853, #00a843)'
                }}
              >
                <span className="btn-shine" />
                {currentSentence.has_target_word ? '✅ පිළිතුරු දෙන්න' : '➡️ ඊළඟ වාක්‍යය'}
              </button>
            </div>
          </>
        )}
      </div>

      {/* Gesture camera panel - always visible in standby mode */}
      {useGesture && (
        <div className="gesture-camera-panel">
          <HandGestureDetector
            onGestureDetected={handleGestureUpdate}
            isActive={false}
          />
        </div>
      )}

      {/* Gesture toggle */}
      <div className="gesture-mode-badge">
        <button className="gesture-toggle-btn" onClick={() => setUseGesture(!useGesture)}>
          {useGesture ? '🖱️ බොත්තම්' : '🖐️ ඇඟිලි'}
        </button>
      </div>

      {/* Gaze Tracker (invisible) */}
      {userId && isGazeTrackingActive && (
        <GazeTracker userId={userId} showVisualization={false} isActive={!showFinalReward && !answered} />
      )}
      {userId && showHeatmap && (
        <AttentionHeatmapOverlay userId={userId} colorScheme="hot" showGazePath={true} showHotspots={true} opacity={0.6} isVisible={showHeatmap} />
      )}

      {/* Heatmap Toggle */}
      {userId && (
        <button
          onClick={() => setShowHeatmap(!showHeatmap)}
          className="gesture-toggle-btn"
          style={{ position: 'absolute', bottom: 12, left: 12, zIndex: 1000 }}
        >
          {showHeatmap ? '👁️ Hide Heatmap' : '👁️ Show Heatmap'}
        </button>
      )}
    </StorytellingScene>
  );
}

export default SentenceBySentenceStory;
