import React, { useState, useEffect, useRef } from 'react';
import StorytellingScene from './StorytellingScene';
import API_BASE_URL from '../config';

/**
 * SentenceBySentenceStory — Core Game Component
 *
 * This is the main game loop for word memorization:
 *   1. LISTEN  — Child hears the sentence via Gemini TTS
 *   2. ANSWER  — Sentence shown with a blank (____); child picks the missing word
 *   3. FEEDBACK — Correct → star + celebration  |  Wrong → show correct answer
 *   4. Repeat for every sentence, then show final reward screen
 *
 * Input modes:
 *   - Click mode  — child clicks an answer option
 *   - Gesture mode — child holds up 1–4 fingers for 2 sec (MediaPipe hand detection)
 *
 * Props:
 *   storyData     — { story_sentences: [{ text, target_word, options, has_target_word }] }
 *   onComplete    — called with earned star count when the story finishes
 *   onScoreUpdate — called with live score for the header badge
 *   userId        — current child's user ID
 */
function SentenceBySentenceStory({ storyData, onComplete, onScoreUpdate, userId }) {
  // ─── Game state ───
  const [currentSentenceIndex, setCurrentSentenceIndex] = useState(0);
  const [showQuestion, setShowQuestion] = useState(false);      // true → answer phase
  const [hasListened, setHasListened] = useState(false);         // true after clicking Listen
  const [earnedStars, setEarnedStars] = useState(0);            // correct answer count
  const [totalQuestions, setTotalQuestions] = useState(0);       // total answered count
  const [showFinalReward, setShowFinalReward] = useState(false);
  const [answered, setAnswered] = useState(false);
  const [selectedAnswer, setSelectedAnswer] = useState(null);
  const [isCorrect, setIsCorrect] = useState(false);
  const [useGesture, setUseGesture] = useState(false);          // click vs gesture toggle

  // ─── Visual / animation state ───
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [childrenReaction, setChildrenReaction] = useState('listening');
  const [celebrationType, setCelebrationType] = useState(null);
  const [showFeedback, setShowFeedback] = useState(false);

  // ─── Gesture hold-to-confirm state ───
  const [hoveredOptionIndex, setHoveredOptionIndex] = useState(null);
  const [confirmProgress, setConfirmProgress] = useState(0);
  const confirmTimerRef = useRef(null);
  const confirmStartTimeRef = useRef(null);
  const hoveredOptionRef = useRef(null);
  const CONFIRM_DURATION = 2000; // hold 2 seconds to confirm

  // Clean up gesture timer on unmount
  useEffect(() => {
    return () => { if (confirmTimerRef.current) clearInterval(confirmTimerRef.current); };
  }, []);

  // Reset gesture hover when sentence changes or after answering
  useEffect(() => {
    setHoveredOptionIndex(null);
    setConfirmProgress(0);
    if (confirmTimerRef.current) clearInterval(confirmTimerRef.current);
  }, [currentSentenceIndex, answered]);

  // ─── Safety: no story data ───
  if (!storyData?.story_sentences?.length) {
    return <div className="text-center p-8 text-red-500">කතාවක් නොමැත</div>;
  }
  const currentSentence = storyData.story_sentences[currentSentenceIndex];
  const isLastSentence = currentSentenceIndex === storyData.story_sentences.length - 1;
  if (!currentSentence) {
    return <div className="text-center p-8 text-red-500">දෝෂයකි</div>;
  }

  // ═══════════════════════════════════════
  // GAME LOGIC
  // ═══════════════════════════════════════

  /** Called when child selects an answer (click or gesture confirm) */
  const handleAnswer = (selectedWord) => {
    if (!currentSentence || answered) return;

    const correct = selectedWord === currentSentence.target_word;
    setIsCorrect(correct);
    setSelectedAnswer(selectedWord);
    setAnswered(true);
    setShowFeedback(true);

    // Animate children characters
    setChildrenReaction(correct ? 'excited' : 'sad');
    setCelebrationType(correct ? 'correct' : 'wrong');
    setTimeout(() => setCelebrationType(null), 2500);

    // Update star count
    if (correct) {
      setEarnedStars(prev => {
        const newStars = prev + 1;
        if (onScoreUpdate) onScoreUpdate(newStars * 10);
        return newStars;
      });
    }
    setTotalQuestions(prev => prev + 1);

    // Speak Sinhala feedback
    setIsSpeaking(true);
    speakText(correct ? "හරි! ඉතා හොඳයි!" : `වැරදියි. නිවැරදි වචනය ${currentSentence.target_word}`);
    setTimeout(() => setIsSpeaking(false), 2000);

    // Auto-advance after 3 seconds
    setTimeout(() => { setShowFeedback(false); moveToNext(); }, 3000);
  };

  /** Gesture input: child holds up 1–4 fingers → hover option → 2s confirm */
  const handleGestureUpdate = (fingerCount) => {
    if (answered || !currentSentence?.options) return;
    const optionIndex = fingerCount - 1;

    // No fingers or out of range → reset
    if (fingerCount === 0 || optionIndex < 0 || optionIndex >= currentSentence.options.length) {
      setHoveredOptionIndex(null);
      hoveredOptionRef.current = null;
      setConfirmProgress(0);
      if (confirmTimerRef.current) { clearInterval(confirmTimerRef.current); confirmTimerRef.current = null; }
      confirmStartTimeRef.current = null;
      return;
    }

    // New finger count → start hold-to-confirm timer
    if (hoveredOptionRef.current !== optionIndex) {
      setHoveredOptionIndex(optionIndex);
      hoveredOptionRef.current = optionIndex;
      setConfirmProgress(0);
      if (confirmTimerRef.current) clearInterval(confirmTimerRef.current);

      confirmStartTimeRef.current = Date.now();
      confirmTimerRef.current = setInterval(() => {
        if (!confirmStartTimeRef.current) return;
        const progress = Math.min(((Date.now() - confirmStartTimeRef.current) / CONFIRM_DURATION) * 100, 100);
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

  /** Move to next sentence or show reward screen */
  const moveToNext = () => {
    setChildrenReaction('listening');
    if (isLastSentence) {
      setShowFinalReward(true);
    } else {
      setCurrentSentenceIndex(prev => prev + 1);
      setShowQuestion(false);
      setHasListened(false);
      setAnswered(false);
      setSelectedAnswer(null);
      setIsCorrect(false);
    }
  };

  /** Play current sentence audio via Gemini TTS */
  const handleListen = () => {
    if (!currentSentence?.text) return;
    setIsSpeaking(true);
    speakText(currentSentence.text);
    setHasListened(true);
    setChildrenReaction('listening');
    setTimeout(() => setIsSpeaking(false), 3000);
  };

  /** Transition from listen → answer phase (or skip if no question) */
  const handleShowQuestion = () => {
    if (!currentSentence) return;
    if (currentSentence.has_target_word) {
      setShowQuestion(true);
      setChildrenReaction('thinking');
    } else {
      // No question for this sentence → auto-advance
      if (isLastSentence) { setShowFinalReward(true); }
      else { setCurrentSentenceIndex(prev => prev + 1); setHasListened(false); }
    }
  };

  /** Gemini TTS with browser speech synthesis fallback */
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
        for (let i = 0; i < audioData.length; i++) view[i] = audioData.charCodeAt(i);
        const blob = new Blob([arrayBuffer], { type: 'audio/wav' });
        new Audio(URL.createObjectURL(blob)).play();
        return;
      }
    } catch (err) {
      console.warn('Gemini TTS failed, falling back to browser TTS:', err);
    }
    // Fallback: browser Web Speech API (si-LK Sinhala)
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'si-LK';
    utterance.rate = 0.8;
    utterance.pitch = 1.1;
    window.speechSynthesis.speak(utterance);
  };

  /** Replace target word with ____ for fill-in-the-blank display */
  const getDisplayText = () => {
    if (!currentSentence?.text) return '';
    if (!currentSentence.has_target_word || !currentSentence.target_word) return currentSentence.text;
    return currentSentence.text.replace(currentSentence.target_word, '____');
  };

  const storyProgress = ((currentSentenceIndex + 1) / storyData.story_sentences.length) * 100;

  // ═══════════════════════════════════════
  // RENDER: FINAL REWARD SCREEN
  // ═══════════════════════════════════════
  if (showFinalReward) {
    const percentage = totalQuestions > 0 ? Math.round((earnedStars / totalQuestions) * 100) : 0;
    let emoji = '🏆', message = '';
    if (percentage === 100) {
      message = 'පරිපූර්ණයි! සියල්ල නිවැරදියි! 🎉';
      speakText('අති විශිෂ්ටයි! ඔබ සියලුම පිළිතුරු නිවැරදිව දුන්නා!');
    } else if (percentage >= 70) {
      message = 'ඉතා හොඳයි! 👏'; emoji = '🌟';
      speakText('හොඳයි! ඔබ හොඳින් කළා!');
    } else if (percentage >= 50) {
      message = 'හොඳ උත්සාහයක්! 💪'; emoji = '⭐';
      speakText('හොඳ උත්සාහයක්! තව පුහුණු වන්න!');
    } else {
      message = 'පුහුණු වන්න! 🌈'; emoji = '✨';
      speakText('තව පුහුණු වන්න! ඔබට පුළුවන්!');
    }

    return (
      <StorytellingScene isSpeaking={true} childrenReaction="excited" celebrationType="complete"
        storyProgress={100} earnedStars={earnedStars} totalQuestions={totalQuestions}
        currentSentence={storyData.story_sentences.length} totalSentences={storyData.story_sentences.length}
      >
        <div className="final-reward-scene">
          <div className="reward-trophy">{emoji}</div>
          <div className="reward-message">{message}</div>
          <div className="reward-stars-display">
            {[...Array(totalQuestions)].map((_, i) => (
              <span key={i} className="reward-star" style={{ '--delay': `${0.5 + i * 0.15}s` }}>
                {i < earnedStars ? '⭐' : '☆'}
              </span>
            ))}
          </div>
          <div className="reward-score-text">{earnedStars} / {totalQuestions}</div>
          <div className="reward-percentage">{percentage}% නිවැරදියි</div>
          <button onClick={() => onComplete(earnedStars)} className="reward-finish-btn">✅ අවසන්</button>
        </div>
      </StorytellingScene>
    );
  }

  // ═══════════════════════════════════════
  // RENDER: QUESTION PHASE (answer selection)
  // ═══════════════════════════════════════
  if (showQuestion && currentSentence.has_target_word) {
    return (
      <StorytellingScene isSpeaking={isSpeaking} childrenReaction={childrenReaction}
        celebrationType={celebrationType} storyProgress={storyProgress}
        earnedStars={earnedStars} totalQuestions={totalQuestions}
        currentSentence={currentSentenceIndex + 1} totalSentences={storyData.story_sentences.length}
      >
        {/* Sentence with blank */}
        <div className="story-bubble-container">
          <div className="story-speech-bubble">
            <div className="bubble-sparkle" style={{ top: -8, right: 10 }}>✨</div>
            <div className="bubble-sparkle" style={{ top: -5, left: 15, animationDelay: '1s' }}>💫</div>
            <div className="bubble-story-text">
              {getDisplayText().split('____').map((part, i, arr) => (
                <React.Fragment key={i}>
                  {part}{i < arr.length - 1 && <span className="blank-space" />}
                </React.Fragment>
              ))}
            </div>
          </div>
        </div>

        <div className="question-text">👂 ඔබට ඇහුණු වචනය කුමක්ද?</div>
        <button className="relisten-btn" onClick={handleListen}>🔊 නැවත අහන්න</button>

        {/* Feedback toast */}
        {showFeedback && (
          <div className="feedback-toast">
            <span className="feedback-emoji">{isCorrect ? '🎉' : '💪'}</span>
            <span className="feedback-text">{isCorrect ? 'හරි! ඉතා හොඳයි!' : currentSentence.target_word}</span>
          </div>
        )}

        {/* Answer options — before answering */}
        {!answered && (
          <div className="answer-options-container">
            {currentSentence.options?.map((option, index) => {
              const isHovered = hoveredOptionIndex === index;
              const showProg = isHovered && confirmTimerRef.current !== null;
              return (
                <div key={index} className="answer-option-card"
                  onClick={() => !useGesture && handleAnswer(option)}
                  style={{ cursor: useGesture ? 'default' : 'pointer' }}
                >
                  {showProg && (
                    <div className="option-progress-bg">
                      <div className="option-progress-fill" style={{ height: `${confirmProgress}%` }} />
                    </div>
                  )}
                  <div className="option-number">{index + 1}</div>
                  <div className="option-text">{option}</div>
                  {showProg && (
                    <div className="option-progress-ring">
                      <svg width="50" height="50" viewBox="0 0 36 36" style={{ transform: 'rotate(-90deg)' }}>
                        <circle cx="18" cy="18" r="15" fill="none" stroke="rgba(156,39,176,0.2)" strokeWidth="3" />
                        <circle cx="18" cy="18" r="15" fill="none" stroke="#9c27b0" strokeWidth="3"
                          strokeDasharray={`${(confirmProgress / 100) * 94.2} 94.2`} strokeLinecap="round" />
                      </svg>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* After answering — highlight correct/incorrect + next button */}
        {answered && (
          <>
            <div className="answer-options-container">
              {currentSentence.options?.map((option, index) => (
                <div key={index} className={`answer-option-card ${
                  option === currentSentence.target_word ? 'correct'
                    : option === selectedAnswer ? 'incorrect' : 'revealed'
                }`}>
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

        {/* Gesture mode toggle */}
        <div className="gesture-mode-badge">
          <button className="gesture-toggle-btn" onClick={() => setUseGesture(!useGesture)}>
            {useGesture ? '🖱️ බොත්තම්' : '🖐️ ඇඟිලි'}
          </button>
          {useGesture && hoveredOptionIndex !== null && (
            <span className="gesture-live-indicator">☝️ {hoveredOptionIndex + 1} තෝරාගෙන...</span>
          )}
        </div>
      </StorytellingScene>
    );
  }

  // ═══════════════════════════════════════
  // RENDER: LISTENING PHASE
  // ═══════════════════════════════════════
  return (
    <StorytellingScene isSpeaking={isSpeaking} childrenReaction={childrenReaction}
      storyProgress={storyProgress} earnedStars={earnedStars} totalQuestions={totalQuestions}
      currentSentence={currentSentenceIndex + 1} totalSentences={storyData.story_sentences.length}
    >
      {/* Show sentence text once child has listened */}
      {hasListened && (
        <div className="story-bubble-container">
          <div className="story-speech-bubble">
            <div className="bubble-sparkle" style={{ top: -8, right: 10 }}>✨</div>
            <div className="bubble-sparkle" style={{ top: -5, left: 15, animationDelay: '1s' }}>💫</div>
            <div className="bubble-sparkle" style={{ bottom: -5, right: 30, animationDelay: '2s' }}>🌟</div>
            <div className="bubble-story-text">
              {getDisplayText().split('____').map((part, i, arr) => (
                <React.Fragment key={i}>
                  {part}{i < arr.length - 1 && <span className="blank-space" />}
                </React.Fragment>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Center: Listen / Answer buttons */}
      <div className="listen-phase-content">
        {!hasListened && (
          <>
            <div className="listen-instruction">🎵 අහන්න... දේවදූතයා කතාව කියනවා!</div>
            <button className="magical-listen-btn" onClick={handleListen}>
              <span className="btn-shine" />🔊 අහන්න - Listen
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
              <button className="magical-listen-btn" onClick={handleListen}
                style={{ padding: '12px 24px', fontSize: '1rem' }}>
                <span className="btn-shine" />🔊 නැවත අහන්න
              </button>
              <button className="magical-listen-btn" onClick={handleShowQuestion}
                style={{ padding: '12px 24px', fontSize: '1rem', background: 'linear-gradient(135deg, #00e676, #00c853, #00a843)' }}>
                <span className="btn-shine" />
                {currentSentence.has_target_word ? '✅ පිළිතුරු දෙන්න' : '➡️ ඊළඟ වාක්‍යය'}
              </button>
            </div>
          </>
        )}
      </div>

      {/* Gesture toggle */}
      <div className="gesture-mode-badge">
        <button className="gesture-toggle-btn" onClick={() => setUseGesture(!useGesture)}>
          {useGesture ? '🖱️ බොත්තම්' : '🖐️ ඇඟිලි'}
        </button>
      </div>
    </StorytellingScene>
  );
}

export default SentenceBySentenceStory;
