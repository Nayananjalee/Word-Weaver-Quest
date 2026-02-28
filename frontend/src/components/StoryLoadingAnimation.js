import React, { useState, useEffect, useCallback } from 'react';
import './StoryLoadingAnimation.css';

/**
 * StoryLoadingAnimation
 * 
 * A rich, magical loading experience for children while the AI generates a story.
 * Features animated angel, magical book, floating particles, progress messages,
 * and interactive sparkle-tap elements to maintain engagement.
 */

const StoryLoadingAnimation = ({ topic }) => {
  const [currentPhase, setCurrentPhase] = useState(0);
  const [sparkles, setSparkles] = useState([]);
  const [tapSparkles, setTapSparkles] = useState([]);
  const [sparkleCount, setSparkleCount] = useState(0);

  // Loading phases with Sinhala + English messages - cycle through them
  const phases = [
    {
      icon: '📖',
      sinhala: 'මායා පොත විවෘත වෙමින්...',
      english: 'Opening the magic book...',
      emoji: '✨📖✨',
    },
    {
      icon: '🪄',
      sinhala: 'මායා වචන එකතු කරමින්...',
      english: 'Gathering magical words...',
      emoji: '🌟🪄🌟',
    },
    {
      icon: '🎨',
      sinhala: 'කතාව පින්තාරු කරමින්...',
      english: 'Painting the story...',
      emoji: '🎨🖌️🌈',
    },
    {
      icon: '🦋',
      sinhala: 'චරිත ජීවත් කරමින්...',
      english: 'Characters coming alive...',
      emoji: '🦋🌸🐦',
    },
    {
      icon: '🌺',
      sinhala: 'මායා උයනට මල් පිපෙමින්...',
      english: 'Flowers blooming in the magic garden...',
      emoji: '🌺🌻🌷',
    },
    {
      icon: '👼',
      sinhala: 'දේවදූතයා කතාව සූදානම් කරමින්...',
      english: 'The Angel is finishing the story...',
      emoji: '👼💫⭐',
    },
  ];

  // Cycle through phases every 3.5 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentPhase(prev => (prev + 1) % phases.length);
    }, 3500);
    return () => clearInterval(interval);
  }, [phases.length]);

  // Generate floating background sparkles
  useEffect(() => {
    const initialSparkles = Array.from({ length: 20 }, (_, i) => ({
      id: i,
      x: Math.random() * 100,
      y: Math.random() * 100,
      size: Math.random() * 20 + 10,
      delay: Math.random() * 5,
      duration: Math.random() * 3 + 4,
      emoji: ['✨', '⭐', '🌟', '💫', '🦋', '🌸', '🌺', '🎵', '💛', '🌈'][Math.floor(Math.random() * 10)],
    }));
    setSparkles(initialSparkles);
  }, []);

  // Handle tap/click to create interactive sparkle bursts
  const handleTap = useCallback((e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX || e.touches?.[0]?.clientX || 0) - rect.left);
    const y = ((e.clientY || e.touches?.[0]?.clientY || 0) - rect.top);

    const burstEmojis = ['⭐', '🌟', '✨', '💫', '💛', '🎵', '🦋', '🌸', '🌺', '🌈', '💜', '💙'];

    const newSparkles = Array.from({ length: 8 }, (_, i) => ({
      id: Date.now() + i,
      x,
      y,
      emoji: burstEmojis[Math.floor(Math.random() * burstEmojis.length)],
      angle: (Math.PI * 2 * i) / 8,
      distance: Math.random() * 60 + 40,
    }));

    setTapSparkles(prev => [...prev, ...newSparkles]);
    setSparkleCount(prev => prev + 1);

    // Remove after animation
    setTimeout(() => {
      setTapSparkles(prev => prev.filter(s => !newSparkles.find(ns => ns.id === s.id)));
    }, 1200);
  }, []);

  const phase = phases[currentPhase];

  // Get topic-specific decorative emoji
  const getTopicDecorations = () => {
    if (!topic) return ['🌳', '🌺', '🦋', '🌈', '⭐'];
    if (topic.includes('animal')) return ['🐘', '🐒', '🦜', '🐿️', '🦋'];
    if (topic.includes('school')) return ['📚', '✏️', '🎒', '🏫', '📝'];
    if (topic.includes('food')) return ['🍛', '🍌', '🥥', '🍚', '🫖'];
    if (topic.includes('nature')) return ['🌳', '🌺', '🦋', '🌿', '🍃'];
    if (topic.includes('family')) return ['👨‍👩‍👧', '🏠', '❤️', '👶', '🤗'];
    if (topic.includes('festival')) return ['🎊', '🎆', '🪔', '🎭', '🥁'];
    return ['🌳', '🌺', '🦋', '🌈', '⭐'];
  };

  const topicDecos = getTopicDecorations();

  return (
    <div
      className="story-loading-container"
      onClick={handleTap}
      onTouchStart={handleTap}
    >
      {/* Background gradient overlay */}
      <div className="story-loading-bg" />

      {/* Floating background sparkles */}
      <div className="story-loading-sparkles">
        {sparkles.map(sparkle => (
          <span
            key={sparkle.id}
            className="story-loading-sparkle"
            style={{
              left: `${sparkle.x}%`,
              top: `${sparkle.y}%`,
              fontSize: `${sparkle.size}px`,
              animationDelay: `${sparkle.delay}s`,
              animationDuration: `${sparkle.duration}s`,
            }}
          >
            {sparkle.emoji}
          </span>
        ))}
      </div>

      {/* Topic-specific floating decorations (orbit) */}
      <div className="story-loading-orbit">
        {topicDecos.map((emoji, i) => (
          <span
            key={i}
            className="story-loading-orbit-item"
            style={{
              '--orbit-index': i,
              '--orbit-total': topicDecos.length,
            }}
          >
            {emoji}
          </span>
        ))}
      </div>

      {/* Central Angel character with magical effects */}
      <div className="story-loading-angel-scene">
        {/* Divine glow behind angel */}
        <div className="story-loading-divine-glow" />

        {/* Angel with book */}
        <div className="story-loading-angel">
          <div className="story-loading-angel-wings">
            <span className="story-loading-wing story-loading-wing-left">🪽</span>
            <span className="story-loading-wing story-loading-wing-right">🪽</span>
          </div>
          <div className="story-loading-angel-body">👼</div>
          <div className="story-loading-angel-halo" />
        </div>

        {/* Magical book */}
        <div className="story-loading-book">
          <div className="story-loading-book-inner">
            <span className="story-loading-book-emoji">{phase.icon}</span>
          </div>
          <div className="story-loading-book-sparkles">
            <span className="story-loading-book-sparkle bs-1">✨</span>
            <span className="story-loading-book-sparkle bs-2">💫</span>
            <span className="story-loading-book-sparkle bs-3">⭐</span>
          </div>
        </div>
      </div>

      {/* Phase message with animated transition */}
      <div className="story-loading-message" key={currentPhase}>
        <div className="story-loading-message-emojis">{phase.emoji}</div>
        <p className="story-loading-message-sinhala">{phase.sinhala}</p>
        <p className="story-loading-message-english">{phase.english}</p>
      </div>

      {/* Magical progress trail */}
      <div className="story-loading-progress-trail">
        {phases.map((p, i) => (
          <div
            key={i}
            className={`story-loading-progress-dot ${
              i < currentPhase ? 'completed' : i === currentPhase ? 'active' : ''
            }`}
          >
            <span className="story-loading-progress-icon">
              {i <= currentPhase ? p.icon : '·'}
            </span>
          </div>
        ))}
      </div>

      {/* Interactive tap sparkle bursts */}
      {tapSparkles.map(sparkle => (
        <span
          key={sparkle.id}
          className="story-loading-tap-sparkle"
          style={{
            left: sparkle.x,
            top: sparkle.y,
            '--angle': `${sparkle.angle}rad`,
            '--distance': `${sparkle.distance}px`,
          }}
        >
          {sparkle.emoji}
        </span>
      ))}

      {/* Tap instruction + counter */}
      <div className="story-loading-tap-hint">
        {sparkleCount === 0 ? (
          <>
            <p className="tap-hint-sinhala">👆 තිරය ස්පර්ශ කරන්න! ✨</p>
            <p className="tap-hint-english">Tap the screen for magic sparkles!</p>
          </>
        ) : (
          <>
            <p className="tap-hint-sinhala">
              🌟 මායා තිත් {sparkleCount} ක්! තව ස්පර්ශ කරන්න!
            </p>
            <p className="tap-hint-english">
              {sparkleCount} magic bursts! Keep tapping!
            </p>
          </>
        )}
      </div>

      {/* Bouncing characters at the bottom for extra delight */}
      <div className="story-loading-bouncing-chars">
        {['🐘', '🦋', '🌸', '🐦', '🐿️', '🌺'].map((char, i) => (
          <span
            key={i}
            className="story-loading-bouncing-char"
            style={{ animationDelay: `${i * 0.2}s` }}
          >
            {char}
          </span>
        ))}
      </div>
    </div>
  );
};

export default StoryLoadingAnimation;
