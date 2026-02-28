import React, { useEffect, useState } from 'react';

/**
 * CelebrationEffects - Fireworks, stars, hearts when child answers correctly
 * Props:
 *   - type: 'correct' | 'wrong' | 'complete' | null
 *   - onComplete: callback when animation finishes
 */
function CelebrationEffects({ type, onComplete }) {
  const [stars, setStars] = useState([]);
  const [hearts, setHearts] = useState([]);
  const [showRainbow, setShowRainbow] = useState(false);

  useEffect(() => {
    if (!type) return;

    if (type === 'correct' || type === 'complete') {
      // Generate flying stars
      const newStars = [];
      const count = type === 'complete' ? 20 : 12;
      for (let i = 0; i < count; i++) {
        const angle = (i / count) * 360;
        const distance = 80 + Math.random() * 120;
        newStars.push({
          id: i,
          emoji: ['⭐', '🌟', '✨', '💫'][Math.floor(Math.random() * 4)],
          flyX: Math.cos((angle * Math.PI) / 180) * distance,
          flyY: Math.sin((angle * Math.PI) / 180) * distance,
          delay: Math.random() * 0.3,
        });
      }
      setStars(newStars);

      // Generate floating hearts
      const newHearts = [];
      for (let i = 0; i < 8; i++) {
        newHearts.push({
          id: i,
          left: 20 + Math.random() * 60,
          delay: Math.random() * 0.5,
          emoji: ['❤️', '💛', '💜', '💚', '🧡'][Math.floor(Math.random() * 5)],
        });
      }
      setHearts(newHearts);

      if (type === 'complete') {
        setShowRainbow(true);
      }
    }

    // Clean up after animation
    const timer = setTimeout(() => {
      setStars([]);
      setHearts([]);
      setShowRainbow(false);
      if (onComplete) onComplete();
    }, 2500);

    return () => clearTimeout(timer);
  }, [type, onComplete]);

  if (!type) return null;

  return (
    <div className="celebration-container">
      {/* Star burst from center */}
      {stars.map((star) => (
        <div
          key={star.id}
          className="celebration-star"
          style={{
            top: '50%',
            left: '50%',
            '--fly-x': `${star.flyX}px`,
            '--fly-y': `${star.flyY}px`,
            animationDelay: `${star.delay}s`,
          }}
        >
          {star.emoji}
        </div>
      ))}

      {/* Floating hearts */}
      {hearts.map((heart) => (
        <div
          key={heart.id}
          className="floating-heart"
          style={{
            left: `${heart.left}%`,
            bottom: '20%',
            animationDelay: `${heart.delay}s`,
          }}
        >
          {heart.emoji}
        </div>
      ))}

      {/* Rainbow arc for completion */}
      {showRainbow && <div className="rainbow-arc" />}
    </div>
  );
}

export default CelebrationEffects;
