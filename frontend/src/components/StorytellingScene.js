import React, { useMemo } from 'react';
import AngelCharacter from './AngelCharacter';
import RealisticChildren from './RealisticChildren';
import MagicalParticles from './MagicalParticles';
import CelebrationEffects from './CelebrationEffects';
import './StorytellingScene.css';

/**
 * StorytellingScene - An enchanted heavenly garden storytelling experience
 * 
 * Renders a realistic, beautiful scene with:
 * - Dreamy golden-hour sky with clouds and sun rays
 * - Lush realistic meadow with flowers and butterflies
 * - Majestic trees with sunlight filtering through leaves
 * - Beautiful guardian angel storyteller (with wings & halo)
 * - Realistic children listening in a magical garden
 * - Golden light particles, butterflies, petals
 * - Speech bubble for story text
 * - Answer cards at the bottom
 * - Celebration effects
 * 
 * Props:
 *   - isSpeaking: angel is telling a story
 *   - childrenReaction: how children react
 *   - celebrationType: celebration animation type
 *   - children: React children (content to render inside scene)
 */
function StorytellingScene({ 
  isSpeaking = false, 
  childrenReaction = 'listening', 
  celebrationType = null,
  storyProgress = 0,
  earnedStars = 0,
  totalQuestions = 0,
  currentSentence = 0,
  totalSentences = 0,
  children 
}) {
  // Generate floating clouds
  const clouds = useMemo(() => {
    const items = [];
    for (let i = 0; i < 8; i++) {
      items.push({
        id: i,
        top: 5 + Math.random() * 25,
        left: -10 + Math.random() * 110,
        delay: Math.random() * 15,
        scale: 0.5 + Math.random() * 1,
        duration: 25 + Math.random() * 20,
        opacity: 0.3 + Math.random() * 0.5,
      });
    }
    return items;
  }, []);

  // Generate golden light rays
  const sunRays = useMemo(() => {
    const items = [];
    for (let i = 0; i < 6; i++) {
      items.push({
        id: i,
        angle: -15 + (i * 8),
        delay: i * 0.5,
        opacity: 0.08 + Math.random() * 0.07,
      });
    }
    return items;
  }, []);

  // Generate flowers in meadow
  const flowers = useMemo(() => {
    const types = ['🌸', '🌺', '🌷', '🌼', '💐', '🌻', '🪻'];
    const items = [];
    for (let i = 0; i < 12; i++) {
      items.push({
        id: i,
        emoji: types[Math.floor(Math.random() * types.length)],
        left: 2 + Math.random() * 96,
        bottom: Math.random() * 18,
        size: 12 + Math.random() * 14,
        delay: Math.random() * 3,
      });
    }
    return items;
  }, []);

  // Generate butterflies
  const butterflies = useMemo(() => {
    const items = [];
    for (let i = 0; i < 5; i++) {
      items.push({
        id: i,
        emoji: ['🦋', '🦋', '🦋'][Math.floor(Math.random() * 3)],
        top: 15 + Math.random() * 50,
        left: 10 + Math.random() * 80,
        delay: Math.random() * 8,
        duration: 8 + Math.random() * 6,
      });
    }
    return items;
  }, []);

  // Generate realistic trees
  const trees = useMemo(() => [
    { id: 1, left: '-3%', type: 'oak', scale: 1.1, delay: 0 },
    { id: 2, left: '88%', type: 'willow', scale: 1.0, delay: 2 },
    { id: 3, left: '96%', type: 'birch', scale: 0.75, delay: 4 },
    { id: 4, left: '-5%', type: 'maple', scale: 0.85, delay: 1 },
  ], []);

  return (
    <div className="storytelling-scene realistic-scene">
      {/* Dreamy sky background - golden hour */}
      <div className="scene-background realistic-sky" />
      
      {/* Sun glow in upper right */}
      <div className="realistic-sun">
        <div className="sun-core" />
        <div className="sun-corona" />
      </div>

      {/* God rays / sunlight beams */}
      <div className="sun-rays-container">
        {sunRays.map((ray) => (
          <div
            key={ray.id}
            className="sun-ray"
            style={{
              transform: `rotate(${ray.angle}deg)`,
              animationDelay: `${ray.delay}s`,
              opacity: ray.opacity,
            }}
          />
        ))}
      </div>

      {/* Floating clouds */}
      <div className="clouds-container">
        {clouds.map((cloud) => (
          <div
            key={cloud.id}
            className="realistic-cloud"
            style={{
              top: `${cloud.top}%`,
              left: `${cloud.left}%`,
              transform: `scale(${cloud.scale})`,
              animationDelay: `${cloud.delay}s`,
              animationDuration: `${cloud.duration}s`,
              opacity: cloud.opacity,
            }}
          >
            <div className="cloud-puff cloud-puff-1" />
            <div className="cloud-puff cloud-puff-2" />
            <div className="cloud-puff cloud-puff-3" />
          </div>
        ))}
      </div>

      {/* Realistic trees */}
      <div className="scene-trees realistic-trees">
        {trees.map((tree) => (
          <div
            key={tree.id}
            className={`realistic-tree tree-${tree.type}`}
            style={{
              left: tree.left,
              transform: `scale(${tree.scale})`,
              animationDelay: `${tree.delay}s`,
            }}
          >
            <div className="tree-canopy">
              <div className="canopy-layer canopy-1" />
              <div className="canopy-layer canopy-2" />
              <div className="canopy-layer canopy-3" />
              <div className="tree-sunlight-filter" />
            </div>
            <div className="realistic-trunk">
              <div className="trunk-detail" />
              <div className="trunk-roots" />
            </div>
          </div>
        ))}
      </div>

      {/* Lush meadow ground */}
      <div className="scene-ground realistic-meadow">
        <div className="meadow-grass-overlay" />
        <div className="grass-blades">
          {[...Array(30)].map((_, i) => (
            <div 
              key={i} 
              className="grass-blade" 
              style={{ 
                left: `${(i / 30) * 100}%`,
                height: `${8 + Math.random() * 12}px`,
                animationDelay: `${Math.random() * 2}s`
              }} 
            />
          ))}
        </div>
      </div>

      {/* Meadow flowers */}
      <div className="meadow-flowers">
        {flowers.map((f) => (
          <div
            key={f.id}
            className="meadow-flower"
            style={{
              left: `${f.left}%`,
              bottom: `${f.bottom}%`,
              fontSize: `${f.size}px`,
              animationDelay: `${f.delay}s`,
            }}
          >
            {f.emoji}
          </div>
        ))}
      </div>

      {/* Butterflies */}
      <div className="butterflies-container">
        {butterflies.map((b) => (
          <div
            key={b.id}
            className="butterfly"
            style={{
              top: `${b.top}%`,
              left: `${b.left}%`,
              animationDelay: `${b.delay}s`,
              animationDuration: `${b.duration}s`,
            }}
          >
            {b.emoji}
          </div>
        ))}
      </div>

      {/* Golden particles and magical atmosphere */}
      <MagicalParticles />

      {/* Angel storyteller */}
      <AngelCharacter isSpeaking={isSpeaking} />

      {/* Realistic children */}
      <RealisticChildren
        reaction={childrenReaction}
        showThoughts={childrenReaction !== 'listening'}
      />

      {/* Progress bar at top */}
      <div className="story-progress-bar">
        <div
          className="story-progress-fill"
          style={{ width: `${storyProgress}%` }}
        />
      </div>

      {/* Stars counter */}
      <div className="stars-counter">
        <span className="stars-counter-icon">⭐</span>
        <span className="stars-counter-text">
          {earnedStars} / {totalQuestions}
        </span>
      </div>

      {/* Sentence counter */}
      <div className="sentence-counter">
        📖 {currentSentence} / {totalSentences}
      </div>

      {/* Celebration effects overlay */}
      <CelebrationEffects type={celebrationType} />

      {/* Dynamic content (bubbles, answers, etc.) */}
      {children}
    </div>
  );
}

export default StorytellingScene;
