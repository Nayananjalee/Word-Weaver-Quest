import React from 'react';

/**
 * AngelCharacter - A realistic, beautiful guardian angel storyteller
 * Replaces the cartoon grandmother with an ethereal, glowing angel figure.
 * Features: flowing robes, luminous wings, gentle halo, divine glow, speaking animation
 * 
 * Props:
 *   - isSpeaking: boolean - whether she's currently telling the story
 */
function AngelCharacter({ isSpeaking = false }) {
  return (
    <div className="angel-container">
      {/* Divine light rays behind the angel */}
      <div className="angel-divine-rays">
        <div className="divine-ray" />
        <div className="divine-ray" />
        <div className="divine-ray" />
        <div className="divine-ray" />
        <div className="divine-ray" />
      </div>

      {/* Soft ambient glow */}
      <div className="angel-ambient-glow" />

      <div className="angel-body">
        {/* Speaking golden particles */}
        {isSpeaking && (
          <div className="angel-speaking-particles">
            <div className="angel-particle" />
            <div className="angel-particle" />
            <div className="angel-particle" />
            <div className="angel-particle" />
            <div className="angel-particle" />
          </div>
        )}

        {/* Halo - golden ring above head */}
        <div className="angel-halo" />

        {/* Wings - large, feathered, luminous */}
        <div className="angel-wings">
          <div className="angel-wing wing-left">
            <div className="wing-feather feather-1" />
            <div className="wing-feather feather-2" />
            <div className="wing-feather feather-3" />
            <div className="wing-feather feather-4" />
            <div className="wing-feather feather-5" />
          </div>
          <div className="angel-wing wing-right">
            <div className="wing-feather feather-1" />
            <div className="wing-feather feather-2" />
            <div className="wing-feather feather-3" />
            <div className="wing-feather feather-4" />
            <div className="wing-feather feather-5" />
          </div>
        </div>

        {/* Hair - long, flowing, golden */}
        <div className="angel-hair">
          <div className="hair-strand strand-left" />
          <div className="hair-strand strand-right" />
          <div className="hair-strand strand-back" />
        </div>

        {/* Head - realistic proportions, soft skin */}
        <div className="angel-head">
          {/* Eyes - large, warm, kind */}
          <div className="angel-eyes">
            <div className="angel-eye left-eye">
              <div className="angel-iris" />
              <div className="angel-pupil" />
              <div className="angel-eye-highlight" />
              <div className="angel-eyelash" />
            </div>
            <div className="angel-eye right-eye">
              <div className="angel-iris" />
              <div className="angel-pupil" />
              <div className="angel-eye-highlight" />
              <div className="angel-eyelash" />
            </div>
          </div>

          {/* Soft nose */}
          <div className="angel-nose" />

          {/* Gentle smile / speaking mouth */}
          <div className={`angel-mouth ${isSpeaking ? 'speaking' : ''}`} />

          {/* Rosy cheeks */}
          <div className="angel-cheek left-cheek" />
          <div className="angel-cheek right-cheek" />
        </div>

        {/* Neck */}
        <div className="angel-neck" />

        {/* Flowing white/golden robe */}
        <div className="angel-robe">
          <div className="robe-fold fold-1" />
          <div className="robe-fold fold-2" />
          <div className="robe-fold fold-3" />
          <div className="angel-sash" />
        </div>

        {/* Graceful hands */}
        <div className="angel-hand left-hand">
          <div className="hand-glow" />
        </div>
        <div className="angel-hand right-hand">
          <div className="hand-glow" />
          {/* Holding a magical book */}
          <div className="angel-book">
            <div className="book-pages" />
            <div className="book-glow" />
          </div>
        </div>
      </div>

      {/* Floating sparkles around the angel */}
      <div className="angel-sparkles">
        <div className="angel-sparkle" style={{ top: '10%', left: '15%', animationDelay: '0s' }}>✦</div>
        <div className="angel-sparkle" style={{ top: '30%', right: '10%', animationDelay: '0.5s' }}>✧</div>
        <div className="angel-sparkle" style={{ top: '60%', left: '5%', animationDelay: '1s' }}>✦</div>
        <div className="angel-sparkle" style={{ top: '20%', left: '80%', animationDelay: '1.5s' }}>✧</div>
        <div className="angel-sparkle" style={{ top: '70%', right: '15%', animationDelay: '2s' }}>✦</div>
      </div>
    </div>
  );
}

export default AngelCharacter;
