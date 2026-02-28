import React from 'react';

/**
 * GrandmotherCharacter - Animated CSS grandmother in a rocking chair
 * She sways gently, blinks, and her mouth animates when speaking.
 * Props:
 *   - isSpeaking: boolean - whether she's currently telling the story
 */
function GrandmotherCharacter({ isSpeaking = false }) {
  return (
    <div className="grandmother-container">
      <div className="grandmother-body">
        {/* Speaking indicator dots */}
        {isSpeaking && (
          <div className="grandma-speaking-indicator">
            <div className="speak-dot" />
            <div className="speak-dot" />
            <div className="speak-dot" />
          </div>
        )}

        {/* Hair bun */}
        <div className="grandma-bun" />
        <div className="grandma-hair" />

        {/* Head */}
        <div className="grandma-head">
          {/* Glasses */}
          <div className="grandma-glasses">
            <div className="glasses-lens left" />
            <div className="glasses-bridge" />
            <div className="glasses-lens right" />
          </div>

          {/* Eyes */}
          <div className="grandma-eyes">
            <div className="grandma-eye" />
            <div className="grandma-eye" />
          </div>

          {/* Smile / Mouth */}
          <div className={`grandma-smile ${isSpeaking ? 'speaking' : ''}`} />
        </div>

        {/* Shawl */}
        <div className="grandma-shawl" />

        {/* Body */}
        <div className="grandma-body-shape" />

        {/* Hands with gestures */}
        <div className="grandma-hand left" />
        <div className="grandma-hand right" />

        {/* Rocking Chair */}
        <div className="rocking-chair">
          <div className="chair-back" />
          <div className="chair-seat" />
          <div className="chair-rocker" />
        </div>
      </div>
    </div>
  );
}

export default GrandmotherCharacter;
