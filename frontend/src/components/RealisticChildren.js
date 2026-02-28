import React, { useState, useEffect } from 'react';

/**
 * RealisticChildren - Beautiful, realistic children sitting in a magical garden
 * They have detailed features, natural skin tones, expressive faces.
 * 
 * Props:
 *   - reaction: 'listening' | 'excited' | 'sad' | 'thinking'
 *   - showThoughts: boolean - show thought emojis above heads
 */
function RealisticChildren({ reaction = 'listening', showThoughts = false }) {
  const [thoughts, setThoughts] = useState(['', '', '']);

  useEffect(() => {
    if (reaction === 'excited') {
      setThoughts(['🎉', '⭐', '🥳']);
    } else if (reaction === 'sad') {
      setThoughts(['😢', '💪', '🤔']);
    } else if (reaction === 'thinking') {
      setThoughts(['🤔', '💭', '🧐']);
    } else {
      setThoughts(['', '', '']);
    }
  }, [reaction]);

  const children = [
    { 
      skinClass: 'realistic-skin-1', 
      hairClass: 'realistic-hair-1', 
      bodyClass: 'outfit-1', 
      name: 'child-1',
      hairColor: 'dark-brown',
      accessory: 'bow'
    },
    { 
      skinClass: 'realistic-skin-2', 
      hairClass: 'realistic-hair-2', 
      bodyClass: 'outfit-2', 
      name: 'child-2',
      hairColor: 'black',
      accessory: 'cap'
    },
    { 
      skinClass: 'realistic-skin-3', 
      hairClass: 'realistic-hair-3', 
      bodyClass: 'outfit-3', 
      name: 'child-3',
      hairColor: 'light-brown',
      accessory: 'band'
    },
  ];

  return (
    <div className="realistic-children-container">
      {/* Garden bench / sitting area */}
      <div className="children-sitting-area">
        <div className="sitting-bench">
          <div className="bench-top" />
          <div className="bench-leg bench-leg-left" />
          <div className="bench-leg bench-leg-right" />
        </div>
      </div>

      <div className="children-row">
        {children.map((child, index) => (
          <div
            key={child.name}
            className={`realistic-child ${reaction === 'excited' ? 'child-excited' : ''} ${reaction === 'thinking' ? 'child-thinking' : ''}`}
            style={{ animationDelay: `${index * 0.3}s` }}
          >
            {/* Thought bubble */}
            {showThoughts && thoughts[index] && (
              <div className="realistic-thought-bubble">
                <span className="thought-emoji">{thoughts[index]}</span>
                <div className="thought-dots">
                  <span className="thought-dot" />
                  <span className="thought-dot" />
                </div>
              </div>
            )}

            {/* Full child figure */}
            <div className="child-figure">
              {/* Hair */}
              <div className={`realistic-child-hair ${child.hairClass} ${child.hairColor}`}>
                {child.accessory === 'bow' && <div className="hair-bow" />}
                {child.accessory === 'cap' && <div className="hair-cap" />}
                {child.accessory === 'band' && <div className="hair-band" />}
              </div>

              {/* Head with detailed face */}
              <div className={`realistic-child-head ${child.skinClass}`}>
                {/* Forehead shadow */}
                <div className="head-shadow" />

                {/* Eyebrows */}
                <div className="child-eyebrows">
                  <div className={`child-eyebrow left-brow ${reaction === 'excited' ? 'brow-raised' : ''}`} />
                  <div className={`child-eyebrow right-brow ${reaction === 'excited' ? 'brow-raised' : ''}`} />
                </div>

                {/* Eyes with detail */}
                <div className="realistic-child-eyes">
                  <div className="realistic-child-eye">
                    <div className="eye-white" />
                    <div className="eye-iris" />
                    <div className="eye-pupil" />
                    <div className="eye-shine" />
                    <div className="eyelid" />
                  </div>
                  <div className="realistic-child-eye">
                    <div className="eye-white" />
                    <div className="eye-iris" />
                    <div className="eye-pupil" />
                    <div className="eye-shine" />
                    <div className="eyelid" />
                  </div>
                </div>

                {/* Nose */}
                <div className="child-nose-realistic" />

                {/* Mouth */}
                <div className={`realistic-child-mouth ${reaction === 'excited' ? 'mouth-smile' : ''} ${reaction === 'sad' ? 'mouth-sad' : ''}`} />

                {/* Cheeks - rosy when excited */}
                <div className={`child-cheek left ${reaction === 'excited' ? 'cheek-flush' : ''}`} />
                <div className={`child-cheek right ${reaction === 'excited' ? 'cheek-flush' : ''}`} />

                {/* Ears */}
                <div className={`child-ear left-ear ${child.skinClass}`} />
                <div className={`child-ear right-ear ${child.skinClass}`} />
              </div>

              {/* Neck */}
              <div className={`child-neck ${child.skinClass}`} />

              {/* Body / Outfit */}
              <div className={`realistic-child-body ${child.bodyClass}`}>
                <div className="outfit-detail" />
                <div className="outfit-collar" />
              </div>

              {/* Arms */}
              <div className={`child-arm left-arm ${child.skinClass} ${reaction === 'excited' ? 'arm-raised' : ''}`}>
                <div className="child-hand-realistic" />
              </div>
              <div className={`child-arm right-arm ${child.skinClass}`}>
                <div className="child-hand-realistic" />
              </div>

              {/* Legs */}
              <div className="child-legs">
                <div className={`child-leg ${child.bodyClass}-leg`} />
                <div className={`child-leg ${child.bodyClass}-leg`} />
              </div>

              {/* Shoes */}
              <div className="child-shoes">
                <div className="child-shoe left-shoe" />
                <div className="child-shoe right-shoe" />
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default RealisticChildren;
