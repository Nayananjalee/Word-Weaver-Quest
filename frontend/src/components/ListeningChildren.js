import React, { useState, useEffect } from 'react';

/**
 * ListeningChildren - Animated children sitting and listening to the story
 * They react to correct/wrong answers and show emotions.
 * Props:
 *   - reaction: 'listening' | 'excited' | 'sad' | 'thinking'
 *   - showThoughts: boolean - show thought emojis above heads
 */
function ListeningChildren({ reaction = 'listening', showThoughts = false }) {
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
    { skinClass: 'skin-1', hairClass: 'hair-1', bodyClass: 'color-1', name: 'child-1' },
    { skinClass: 'skin-2', hairClass: 'hair-2', bodyClass: 'color-2', name: 'child-2' },
    { skinClass: 'skin-3', hairClass: 'hair-3', bodyClass: 'color-3', name: 'child-3' },
  ];

  return (
    <div className="children-container">
      {children.map((child, index) => (
        <div
          key={child.name}
          className={`child-character ${reaction === 'excited' ? 'excited' : ''}`}
        >
          {/* Thought bubble */}
          {showThoughts && thoughts[index] && (
            <div className="child-thought">{thoughts[index]}</div>
          )}

          {/* Head */}
          <div className={`child-head ${child.skinClass}`}>
            <div className={`child-hair ${child.hairClass}`} />
            <div className="child-eyes">
              <div className="child-eye" />
              <div className="child-eye" />
            </div>
            <div className={`child-mouth ${reaction === 'excited' ? 'excited' : ''}`} />
          </div>

          {/* Body */}
          <div className={`child-body ${child.bodyClass}`} />
        </div>
      ))}
    </div>
  );
}

export default ListeningChildren;
