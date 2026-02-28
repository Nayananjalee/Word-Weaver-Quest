import React, { useMemo } from 'react';

/**
 * MagicalParticles - Golden light particles, floating petals, and divine sparkles
 * Creates an ethereal, heavenly atmosphere.
 * Pure CSS-driven particles for performance.
 */
function MagicalParticles() {
  // Golden light orbs floating in the air
  const goldenOrbs = useMemo(() => {
    const items = [];
    for (let i = 0; i < 18; i++) {
      items.push({
        id: i,
        top: 10 + Math.random() * 70,
        left: Math.random() * 100,
        duration: 5 + Math.random() * 8,
        delay: Math.random() * 6,
        driftX: -25 + Math.random() * 50,
        driftY: -35 + Math.random() * 35,
        size: 3 + Math.random() * 5,
      });
    }
    return items;
  }, []);

  // Floating flower petals
  const petals = useMemo(() => {
    const items = [];
    const colors = ['#ffb6c1', '#ffd1dc', '#fff0f5', '#ffe4e1', '#ffc0cb'];
    for (let i = 0; i < 10; i++) {
      items.push({
        id: i,
        left: Math.random() * 100,
        delay: Math.random() * 8,
        duration: 8 + Math.random() * 7,
        color: colors[Math.floor(Math.random() * colors.length)],
        size: 6 + Math.random() * 6,
        drift: -40 + Math.random() * 80,
      });
    }
    return items;
  }, []);

  // Divine sparkle dust
  const dustMotes = useMemo(() => {
    const items = [];
    for (let i = 0; i < 12; i++) {
      items.push({
        id: i,
        top: Math.random() * 80,
        left: Math.random() * 100,
        delay: Math.random() * 4,
        size: 2 + Math.random() * 3,
      });
    }
    return items;
  }, []);

  return (
    <>
      {/* Golden light orbs */}
      <div className="golden-orbs-container">
        {goldenOrbs.map((f) => (
          <div
            key={f.id}
            className="golden-orb"
            style={{
              top: `${f.top}%`,
              left: `${f.left}%`,
              width: `${f.size}px`,
              height: `${f.size}px`,
              '--duration': `${f.duration}s`,
              '--drift-x': `${f.driftX}px`,
              '--drift-y': `${f.driftY}px`,
              animationDelay: `${f.delay}s`,
            }}
          />
        ))}
      </div>

      {/* Floating petals */}
      <div className="petals-container">
        {petals.map((p) => (
          <div
            key={p.id}
            className="floating-petal"
            style={{
              left: `${p.left}%`,
              width: `${p.size}px`,
              height: `${p.size * 1.3}px`,
              background: p.color,
              animationDelay: `${p.delay}s`,
              animationDuration: `${p.duration}s`,
              '--petal-drift': `${p.drift}px`,
            }}
          />
        ))}
      </div>

      {/* Divine sparkle dust */}
      {dustMotes.map((d) => (
        <div
          key={d.id}
          className="divine-dust"
          style={{
            top: `${d.top}%`,
            left: `${d.left}%`,
            width: `${d.size}px`,
            height: `${d.size}px`,
            animationDelay: `${d.delay}s`,
          }}
        />
      ))}
    </>
  );
}

export default MagicalParticles;
