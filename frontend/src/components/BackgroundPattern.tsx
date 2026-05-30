import { useMemo } from 'react';

export function BackgroundPattern() {
  // Generate 60 stars with stable random positions and twinkle delays
  const stars = useMemo(() => {
    return Array.from({ length: 60 }, (_, i) => ({
      id: i,
      cx: `${(i * 17) % 100}%`,
      cy: `${(i * 23) % 100}%`,
      r: (i % 3 === 0) ? 1.5 : (i % 2 === 0) ? 1.0 : 0.6,
      delay: `${(i * 0.13) % 5}s`,
      duration: `${3 + (i % 4)}s`,
    }));
  }, []);

  return (
    <div className="background-pattern">
      {/* Premium Ambient Nebula Clouds */}
      <div className="nebula-cloud nebula-blue" />
      <div className="nebula-cloud nebula-purple" />
      <div className="nebula-cloud nebula-gold" />
      
      {/* Twinkling Starfield */}
      <svg className="starfield" width="100%" height="100%">
        {stars.map((star) => (
          <circle
            key={star.id}
            cx={star.cx}
            cy={star.cy}
            r={star.r}
            fill="#ffffff"
            className="star"
            style={{
              animationDelay: star.delay,
              animationDuration: star.duration,
            }}
          />
        ))}
      </svg>
      
      <div className="gradient-orb orb-1" />
      <div className="gradient-orb orb-2" />
      <div className="gradient-orb orb-3" />
      <div className="noise-overlay" />
    </div>
  );
}

