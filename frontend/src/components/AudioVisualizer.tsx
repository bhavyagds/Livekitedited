import { motion } from 'framer-motion';

interface AudioVisualizerProps {
  isActive: boolean;
}

export function AudioVisualizer({ isActive }: AudioVisualizerProps) {
  const bars = Array.from({ length: 15 }, (_, i) => i);

  return (
    <div className={`audio-visualizer ${isActive ? 'active' : ''}`}>
      {bars.map((i) => (
        <motion.div
          key={i}
          className="visualizer-bar"
          animate={
            isActive
              ? {
                  height: [
                    '8px',
                    `${12 + Math.sin(i * 0.5) * 20}px`,
                    `${10 + Math.cos(i * 0.3) * 14}px`,
                    `${14 + Math.sin(i * 0.8) * 18}px`,
                    '8px'
                  ],
                }
              : { height: '8px' }
          }
          transition={{
            duration: 1.0,
            repeat: Infinity,
            repeatType: 'reverse',
            delay: i * 0.05,
            ease: 'easeInOut',
          }}
        />
      ))}
    </div>
  );
}

