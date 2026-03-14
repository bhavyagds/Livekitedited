import { motion } from 'framer-motion';

interface AudioVisualizerProps {
  isActive: boolean;
}

export function AudioVisualizer({ isActive }: AudioVisualizerProps) {
  const bars = Array.from({ length: 5 }, (_, i) => i);

  return (
    <div className={`audio-visualizer ${isActive ? 'active' : ''}`}>
      {bars.map((i) => (
        <motion.div
          key={i}
          className="visualizer-bar"
          animate={
            isActive
              ? {
                  height: ['8px', '32px', '16px', '28px', '8px'],
                }
              : { height: '8px' }
          }
          transition={{
            duration: 0.8,
            repeat: Infinity,
            repeatType: 'reverse',
            delay: i * 0.1,
            ease: 'easeInOut',
          }}
        />
      ))}
    </div>
  );
}
