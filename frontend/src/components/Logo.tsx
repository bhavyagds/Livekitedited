import { motion } from 'framer-motion';

interface LogoProps {
  size?: 'small' | 'large';
}

export function Logo({ size = 'large' }: LogoProps) {
  const isSmall = size === 'small';

  return (
    <motion.div
      className={`logo-container ${isSmall ? 'small' : ''}`}
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      <h1 className="logo-text">MEALLION</h1>
      {!isSmall && <p className="logo-tagline">Voice Assistant</p>}
    </motion.div>
  );
}
