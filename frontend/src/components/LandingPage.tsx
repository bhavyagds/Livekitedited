import { motion } from 'framer-motion';
import { Phone, Loader2, AlertCircle } from 'lucide-react';
import { Logo } from './Logo';
import { BackgroundPattern } from './BackgroundPattern';

interface LandingPageProps {
  onConnect: () => void;
  isConnecting: boolean;
  error: string | null;
}

export function LandingPage({ onConnect, isConnecting, error }: LandingPageProps) {
  return (
    <div className="landing-page">
      <BackgroundPattern />
      
      <motion.div
        className="landing-content"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
      >
        <Logo />
        
        <motion.div
          className="hero-section"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3, duration: 0.8 }}
        >
          <h2 className="hero-title">Speak with Elena</h2>
          <p className="hero-subtitle">
            Your personal AI assistant for Meallion.
            <br />
            Order tracking, support, and more.
          </p>
        </motion.div>

        <motion.div
          className="cta-section"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.6, duration: 0.5 }}
        >
          <button
            className="connect-button"
            onClick={onConnect}
            disabled={isConnecting}
          >
            {isConnecting ? (
              <>
                <Loader2 className="button-icon spinning" />
                <span>Connecting...</span>
              </>
            ) : (
              <>
                <Phone className="button-icon" />
                <span>Start Conversation</span>
              </>
            )}
          </button>

          {error && (
            <motion.div
              className="error-message"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <AlertCircle size={16} />
              <span>{error}</span>
            </motion.div>
          )}
        </motion.div>

        <motion.div
          className="features-section"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.9, duration: 0.8 }}
        >
          <div className="feature">
            <span className="feature-icon">📦</span>
            <span className="feature-text">Track Orders</span>
          </div>
          <div className="feature">
            <span className="feature-icon">🎫</span>
            <span className="feature-text">Get Support</span>
          </div>
          <div className="feature">
            <span className="feature-icon">❓</span>
            <span className="feature-text">Ask Questions</span>
          </div>
        </motion.div>
      </motion.div>

      <footer className="landing-footer">
        <p>Powered by Meallion • Chef Lambros Vakiaros</p>
      </footer>
    </div>
  );
}
