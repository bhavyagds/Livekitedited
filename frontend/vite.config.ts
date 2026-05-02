import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
// API base is injected at container start-time via VITE_API_URL env var.
// Falling back to 'http://meallion-api:8000' (explicit container name) instead of
// the short 'api' alias, which can fail DNS resolution after container restarts.
const API_TARGET = process.env.VITE_API_URL || 'http://meallion-api:8000'
const WS_TARGET  = process.env.VITE_WS_URL  || 'ws://meallion-api:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 3000,
    strictPort: true,
    proxy: {
      // Proxy API requests to the FastAPI backend
      '/api': {
        target: API_TARGET,
        changeOrigin: true,
      },
      '/webhook': {
        target: API_TARGET,
        changeOrigin: true,
      },
      '/ws': {
        target: WS_TARGET,
        ws: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
})
