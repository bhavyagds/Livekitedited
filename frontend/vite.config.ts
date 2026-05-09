import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
// API base is updated to the FastAPI backend.
const API_TARGET = process.env.VITE_API_URL || 'http://localhost:8000'
const WS_TARGET  = process.env.VITE_WS_URL || 'ws://localhost:8000'


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
