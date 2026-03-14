import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 3000,
    strictPort: true,
    proxy: {
      // Proxy API requests to the FastAPI backend
      '/api': {
        target: 'http://api:8000',
        changeOrigin: true,
      },
      '/webhook': {
        target: 'http://api:8000',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://api:8000',
        ws: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
})
