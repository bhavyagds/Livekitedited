import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3001,
    host: '0.0.0.0',
    allowedHosts: ['admin.vakmedia.co', 'localhost', '127.0.0.1'],
    proxy: {
      '/api': {
        target: process.env.VITE_API_URL || 'http://api:8000',
        changeOrigin: true,
        rewrite: (path) => path,
      },
    },
  },
})
