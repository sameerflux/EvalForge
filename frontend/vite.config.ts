
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    allowedHosts: [
      'm-s-use4b0-1a2ldv0l4c3ag.us-east4-b.c.codatalab-user-runtimes.internal',
      'm-s-use4a3-2ju7loezjlylq.us-east4-a.c.codatalab-user-runtimes.internal',
      'm-s-usw4b1-3cd0lcl74abzz.us-west4-b.c.codatalab-user-runtimes.internal'
    ],
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true
      }
    }
  }
})
