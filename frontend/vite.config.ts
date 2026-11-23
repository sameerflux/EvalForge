
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    allowedHosts: [
      'm-s-use4a3-2ju7loezjlylq.us-east4-a.c.codatalab-user-runtimes.internal'
    ]
  }
})
