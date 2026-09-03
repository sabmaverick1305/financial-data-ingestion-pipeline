import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev-server proxy so the frontend always calls relative paths (/api/ask,
// /healthz, /readyz) in both dev and prod — no base-URL env var, no CORS
// setup needed locally. In prod the same paths are served same-origin by
// FastAPI's StaticFiles mount (see api/main.py), so nothing changes there.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:8080',
      '/healthz': 'http://localhost:8080',
      '/readyz': 'http://localhost:8080',
    },
  },
})
