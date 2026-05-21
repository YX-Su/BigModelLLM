import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The backend (FastAPI) enables permissive CORS, so the dev server can call it
// directly. Override the target with VITE_API_BASE when not on localhost:8000.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
  },
})
