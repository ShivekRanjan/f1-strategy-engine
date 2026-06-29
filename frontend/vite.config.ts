import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The API base is read at runtime from VITE_API_BASE (see src/api/client.ts),
// defaulting to the local uvicorn server. No dev proxy needed — the FastAPI
// layer enables CORS.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
});
