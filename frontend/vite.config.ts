import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import pkg from "./package.json";

// The API base is read at runtime from VITE_API_BASE (see src/api/client.ts),
// defaulting to the local uvicorn server. No dev proxy needed — the FastAPI
// layer enables CORS.
//
// Build identity: the app footer shows "v<version> · <sha>". Vercel exposes
// the commit as VERCEL_GIT_COMMIT_SHA at build time; local dev shows "dev".
const sha = (process.env.VERCEL_GIT_COMMIT_SHA ?? "dev").slice(0, 7);

export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
  define: {
    __APP_VERSION__: JSON.stringify(pkg.version),
    __BUILD_SHA__: JSON.stringify(sha),
  },
});
