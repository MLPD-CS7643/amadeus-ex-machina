/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173, // matches the backend CORS allow-list
    proxy: {
      // Optional convenience: proxy /api to the backend during dev so the app
      // works even if VITE_API_BASE_URL is left empty.
      "/api": {
        // Explicit IPv4 so the proxy matches uvicorn's default 127.0.0.1 bind
        // (avoids localhost -> ::1 resolution mismatches). Start the API with
        // `just api` (uvicorn on :8000).
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts",
    css: false,
  },
});
