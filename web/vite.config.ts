import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig({
  plugins: [react()],
  server: {
    port: 18001,
    proxy: {
      "/api": "http://127.0.0.1:18000",
      "/auth": "http://127.0.0.1:18000",
    },
  },
  build: { sourcemap: false },
});
