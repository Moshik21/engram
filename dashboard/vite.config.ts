/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes("node_modules")) return;

          if (id.includes("/three/examples/")) {
            return "three-extras";
          }

          if (
            id.includes("react-force-graph-3d") ||
            id.includes("3d-force-graph") ||
            id.includes("three-forcegraph") ||
            id.includes("three-render-objects") ||
            id.includes("three-spritetext") ||
            id.includes("d3-force-3d")
          ) {
            return "graph-3d-vendor";
          }

          if (id.includes("/three/")) {
            return "three-core";
          }
        },
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://localhost:8100", changeOrigin: true },
      "/health": { target: "http://localhost:8100", changeOrigin: true },
      "/ws": { target: "ws://localhost:8100", ws: true },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/test/setup.ts",
  },
});
