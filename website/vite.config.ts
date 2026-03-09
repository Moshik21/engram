import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 4001 },
  build: {
    // The homepage 3D scene now ships as a deferred route-level chunk.
    // Keep warning noise focused on the entry bundle rather than this optional asset.
    chunkSizeWarningLimit: 900,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("@react-three") || id.includes("/three/") || id.includes("three/examples")) {
            return "three-vendor";
          }
          if (id.includes("react-router-dom")) {
            return "router-vendor";
          }
          if (id.includes("/react/") || id.includes("/react-dom/")) {
            return "react-vendor";
          }
          return undefined;
        },
      },
    },
  },
});
