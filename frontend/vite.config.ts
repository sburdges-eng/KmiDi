import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

// @ts-expect-error process is a nodejs global
const host = process.env.TAURI_DEV_HOST;

// Plugin to handle Tauri imports in browser mode
const tauriStubPlugin = () => ({
  name: "tauri-stub",
  resolveId(id: string) {
    // Only stub in browser mode (when not in Tauri)
    if (id.startsWith("@tauri-apps/api/") && !process.env.TAURI_PLATFORM) {
      // Return a virtual module that exports stubs
      if (id === "@tauri-apps/api/dialog") {
        return "\0tauri-dialog-stub";
      }
      if (id === "@tauri-apps/api/fs") {
        return "\0tauri-fs-stub";
      }
    }
    return null;
  },
  load(id: string) {
    if (id === "\0tauri-dialog-stub") {
      return "export const open = () => Promise.reject(new Error('Tauri not available'));";
    }
    if (id === "\0tauri-fs-stub") {
      return "export const readTextFile = () => Promise.reject(new Error('Tauri not available'));";
    }
    return null;
  },
});

// https://vite.dev/config/
export default defineConfig(async () => ({
  plugins: [react(), tauriStubPlugin()],

  // Use project root - index.html will be moved here
  root: path.resolve(__dirname),

  // Public assets are in the root public folder
  publicDir: path.resolve(__dirname, "public"),

// Vite options kept for latent/historical Tauri compatibility surfaces.
// They matter only if a Tauri dev/build path is deliberately restored.
//
// 1. prevent Vite from obscuring rust errors on that historical path
  clearScreen: false,
  // 2. the historical Tauri path expects a fixed port, fail if that port is not available
  server: {
    port: 1420,
    strictPort: true,
    host: host || "0.0.0.0", // Allow network access (defaults to 0.0.0.0 if not in Tauri mode)
    hmr: host
      ? {
        protocol: "ws",
        host,
        port: 1421,
      }
      : {
        protocol: "ws",
        host: "localhost",
        port: 1421,
      },
    watch: {
      // 3. tell Vite to ignore watching `src-tauri` for that compatibility path
      ignored: ["**/src-tauri/**"],
    },
  },

  optimizeDeps: {
    exclude: ["@tauri-apps/api/dialog", "@tauri-apps/api/fs"],
  },
  build: {
    outDir: path.resolve(__dirname, "dist"),
    chunkSizeWarningLimit: 900,
    rollupOptions: {
      output: {
        manualChunks: {
          react: ["react", "react-dom"],
          markdown: ["react-markdown", "remark-gfm"],
        },
      },
    },
  },
}));
