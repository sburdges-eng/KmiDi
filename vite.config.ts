// @ts-nocheck
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

// @ts-expect-error process is a nodejs global
const host = process.env.TAURI_DEV_HOST;

// Plugin to handle Tauri imports in browser mode
const tauriStubPlugin = () => ({
    name: "tauri-stub",
    resolveId(id: string) {
        if (id.startsWith("@tauri-apps/api/") && !process.env.TAURI_PLATFORM) {
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

export default defineConfig(async () => ({
    plugins: [react(), tauriStubPlugin()],
    root: path.resolve(__dirname),
    publicDir: path.resolve(__dirname, "public"),
    clearScreen: false,
    server: {
        port: 1420,
        strictPort: true,
        host: host || "0.0.0.0",
        hmr: host
            ? { protocol: "ws", host, port: 1421 }
            : { protocol: "ws", host: "localhost", port: 1421 },
        watch: {
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
