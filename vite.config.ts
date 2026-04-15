import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

declare const process: { env: Record<string, string | undefined> };

// Stub all @tauri-apps/api/* imports so the React web build works without
// Tauri installed. Dynamic imports in IntentBuilder are guarded by IS_TAURI
// (checks window.__TAURI_INTERNALS__) so these stubs are never called at
// runtime -- but we still need them to satisfy the bundler.
const tauriStubPlugin = () => ({
    name: "tauri-stub",
    resolveId(id: string) {
        if (id.startsWith("@tauri-apps/api/") && !process.env.TAURI_PLATFORM) {
            return `\0tauri-stub:${id}`;
        }
        return null;
    },
    load(id: string) {
        if (id.startsWith("\0tauri-stub:@tauri-apps/api/")) {
            // Generic stub: every named export rejects so runtime errors are
            // obvious if the IS_TAURI guard is ever bypassed accidentally.
            return `
export const invoke = () => Promise.reject(new Error('Tauri not available'));
export const listen = () => Promise.reject(new Error('Tauri not available'));
export const open   = () => Promise.reject(new Error('Tauri not available'));
export const readTextFile = () => Promise.reject(new Error('Tauri not available'));
export default {};
`;
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
        host: "0.0.0.0",
        hmr: { protocol: "ws", host: "localhost", port: 1421 },
    },
    optimizeDeps: {
        exclude: [],
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
