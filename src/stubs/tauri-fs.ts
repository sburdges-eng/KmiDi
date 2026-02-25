// Stub for @tauri-apps/api/fs when running in browser mode
export const readTextFile = async (path: string) => {
  throw new Error("Tauri FS API not available in browser mode");
};
