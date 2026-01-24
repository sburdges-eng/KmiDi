/**
 * Persistence utilities using Tauri store API
 *
 * Provides functions to save and restore application state including:
 * - Panel visibility states
 * - Panel sizes/positions
 * - Window preferences
 * - Side A/B preference
 *
 * Per Spec 02: Layout & Navigation - Persistence requirements
 */

// Store interface for compatibility
interface Store {
  get<T>(key: string): Promise<T | null>;
  set(key: string, value: any): Promise<void>;
  has(key: string): Promise<boolean>;
  delete(key: string): Promise<boolean>;
  clear(): Promise<void>;
  keys(): Promise<string[]>;
  save(): Promise<void>;
}

// Store instance (lazy-loaded)
let storeInstance: Store | null = null;

/**
 * Get or create the store instance (Tauri store or localStorage fallback)
 */
async function getStore(): Promise<Store> {
  if (!storeInstance) {
    try {
      // Try to use Tauri store plugin if available
      const { Store } = await import("@tauri-apps/plugin-store");
      storeInstance = new Store(".settings.dat") as Store;
    } catch (error) {
      console.warn(
        "Tauri store plugin not available, using localStorage fallback:",
        error,
      );
      // Return a localStorage-based store
      storeInstance = createLocalStorageStore();
    }
  }
  return storeInstance;
}

/**
 * Create a localStorage-based fallback store
 */
function createLocalStorageStore(): Store {
  const prefix = "kmidi-";

  return {
    async get<T>(key: string): Promise<T | null> {
      try {
        const value = localStorage.getItem(prefix + key);
        return value ? JSON.parse(value) : null;
      } catch {
        return null;
      }
    },
    async set(key: string, value: any): Promise<void> {
      try {
        localStorage.setItem(prefix + key, JSON.stringify(value));
      } catch (error) {
        console.warn("Failed to save to localStorage:", error);
      }
    },
    async has(key: string): Promise<boolean> {
      return localStorage.getItem(prefix + key) !== null;
    },
    async delete(key: string): Promise<boolean> {
      try {
        localStorage.removeItem(prefix + key);
        return true;
      } catch {
        return false;
      }
    },
    async clear(): Promise<void> {
      // Only clear our prefixed keys
      const keys = Object.keys(localStorage).filter((k) =>
        k.startsWith(prefix),
      );
      keys.forEach((k) => localStorage.removeItem(k));
    },
    async keys(): Promise<string[]> {
      return Object.keys(localStorage)
        .filter((k) => k.startsWith(prefix))
        .map((k) => k.substring(prefix.length));
    },
    async save(): Promise<void> {
      // localStorage is automatically saved
    },
  };
}

/**
 * Panel state interface
 */
export interface PanelState {
  inspector: {
    visible: boolean;
    width?: number;
  };
  timeline: {
    visible: boolean;
  };
  browser: {
    visible: boolean;
    width?: number;
  };
}

/**
 * Window preferences interface
 */
export interface WindowPreferences {
  width: number;
  height: number;
  x?: number;
  y?: number;
  maximized?: boolean;
  sideA: boolean; // Side A/B preference
}

/**
 * Save panel state
 */
export async function savePanelState(state: PanelState): Promise<void> {
  try {
    const store = await getStore();
    await store.set("panel-state", state);
    await store.save();
  } catch (error) {
    console.error("Failed to save panel state:", error);
  }
}

/**
 * Load panel state
 */
export async function loadPanelState(): Promise<PanelState | null> {
  try {
    const store = await getStore();
    const state = await store.get<PanelState>("panel-state");
    return state;
  } catch (error) {
    console.error("Failed to load panel state:", error);
    return null;
  }
}

/**
 * Save window preferences
 */
export async function saveWindowPreferences(
  prefs: WindowPreferences,
): Promise<void> {
  try {
    const store = await getStore();
    await store.set("window-preferences", prefs);
    await store.save();
  } catch (error) {
    console.error("Failed to save window preferences:", error);
  }
}

/**
 * Load window preferences
 */
export async function loadWindowPreferences(): Promise<WindowPreferences | null> {
  try {
    const store = await getStore();
    const prefs = await store.get<WindowPreferences>("window-preferences");
    return prefs;
  } catch (error) {
    console.error("Failed to load window preferences:", error);
    return null;
  }
}

/**
 * Save Side A/B preference
 */
export async function saveSidePreference(sideA: boolean): Promise<void> {
  try {
    const store = await getStore();
    await store.set("side-preference", sideA);
    await store.save();
  } catch (error) {
    console.error("Failed to save side preference:", error);
  }
}

/**
 * Load Side A/B preference
 */
export async function loadSidePreference(): Promise<boolean | null> {
  try {
    const store = await getStore();
    const sideA = await store.get<boolean>("side-preference");
    return sideA;
  } catch (error) {
    console.error("Failed to load side preference:", error);
    return null;
  }
}

/**
 * Save any key-value pair
 */
export async function saveValue<T>(key: string, value: T): Promise<void> {
  try {
    const store = await getStore();
    await store.set(key, value);
    await store.save();
  } catch (error) {
    console.error(`Failed to save value for key "${key}":`, error);
  }
}

/**
 * Load any key-value pair
 */
export async function loadValue<T>(key: string): Promise<T | null> {
  try {
    const store = await getStore();
    return await store.get<T>(key);
  } catch (error) {
    console.error(`Failed to load value for key "${key}":`, error);
    return null;
  }
}

/**
 * Delete a stored value
 */
export async function deleteValue(key: string): Promise<boolean> {
  try {
    const store = await getStore();
    const result = await store.delete(key);
    await store.save();
    return result;
  } catch (error) {
    console.error(`Failed to delete value for key "${key}":`, error);
    return false;
  }
}
