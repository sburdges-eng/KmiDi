/**
 * React hooks for persistence
 * 
 * Provides React hooks for easy integration of persistence functionality
 */

import { useState, useEffect, useCallback } from 'react';
import {
  savePanelState,
  loadPanelState,
  saveWindowPreferences,
  loadWindowPreferences,
  saveSidePreference,
  loadSidePreference,
  type PanelState,
  type WindowPreferences,
} from '../utils/persistence';

/**
 * Hook for managing panel state persistence
 */
export function usePanelPersistence() {
  const [panelState, setPanelState] = useState<PanelState | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Load state on mount
  useEffect(() => {
    loadPanelState().then(state => {
      setPanelState(state);
      setIsLoading(false);
    });
  }, []);

  // Save state when it changes
  const updatePanelState = useCallback(async (updates: Partial<PanelState>) => {
    const currentState = panelState || {
      inspector: { visible: true },
      timeline: { visible: true },
      browser: { visible: true },
    };
    
    const newState: PanelState = {
      ...currentState,
      ...updates,
    };
    
    setPanelState(newState);
    await savePanelState(newState);
  }, [panelState]);

  return {
    panelState,
    isLoading,
    updatePanelState,
  };
}

/**
 * Hook for managing window preferences persistence
 */
export function useWindowPreferences() {
  const [preferences, setPreferences] = useState<WindowPreferences | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Load preferences on mount
  useEffect(() => {
    loadWindowPreferences().then(prefs => {
      setPreferences(prefs);
      setIsLoading(false);
    });
  }, []);

  // Save preferences when they change
  const updatePreferences = useCallback(async (updates: Partial<WindowPreferences>) => {
    const currentPrefs = preferences || {
      width: 1200,
      height: 800,
      sideA: true,
    };
    
    const newPrefs: WindowPreferences = {
      ...currentPrefs,
      ...updates,
    };
    
    setPreferences(newPrefs);
    await saveWindowPreferences(newPrefs);
  }, [preferences]);

  return {
    preferences,
    isLoading,
    updatePreferences,
  };
}

/**
 * Hook for managing Side A/B preference
 */
export function useSidePreference() {
  const [sideA, setSideA] = useState<boolean>(true);
  const [isLoading, setIsLoading] = useState(true);

  // Load preference on mount
  useEffect(() => {
    loadSidePreference().then(pref => {
      if (pref !== null) {
        setSideA(pref);
      }
      setIsLoading(false);
    });
  }, []);

  // Save preference when it changes
  const updateSidePreference = useCallback(async (newSideA: boolean) => {
    setSideA(newSideA);
    await saveSidePreference(newSideA);
  }, []);

  return {
    sideA,
    isLoading,
    updateSidePreference,
  };
}
