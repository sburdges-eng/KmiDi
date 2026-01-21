import React, { useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import '../App.css';

/**
 * IntentInjector - UI toggle for static IntentFrame injection
 * 
 * Allows bypassing ML and user input for UI testing isolation
 */
export const IntentInjector: React.FC = () => {
  const [isActive, setIsActive] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const handleToggle = async () => {
    setIsLoading(true);
    try {
      if (isActive) {
        // Disable injection
        await invoke('disable_intent_ir_injection');
        setIsActive(false);
      } else {
        // Enable injection with test frame
        await invoke('enable_intent_ir_injection');
        setIsActive(true);
      }
    } catch (error) {
      console.error('Failed to toggle intent injection:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="p-2 border border-border-light rounded bg-bg-secondary">
      <label className="flex items-center gap-2 cursor-pointer">
        <input
          type="checkbox"
          checked={isActive}
          onChange={handleToggle}
          disabled={isLoading}
          className="cursor-pointer"
          aria-label={isActive ? 'Disable static IR injection' : 'Enable static IR injection'}
          aria-describedby="intent-injector-description"
          aria-busy={isLoading}
        />
        <span className={`text-xs ${isActive ? 'text-accent-success' : 'text-text-tertiary'}`}>
          {isActive ? 'Static IR Injection Active' : 'Enable Static IR Injection'}
        </span>
      </label>
      {isActive && (
        <div id="intent-injector-description" className="mt-2 text-xs text-text-tertiary" role="status">
          ML and user input are bypassed. Test frame is active.
        </div>
      )}
    </div>
  );
};
