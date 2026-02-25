import React, { useState } from 'react';
import { invoke } from '@tauri-apps/api/core';

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
    <div style={{
      padding: '8px',
      border: '1px solid #3e3e3e',
      borderRadius: '4px',
      backgroundColor: isActive ? '#2a4a2a' : '#2a2a2a',
    }}>
      <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
        <input
          type="checkbox"
          checked={isActive}
          onChange={handleToggle}
          disabled={isLoading}
          style={{ cursor: 'pointer' }}
        />
        <span style={{ fontSize: '12px', color: isActive ? '#4CAF50' : '#888' }}>
          {isActive ? 'Static IR Injection Active' : 'Enable Static IR Injection'}
        </span>
      </label>
      {isActive && (
        <div style={{ marginTop: '8px', fontSize: '11px', color: '#888' }}>
          ML and user input are bypassed. Test frame is active.
        </div>
      )}
    </div>
  );
};
