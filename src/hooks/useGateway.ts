/**
 * useGateway — WebSocket connection to the KMiDi Gateway API.
 *
 * Handles the cascade: if cloud is unavailable or times out,
 * falls back to local useMusicBrain API calls.
 *
 * Latency note: WebSocket connect/reconnect is async and never blocks
 * the audio thread. All state updates are batched via React setState.
 * The audio thread (JUCE/C++) never waits on this hook.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import type {
  CascadeTier,
  DomainInfluences,
  GatewayConnectionState,
  GatewayHealth,
  MusicalContext,
  PluginGatewayRequest,
  PluginGatewayResponse,
  PluginId,
  Thought,
} from '../types/Architecture';
import { intensityToTier } from '../types/Architecture';

const GATEWAY_URL = import.meta.env.VITE_GATEWAY_URL ?? 'ws://127.0.0.1:8001/api/v1/ws/daiw';
const RECONNECT_DELAY_MS = 3000;
const MAX_RECONNECT_ATTEMPTS = 5;

interface GatewayState {
  connection: GatewayConnectionState;
  health: GatewayHealth | null;
  lastThought: Thought | null;
  lastTierUsed: CascadeTier | null;
  lastLatencyMs: number | null;
}

export function useGateway(sessionId: string) {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttempts = useRef(0);
  const pendingCallbacks = useRef<Map<string, {
    resolve: (resp: PluginGatewayResponse) => void;
    reject: (err: Error) => void;
    timer: ReturnType<typeof setTimeout>;
  }>>(new Map());

  const [state, setState] = useState<GatewayState>({
    connection: 'disconnected',
    health: null,
    lastThought: null,
    lastTierUsed: null,
    lastLatencyMs: null,
  });

  // ── Connect ──────────────────────────────────────────────────────────────

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    setState(s => ({ ...s, connection: 'connecting' }));
    const ws = new WebSocket(`${GATEWAY_URL}?session=${sessionId}`);

    ws.onopen = () => {
      reconnectAttempts.current = 0;
      setState(s => ({ ...s, connection: 'connected' }));
      // Request health on connect
      ws.send(JSON.stringify({ type: 'health' }));
    };

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);

      if (msg.type === 'health') {
        setState(s => ({ ...s, health: msg.payload }));
        return;
      }

      if (msg.type === 'thought_response') {
        const reqId = msg.request_id as string;
        const pending = pendingCallbacks.current.get(reqId);
        if (pending) {
          clearTimeout(pending.timer);
          pendingCallbacks.current.delete(reqId);
          pending.resolve(msg.payload as PluginGatewayResponse);
        }
        setState(s => ({
          ...s,
          lastThought: msg.payload.thought,
          lastTierUsed: msg.payload.tier_used,
          lastLatencyMs: msg.payload.latency_ms,
        }));
        return;
      }
    };

    ws.onclose = () => {
      setState(s => ({ ...s, connection: 'disconnected' }));
      // Reject all pending requests
      for (const [, pending] of pendingCallbacks.current) {
        clearTimeout(pending.timer);
        pending.reject(new Error('Gateway connection closed'));
      }
      pendingCallbacks.current.clear();

      // Auto-reconnect
      if (reconnectAttempts.current < MAX_RECONNECT_ATTEMPTS) {
        reconnectAttempts.current += 1;
        setTimeout(connect, RECONNECT_DELAY_MS);
      } else {
        setState(s => ({ ...s, connection: 'error' }));
      }
    };

    ws.onerror = () => {
      setState(s => ({ ...s, connection: 'error' }));
    };

    wsRef.current = ws;
  }, [sessionId]);

  const disconnect = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
    setState(s => ({ ...s, connection: 'disconnected' }));
  }, []);

  // ── Request a Thought ────────────────────────────────────────────────────

  const requestThought = useCallback((
    pluginId: PluginId,
    mlIntensity: number,
    domainInfluences: DomainInfluences,
    context: MusicalContext,
    timeoutMs: number = 300,
  ): Promise<PluginGatewayResponse> => {
    return new Promise((resolve, reject) => {
      const ws = wsRef.current;

      // If not connected, reject immediately — caller falls back to local
      if (!ws || ws.readyState !== WebSocket.OPEN) {
        reject(new Error('Gateway not connected'));
        return;
      }

      const requestId = `${pluginId}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      const targetTier = intensityToTier(mlIntensity);

      const request: PluginGatewayRequest & { type: string; request_id: string } = {
        type: 'thought_request',
        request_id: requestId,
        plugin_id: pluginId,
        ml_intensity: mlIntensity,
        domain_influences: domainInfluences,
        context,
        timeout_ms: timeoutMs,
      };

      // Timeout — if gateway doesn't respond, caller falls back to local
      const timer = setTimeout(() => {
        pendingCallbacks.current.delete(requestId);
        reject(new Error(`Gateway timeout (${timeoutMs}ms) for ${pluginId} at tier ${targetTier}`));
      }, timeoutMs);

      pendingCallbacks.current.set(requestId, { resolve, reject, timer });
      ws.send(JSON.stringify(request));
    });
  }, []);

  // ── Cascade helper ───────────────────────────────────────────────────────

  /**
   * requestThoughtWithCascade — tries gateway, falls back to local on failure.
   * This is the function UI components should call. It NEVER throws.
   * On gateway failure, returns null — caller uses local engine output.
   */
  const requestThoughtWithCascade = useCallback(async (
    pluginId: PluginId,
    mlIntensity: number,
    domainInfluences: DomainInfluences,
    context: MusicalContext,
    timeoutMs: number = 300,
  ): Promise<PluginGatewayResponse | null> => {
    // If intensity is too low for cloud, skip the call entirely
    if (mlIntensity <= 50) return null;

    try {
      return await requestThought(pluginId, mlIntensity, domainInfluences, context, timeoutMs);
    } catch {
      // Gateway unavailable or timed out — local cascade takes over
      return null;
    }
  }, [requestThought]);

  // ── Lifecycle ────────────────────────────────────────────────────────────

  useEffect(() => {
    connect();
    return disconnect;
  }, [connect, disconnect]);

  return {
    ...state,
    connect,
    disconnect,
    requestThought,
    requestThoughtWithCascade,
    isConnected: state.connection === 'connected',
    isCloudAvailable: state.health?.status === 'healthy',
    availableTiers: state.health?.available_tiers ?? [],
  };
}
