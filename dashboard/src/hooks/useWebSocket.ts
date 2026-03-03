import { useEffect, useRef } from "react";
import { useEngramStore } from "../store";
import type { Episode, EpisodeStatus, GraphDelta, GraphNode, GraphEdge } from "../store/types";

const BASE_DELAY = 1000;
const MAX_DELAY = 30_000;
const MAX_ATTEMPTS = 5;
const JITTER = 0.2;
const PING_INTERVAL = 25_000;

function backoffDelay(attempt: number): number {
  const base = Math.min(BASE_DELAY * Math.pow(2, attempt), MAX_DELAY);
  const jitter = base * JITTER * (Math.random() * 2 - 1);
  return base + jitter;
}

function getWsUrl(): string {
  const loc = window.location;
  const protocol = loc.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${loc.host}/ws/dashboard`;
}

interface WsEvent {
  type: string;
  seq?: number;
  episode?: Episode;
  episodeId?: string;
  status?: EpisodeStatus;
  error?: string | null;
  delta?: GraphDelta;
  events?: WsEvent[];
  isFull?: boolean;
  // Flattened payload fields (server WebSocket handler merges payload into top level)
  nodes?: GraphNode[];
  edges?: GraphEdge[];
  entityId?: string;
  name?: string;
  entityType?: string;
  activation?: number;
  accessedVia?: string;
  // activation.snapshot sends nested payload (not flattened)
  payload?: {
    topActivated?: Array<{
      entityId: string;
      name: string;
      entityType: string;
      currentActivation: number;
      accessCount: number;
    }>;
  };
}

// Exported ref for components to send commands
let _wsInstance: WebSocket | null = null;

export function getWsInstance(): WebSocket | null {
  return _wsInstance;
}

export function sendWsCommand(command: object): void {
  if (_wsInstance && _wsInstance.readyState === WebSocket.OPEN) {
    _wsInstance.send(JSON.stringify(command));
  }
}

function routeEvent(data: WsEvent) {
  const s = useEngramStore.getState();

  if (data.seq !== undefined) {
    s.setLastSeq(data.seq);
  }

  switch (data.type) {
    case "episode.queued":
      if (data.episode) {
        s.prependEpisode(data.episode);
      }
      break;
    case "episode.completed":
    case "episode.failed":
      if (data.episodeId && data.status) {
        s.updateEpisodeStatus(data.episodeId, data.status, data.error);
      }
      break;
    case "graph.nodes_added":
      if (data.nodes && data.nodes.length > 0) {
        s.mergeGraphDelta({
          nodesAdded: data.nodes,
          edgesAdded: data.edges,
        });
      } else {
        // Fallback for old payloads without full node/edge data
        s.loadInitialGraph();
      }
      break;
    case "graph.delta":
      if (data.delta) {
        s.mergeGraphDelta(data.delta);
      } else {
        s.loadInitialGraph();
      }
      break;
    case "activation.access":
      if (data.entityId) {
        s.addActivationPulse({
          entityId: data.entityId,
          name: data.name ?? "",
          entityType: data.entityType ?? "Other",
          activation: data.activation ?? 0,
          accessedVia: data.accessedVia ?? "unknown",
        });
        s.updateNodeActivations([
          { entityId: data.entityId, activation: data.activation ?? 0 },
        ]);
      }
      break;
    case "activation.snapshot":
      if (data.payload?.topActivated) {
        s.setActivationLeaderboard(
          data.payload.topActivated.map((item) => ({
            ...item,
            lastAccessedAt: null,
            decayRate: 0.5,
          })),
        );
        s.updateNodeActivations(
          data.payload.topActivated.map((item) => ({
            entityId: item.entityId,
            activation: item.currentActivation,
          })),
        );
      }
      break;
    case "consolidation.started":
    case "consolidation.completed":
      s.loadStatus();
      s.loadCycles();
      break;
    case "pong":
      break;
    default:
      if (data.type?.startsWith("consolidation.phase.")) {
        s.loadStatus();
      }
      break;
  }
}

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const pingRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const reconnectRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const attemptRef = useRef(0);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;

    function connect() {
      if (!mountedRef.current) return;

      const store = useEngramStore.getState();
      store.setReadyState(attemptRef.current === 0 ? "connecting" : "reconnecting");
      store.setReconnectAttempt(attemptRef.current);

      const ws = new WebSocket(getWsUrl());
      wsRef.current = ws;
      _wsInstance = ws;

      ws.onopen = () => {
        if (!mountedRef.current) return;
        attemptRef.current = 0;
        const s = useEngramStore.getState();
        s.setReadyState("connected");
        s.setReconnectAttempt(0);

        // Resync if we had a previous sequence
        const lastSeq = s.lastSeq;
        if (lastSeq > 0) {
          ws.send(JSON.stringify({ type: "command", command: "resync", lastSeq }));
        }

        // Start ping interval
        pingRef.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "ping" }));
          }
        }, PING_INTERVAL);
      };

      ws.onmessage = (evt) => {
        if (!mountedRef.current) return;
        try {
          const data = JSON.parse(evt.data) as WsEvent;

          // Handle resync response
          if (data.type === "resync") {
            if (data.isFull) {
              // Full refresh needed
              const s = useEngramStore.getState();
              s.loadInitialGraph();
              s.loadEpisodes();
            } else if (data.events) {
              // Replay missed events
              for (const e of data.events) {
                routeEvent(e);
              }
            }
            return;
          }

          routeEvent(data);
        } catch {
          // Ignore malformed messages
        }
      };

      ws.onclose = () => {
        cleanup();
        _wsInstance = null;
        if (!mountedRef.current) return;

        useEngramStore.getState().setReadyState("disconnected");

        if (attemptRef.current < MAX_ATTEMPTS) {
          const delay = backoffDelay(attemptRef.current);
          attemptRef.current += 1;
          reconnectRef.current = setTimeout(connect, delay);
        }
      };

      ws.onerror = () => {
        // onclose will fire after onerror, let it handle reconnect
      };
    }

    function cleanup() {
      if (pingRef.current) {
        clearInterval(pingRef.current);
        pingRef.current = null;
      }
    }

    connect();

    return () => {
      mountedRef.current = false;
      cleanup();
      if (reconnectRef.current) {
        clearTimeout(reconnectRef.current);
        reconnectRef.current = null;
      }
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.close();
        wsRef.current = null;
      }
      _wsInstance = null;
      useEngramStore.getState().setReadyState("disconnected");
    };
  }, []);
}
