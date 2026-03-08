import { useRef, useCallback } from "react";

export interface CameraBookmark {
  nodeId: string | null;
  position: { x: number; y: number; z: number };
  lookAt: { x: number; y: number; z: number };
}

const MAX_HISTORY = 50;

export function useNavigationHistory() {
  const historyRef = useRef<CameraBookmark[]>([]);
  const indexRef = useRef(-1);
  // Guard against pushes triggered by programmatic fly-to (back/forward)
  const suppressPushRef = useRef(false);

  const push = useCallback((bookmark: CameraBookmark) => {
    if (suppressPushRef.current) return;
    const h = historyRef.current;
    const idx = indexRef.current;
    // Truncate forward history
    historyRef.current = h.slice(0, idx + 1);
    historyRef.current.push(bookmark);
    if (historyRef.current.length > MAX_HISTORY) {
      historyRef.current.shift();
    }
    indexRef.current = historyRef.current.length - 1;
  }, []);

  const canGoBack = useCallback(() => indexRef.current > 0, []);
  const canGoForward = useCallback(
    () => indexRef.current < historyRef.current.length - 1,
    [],
  );

  const goBack = useCallback(
    (
      flyTo: (pos: CameraBookmark["position"], lookAt: CameraBookmark["lookAt"]) => void,
      nodeExists: (id: string) => boolean,
    ) => {
      let idx = indexRef.current - 1;
      // Skip deleted nodes
      while (idx >= 0) {
        const bm = historyRef.current[idx];
        if (!bm.nodeId || nodeExists(bm.nodeId)) break;
        idx--;
      }
      if (idx < 0) return null;
      indexRef.current = idx;
      const bm = historyRef.current[idx];
      suppressPushRef.current = true;
      flyTo(bm.position, bm.lookAt);
      // Reset suppress after fly completes
      setTimeout(() => { suppressPushRef.current = false; }, 50);
      return bm.nodeId;
    },
    [],
  );

  const goForward = useCallback(
    (
      flyTo: (pos: CameraBookmark["position"], lookAt: CameraBookmark["lookAt"]) => void,
      nodeExists: (id: string) => boolean,
    ) => {
      const max = historyRef.current.length - 1;
      let idx = indexRef.current + 1;
      while (idx <= max) {
        const bm = historyRef.current[idx];
        if (!bm.nodeId || nodeExists(bm.nodeId)) break;
        idx++;
      }
      if (idx > max) return null;
      indexRef.current = idx;
      const bm = historyRef.current[idx];
      suppressPushRef.current = true;
      flyTo(bm.position, bm.lookAt);
      setTimeout(() => { suppressPushRef.current = false; }, 50);
      return bm.nodeId;
    },
    [],
  );

  const getVisibleTrail = useCallback((maxVisible = 8) => {
    const h = historyRef.current;
    const idx = indexRef.current;
    if (h.length === 0) return { items: [] as Array<CameraBookmark & { index: number }>, currentIndex: -1, truncatedLeft: false, truncatedRight: false };

    let start = Math.max(0, idx - Math.floor(maxVisible / 2));
    let end = start + maxVisible;
    if (end > h.length) {
      end = h.length;
      start = Math.max(0, end - maxVisible);
    }

    return {
      items: h.slice(start, end).map((bm, i) => ({ ...bm, index: start + i })),
      currentIndex: idx - start,
      truncatedLeft: start > 0,
      truncatedRight: end < h.length,
    };
  }, []);

  const goToIndex = useCallback(
    (
      index: number,
      flyTo: (pos: CameraBookmark["position"], lookAt: CameraBookmark["lookAt"]) => void,
    ) => {
      if (index < 0 || index >= historyRef.current.length) return null;
      indexRef.current = index;
      const bm = historyRef.current[index];
      suppressPushRef.current = true;
      flyTo(bm.position, bm.lookAt);
      setTimeout(() => { suppressPushRef.current = false; }, 50);
      return bm.nodeId;
    },
    [],
  );

  return { push, canGoBack, canGoForward, goBack, goForward, goToIndex, getVisibleTrail };
}
