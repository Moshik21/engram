import { useEffect, useRef } from "react";

export interface KeyboardActions {
  openSearch: () => void;
  deselect: () => void;
  centerOnSelected: () => void;
  goBack: () => void;
  goForward: () => void;
  resetCamera: () => void;
  toggleHeatmap: () => void;
  toggleEdgeLabels: () => void;
  toggleRenderMode: () => void;
  showHelp: () => void;
}

function isInputFocused(): boolean {
  const el = document.activeElement;
  if (!el) return false;
  const tag = el.tagName;
  return (
    tag === "INPUT" ||
    tag === "TEXTAREA" ||
    tag === "SELECT" ||
    (el as HTMLElement).isContentEditable
  );
}

export function useKeyboardNavigation(actions: KeyboardActions) {
  const actionsRef = useRef(actions);

  useEffect(() => {
    actionsRef.current = actions;
  }, [actions]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // Always allow Escape (closes overlays)
      if (e.key === "Escape") {
        actionsRef.current.deselect();
        return;
      }

      // Skip when input is focused (except Escape above)
      if (isInputFocused()) return;

      // Alt+Arrow: history navigation
      if (e.altKey && e.key === "ArrowLeft") {
        e.preventDefault();
        actionsRef.current.goBack();
        return;
      }
      if (e.altKey && e.key === "ArrowRight") {
        e.preventDefault();
        actionsRef.current.goForward();
        return;
      }

      // Single-key shortcuts (no modifiers except shift for ?)
      if (e.ctrlKey || e.metaKey || e.altKey) return;

      switch (e.key) {
        case "/":
          e.preventDefault();
          actionsRef.current.openSearch();
          break;
        case " ":
          e.preventDefault();
          actionsRef.current.centerOnSelected();
          break;
        case "r":
        case "R":
          if (!e.shiftKey) {
            e.preventDefault();
            actionsRef.current.resetCamera();
          }
          break;
        case "h":
        case "H":
          if (!e.shiftKey) {
            e.preventDefault();
            actionsRef.current.toggleHeatmap();
          }
          break;
        case "l":
        case "L":
          if (!e.shiftKey) {
            e.preventDefault();
            actionsRef.current.toggleEdgeLabels();
          }
          break;
        case "g":
        case "G":
          if (!e.shiftKey) {
            e.preventDefault();
            actionsRef.current.toggleRenderMode();
          }
          break;
        case "?":
          e.preventDefault();
          actionsRef.current.showHelp();
          break;
      }
    };

    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);
}
