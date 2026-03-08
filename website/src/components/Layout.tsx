import { type ReactNode, useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import { Navigation } from "./Navigation";
import { Footer } from "./Footer";

interface LayoutProps {
  children: ReactNode;
}

export function Layout({ children }: LayoutProps) {
  const { pathname } = useLocation();
  const [transitioning, setTransitioning] = useState(false);
  const [displayedChildren, setDisplayedChildren] = useState(children);

  /*
   * Simple cross-fade on route change:
   * 1. Fade out (opacity 0)  — 150ms
   * 2. Swap content
   * 3. Fade in  (opacity 1)  — 300ms
   */
  useEffect(() => {
    setTransitioning(true);
    const timeout = setTimeout(() => {
      setDisplayedChildren(children);
      setTransitioning(false);
    }, 150);
    return () => clearTimeout(timeout);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pathname]);

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        minHeight: "100vh",
        position: "relative",
      }}
    >
      <a
        href="#main-content"
        style={{
          position: "absolute",
          left: "-9999px",
          top: "auto",
          width: "1px",
          height: "1px",
          overflow: "hidden",
          zIndex: 10000,
        }}
        onFocus={(e) => {
          e.currentTarget.style.position = "fixed";
          e.currentTarget.style.top = "8px";
          e.currentTarget.style.left = "8px";
          e.currentTarget.style.width = "auto";
          e.currentTarget.style.height = "auto";
          e.currentTarget.style.overflow = "visible";
          e.currentTarget.style.padding = "12px 20px";
          e.currentTarget.style.background = "var(--accent)";
          e.currentTarget.style.color = "var(--text-inverse)";
          e.currentTarget.style.borderRadius = "8px";
          e.currentTarget.style.fontWeight = "600";
          e.currentTarget.style.textDecoration = "none";
        }}
        onBlur={(e) => {
          e.currentTarget.style.position = "absolute";
          e.currentTarget.style.left = "-9999px";
          e.currentTarget.style.width = "1px";
          e.currentTarget.style.height = "1px";
          e.currentTarget.style.overflow = "hidden";
        }}
      >
        Skip to main content
      </a>

      <Navigation />

      <main
        id="main-content"
        style={{
          flex: 1,
          position: "relative",
          zIndex: 1,
          opacity: transitioning ? 0 : 1,
          transition: transitioning
            ? "opacity 150ms ease-out"
            : "opacity 300ms cubic-bezier(0.16, 1, 0.3, 1)",
        }}
      >
        {displayedChildren}
      </main>

      <Footer />
    </div>
  );
}
