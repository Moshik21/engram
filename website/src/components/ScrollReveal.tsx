import { useRef, useEffect, useState, type ReactNode } from "react";
import { useInView } from "../lib/utils";

interface ScrollRevealProps {
  children: ReactNode;
  className?: string;
  style?: React.CSSProperties;
  delay?: number; // ms delay for stagger effect
  direction?: "up" | "down" | "left" | "right"; // slide direction
}

export function ScrollReveal({
  children,
  className = "",
  style: extraStyle,
  delay = 0,
  direction = "up",
}: ScrollRevealProps) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref);
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(false);

  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    setPrefersReducedMotion(mq.matches);
    const handler = (e: MediaQueryListEvent) => setPrefersReducedMotion(e.matches);
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, []);

  const translateMap = {
    up: "translateY(24px)",
    down: "translateY(-24px)",
    left: "translateX(-24px)",
    right: "translateX(24px)",
  };

  const noMotion = prefersReducedMotion;

  return (
    <div
      ref={ref}
      className={className}
      style={{
        ...extraStyle,
        opacity: noMotion || inView ? 1 : 0,
        transform: noMotion || inView ? "none" : translateMap[direction],
        transition: noMotion
          ? "none"
          : `opacity 0.7s ease ${delay}ms, transform 0.7s ease ${delay}ms`,
      }}
    >
      {children}
    </div>
  );
}
