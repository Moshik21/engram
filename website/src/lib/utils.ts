import { useEffect, useRef, useState, type RefObject } from "react";

/** Hook that returns true when element is in viewport */
export function useInView(
  ref: RefObject<Element | null>,
  options?: IntersectionObserverInit,
): boolean {
  const [inView, setInView] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setInView(true);
          observer.disconnect(); // Only trigger once
        }
      },
      { threshold: 0.15, ...options },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [ref, options]);

  return inView;
}

/** Smooth scroll to element by ID */
export function scrollToId(id: string) {
  document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
}

/** cn - simple class name joiner (no clsx dependency) */
export function cn(...classes: (string | false | null | undefined)[]): string {
  return classes.filter(Boolean).join(" ");
}
