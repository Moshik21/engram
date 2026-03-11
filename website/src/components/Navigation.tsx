import { useCallback, useEffect, useState } from "react";
import { Link, useLocation } from "react-router-dom";

/* ---- Internal nav items ---- */
const NAV_LINKS = [
  { to: "/benchmarks", label: "Benchmarks" },
  { to: "/science", label: "Science" },
  { to: "/vision", label: "Vision" },
  { to: "/roadmap", label: "Roadmap" },
  { to: "/docs", label: "Docs" },
] as const;

const GITHUB_URL = "https://github.com/engram-labs/engram";

export function Navigation() {
  const { pathname } = useLocation();
  const [scrolled, setScrolled] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  /* Track scroll to toggle solid background */
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 32);
    onScroll(); // initial check
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  /* Close mobile menu on route change */
  useEffect(() => {
    setMobileOpen(false);
  }, [pathname]);

  /* Close mobile menu on Escape */
  useEffect(() => {
    if (!mobileOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setMobileOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [mobileOpen]);

  /* Lock body scroll when mobile menu is open */
  useEffect(() => {
    if (mobileOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
    }
    return () => {
      document.body.style.overflow = "";
    };
  }, [mobileOpen]);

  const toggleMenu = useCallback(() => setMobileOpen((v) => !v), []);

  const isActive = (to: string) =>
    pathname === to || (to !== "/" && pathname.startsWith(to));

  return (
    <>
      <header
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          right: 0,
          zIndex: 1000,
          padding: "0 var(--section-padding-x)",
          background: scrolled
            ? "rgba(3, 4, 8, 0.88)"
            : "rgba(3, 4, 8, 0)",
          borderBottom: scrolled
            ? "1px solid var(--border)"
            : "1px solid transparent",
          backdropFilter: scrolled ? "blur(20px) saturate(1.3)" : "none",
          WebkitBackdropFilter: scrolled
            ? "blur(20px) saturate(1.3)"
            : "none",
          transition:
            "background 300ms ease, border-color 300ms ease, backdrop-filter 300ms ease",
        }}
      >
        <div
          style={{
            maxWidth: "var(--container-wide)",
            marginInline: "auto",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            height: 64,
          }}
        >
          {/* ---- Logo ---- */}
          <Link
            to="/"
            style={{
              display: "flex",
              alignItems: "center",
              gap: 2,
              textDecoration: "none",
            }}
            aria-label="Engram home"
          >
            <span
              style={{
                fontFamily: "var(--font-display)",
                fontStyle: "italic",
                fontSize: "1.375rem",
                fontWeight: 400,
                letterSpacing: "-0.02em",
                color: "#fff",
                lineHeight: 1,
              }}
            >
              Engram
            </span>
            {/* Cyan dot — matches the dashboard logo accent */}
            <span
              style={{
                display: "inline-block",
                width: 5,
                height: 5,
                borderRadius: "50%",
                background: "var(--accent)",
                boxShadow: "0 0 6px var(--accent-glow)",
                marginLeft: 1,
                marginBottom: 2,
                alignSelf: "flex-end",
              }}
            />
          </Link>

          {/* ---- Desktop links ---- */}
          <nav
            aria-label="Main navigation"
            style={{ display: "flex", alignItems: "center", gap: "2rem" }}
            className="hide-mobile"
          >
            {NAV_LINKS.map(({ to, label }) => (
              <Link
                key={to}
                to={to}
                style={{
                  fontSize: "0.875rem",
                  fontWeight: 400,
                  color: isActive(to)
                    ? "var(--accent)"
                    : "var(--text-secondary)",
                  textDecoration: "none",
                  transition: "color 150ms ease",
                  position: "relative",
                }}
                onMouseEnter={(e) => {
                  if (!isActive(to))
                    (e.currentTarget as HTMLElement).style.color =
                      "var(--text-primary)";
                }}
                onMouseLeave={(e) => {
                  if (!isActive(to))
                    (e.currentTarget as HTMLElement).style.color =
                      "var(--text-secondary)";
                }}
              >
                {label}
                {/* Active indicator bar */}
                {isActive(to) && (
                  <span
                    style={{
                      position: "absolute",
                      bottom: -20,
                      left: 0,
                      right: 0,
                      height: 1,
                      background: "var(--accent)",
                      borderRadius: "var(--radius-full)",
                      opacity: 0.6,
                    }}
                  />
                )}
              </Link>
            ))}

            {/* GitHub link */}
            <a
              href={GITHUB_URL}
              target="_blank"
              rel="noopener noreferrer"
              style={{
                fontSize: "0.875rem",
                fontWeight: 400,
                color: "var(--text-secondary)",
                textDecoration: "none",
                transition: "color 150ms ease",
                display: "flex",
                alignItems: "center",
                gap: "0.375rem",
              }}
              onMouseEnter={(e) =>
                ((e.currentTarget as HTMLElement).style.color =
                  "var(--text-primary)")
              }
              onMouseLeave={(e) =>
                ((e.currentTarget as HTMLElement).style.color =
                  "var(--text-secondary)")
              }
            >
              GitHub
              <svg
                width="12"
                height="12"
                viewBox="0 0 12 12"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                style={{ opacity: 0.6 }}
              >
                <path d="M3 9L9 3" />
                <path d="M4.5 3H9V7.5" />
              </svg>
            </a>
          </nav>

          {/* ---- Mobile hamburger ---- */}
          <button
            className="hide-desktop"
            onClick={toggleMenu}
            aria-label={mobileOpen ? "Close menu" : "Open menu"}
            aria-expanded={mobileOpen}
            style={{
              display: "flex",
              flexDirection: "column",
              gap: mobileOpen ? 0 : 4,
              padding: 8,
              position: "relative",
              width: 36,
              height: 36,
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <span
              style={{
                display: "block",
                width: 18,
                height: 1.5,
                background: "var(--text-secondary)",
                borderRadius: 9999,
                transition: "transform 200ms ease, opacity 150ms ease",
                transform: mobileOpen
                  ? "rotate(45deg) translate(0px, 0px)"
                  : "none",
                transformOrigin: "center",
                position: mobileOpen ? "absolute" : "relative",
              }}
            />
            <span
              style={{
                display: "block",
                width: 18,
                height: 1.5,
                background: "var(--text-secondary)",
                borderRadius: 9999,
                transition: "opacity 150ms ease",
                opacity: mobileOpen ? 0 : 1,
              }}
            />
            <span
              style={{
                display: "block",
                width: 18,
                height: 1.5,
                background: "var(--text-secondary)",
                borderRadius: 9999,
                transition: "transform 200ms ease, opacity 150ms ease",
                transform: mobileOpen
                  ? "rotate(-45deg) translate(0px, 0px)"
                  : "none",
                transformOrigin: "center",
                position: mobileOpen ? "absolute" : "relative",
              }}
            />
          </button>
        </div>
      </header>

      {/* ---- Mobile full-screen overlay ---- */}
      {mobileOpen && (
        <div
          style={{
            position: "fixed",
            inset: 0,
            zIndex: 999,
            background: "rgba(3, 4, 8, 0.97)",
            backdropFilter: "blur(32px) saturate(1.2)",
            WebkitBackdropFilter: "blur(32px) saturate(1.2)",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: "2rem",
            animation: "fade-in 200ms ease-out both",
          }}
          role="dialog"
          aria-modal="true"
          aria-label="Mobile navigation"
        >
          {/* Close button in the top-right area */}
          <button
            onClick={toggleMenu}
            aria-label="Close menu"
            style={{
              position: "absolute",
              top: 16,
              right: "var(--section-padding-x)",
              padding: 8,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <svg
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              stroke="var(--text-secondary)"
              strokeWidth="1.5"
              strokeLinecap="round"
            >
              <path d="M6 6L18 18" />
              <path d="M18 6L6 18" />
            </svg>
          </button>

          {NAV_LINKS.map(({ to, label }) => (
            <Link
              key={to}
              to={to}
              onClick={toggleMenu}
              style={{
                fontFamily: "var(--font-display)",
                fontStyle: "italic",
                fontSize: "2rem",
                fontWeight: 400,
                color: isActive(to)
                  ? "var(--accent)"
                  : "var(--text-primary)",
                textDecoration: "none",
                transition: "color 150ms ease",
                letterSpacing: "-0.01em",
              }}
            >
              {label}
            </Link>
          ))}

          <a
            href={GITHUB_URL}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              fontFamily: "var(--font-display)",
              fontStyle: "italic",
              fontSize: "2rem",
              fontWeight: 400,
              color: "var(--text-secondary)",
              textDecoration: "none",
              transition: "color 150ms ease",
              display: "flex",
              alignItems: "center",
              gap: "0.5rem",
            }}
          >
            GitHub
            <svg
              width="16"
              height="16"
              viewBox="0 0 12 12"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              style={{ opacity: 0.5 }}
            >
              <path d="M3 9L9 3" />
              <path d="M4.5 3H9V7.5" />
            </svg>
          </a>

          {/* Subtle bottom tagline */}
          <p
            style={{
              position: "absolute",
              bottom: 40,
              fontFamily: "var(--font-mono)",
              fontSize: "0.6875rem",
              color: "var(--text-muted)",
              letterSpacing: "0.04em",
            }}
          >
            Private long-term memory for AI agents.
          </p>
        </div>
      )}
    </>
  );
}
