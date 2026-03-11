import { Link } from "react-router-dom";

/* ---- Column definitions ---- */
const COLUMNS = [
  {
    heading: "Product",
    links: [
      { label: "Features", to: "/#features" },
      { label: "Benchmarks", to: "/benchmarks" },
      { label: "Science", to: "/science" },
      { label: "Roadmap", to: "/roadmap" },
      { label: "Dashboard", to: "/#dashboard" },
    ],
  },
  {
    heading: "Resources",
    links: [
      { label: "Documentation", to: "/docs" },
      { label: "Benchmark Page", to: "/benchmarks" },
      {
        label: "GitHub",
        to: "https://github.com/engram-labs/engram",
        external: true,
      },
      { label: "API Reference", to: "/docs#api" },
    ],
  },
  {
    heading: "Company",
    links: [
      { label: "Vision", to: "/vision" },
      { label: "About", to: "/vision#about" },
      { label: "Blog", to: "/docs#blog" },
    ],
  },
] as const;

export function Footer() {
  const year = new Date().getFullYear();

  return (
    <footer
      style={{
        position: "relative",
        width: "100%",
        paddingTop: 0,
        background: "var(--void)",
        zIndex: 2,
      }}
    >
      {/* ---- Accent bar divider ---- */}
      <hr className="accent-bar" />

      <div
        style={{
          maxWidth: "var(--container-wide)",
          marginInline: "auto",
          padding:
            "var(--section-padding-y) var(--section-padding-x) 2.5rem",
          display: "grid",
          gridTemplateColumns: "1fr",
          gap: "3rem",
        }}
        className="footer-grid"
      >
        {/* ---- Left column: logo + tagline ---- */}
        <div style={{ maxWidth: 320 }}>
          {/* Logo */}
          <Link
            to="/"
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 2,
              textDecoration: "none",
              marginBottom: 12,
            }}
            aria-label="Engram home"
          >
            <span
              style={{
                fontFamily: "var(--font-display)",
                fontStyle: "italic",
                fontSize: "1.25rem",
                fontWeight: 400,
                color: "#fff",
                letterSpacing: "-0.02em",
                lineHeight: 1,
              }}
            >
              Engram
            </span>
            <span
              style={{
                display: "inline-block",
                width: 4,
                height: 4,
                borderRadius: "50%",
                background: "var(--accent)",
                boxShadow: "0 0 6px var(--accent-glow)",
                marginLeft: 1,
                marginBottom: 2,
                alignSelf: "flex-end",
              }}
            />
          </Link>

          <p
            style={{
              fontSize: "0.9375rem",
              lineHeight: 1.6,
              color: "var(--text-secondary)",
              marginTop: 8,
            }}
          >
            Private long-term memory for AI agents.
          </p>
        </div>

        {/* ---- Link columns ---- */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))",
            gap: "2rem",
          }}
        >
          {COLUMNS.map((col) => (
            <div key={col.heading}>
              <h4
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: "0.6875rem",
                  fontWeight: 500,
                  letterSpacing: "0.08em",
                  textTransform: "uppercase",
                  color: "var(--text-muted)",
                  marginBottom: "1rem",
                  lineHeight: 1,
                }}
              >
                {col.heading}
              </h4>
              <ul
                style={{
                  listStyle: "none",
                  display: "flex",
                  flexDirection: "column",
                  gap: "0.625rem",
                }}
              >
                {col.links.map((link) => (
                  <li key={link.label}>
                    {"external" in link && link.external ? (
                      <a
                        href={link.to}
                        target="_blank"
                        rel="noopener noreferrer"
                        style={{
                          fontSize: "0.875rem",
                          color: "var(--text-secondary)",
                          textDecoration: "none",
                          transition: "color 150ms ease",
                          display: "inline-flex",
                          alignItems: "center",
                          gap: "0.25rem",
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
                        {link.label}
                        <svg
                          width="10"
                          height="10"
                          viewBox="0 0 12 12"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          style={{ opacity: 0.4 }}
                        >
                          <path d="M3 9L9 3" />
                          <path d="M4.5 3H9V7.5" />
                        </svg>
                      </a>
                    ) : (
                      <Link
                        to={link.to}
                        style={{
                          fontSize: "0.875rem",
                          color: "var(--text-secondary)",
                          textDecoration: "none",
                          transition: "color 150ms ease",
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
                        {link.label}
                      </Link>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>

      {/* ---- Bottom bar ---- */}
      <div
        style={{
          borderTop: "1px solid var(--border)",
          padding: "1.25rem var(--section-padding-x)",
        }}
      >
        <div
          style={{
            maxWidth: "var(--container-wide)",
            marginInline: "auto",
            display: "flex",
            flexWrap: "wrap",
            alignItems: "center",
            justifyContent: "space-between",
            gap: "0.75rem",
          }}
        >
          <p
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "0.6875rem",
              color: "var(--text-muted)",
              letterSpacing: "0.02em",
            }}
          >
            Built with cognitive architecture
          </p>
          <p
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "0.6875rem",
              color: "var(--text-muted)",
              letterSpacing: "0.02em",
            }}
          >
            &copy; {year} Engram. All rights reserved.
          </p>
        </div>
      </div>

      {/* ---- Responsive styles injected via <style> ---- */}
      <style>{`
        .footer-grid {
          grid-template-columns: 1fr;
        }
        @media (min-width: 768px) {
          .footer-grid {
            grid-template-columns: 280px 1fr;
            gap: 4rem;
          }
        }
      `}</style>
    </footer>
  );
}
