import { lazy, Suspense, useEffect } from "react";
import { Routes, Route, useLocation } from "react-router-dom";
import { Layout } from "./components/Layout";

function lazyNamed<TModule, TKey extends keyof TModule>(
  loader: () => Promise<TModule>,
  key: TKey,
) {
  return lazy(async () => {
    const mod = await loader();
    return { default: mod[key] as React.ComponentType };
  });
}

const HomePage = lazyNamed(() => import("./pages/HomePage"), "HomePage");
const BenchmarksPage = lazyNamed(() => import("./pages/BenchmarksPage"), "BenchmarksPage");
const SciencePage = lazy(() => import("./pages/SciencePage"));
const VisionPage = lazyNamed(() => import("./pages/VisionPage"), "VisionPage");
const RoadmapPage = lazyNamed(() => import("./pages/RoadmapPage"), "RoadmapPage");
const DocsPage = lazyNamed(() => import("./pages/DocsPage"), "DocsPage");

function RouteFallback() {
  return (
    <div
      style={{
        minHeight: "70vh",
        display: "grid",
        placeItems: "center",
        padding: "0 24px",
        background: "var(--void)",
      }}
    >
      <div
        style={{
          width: "100%",
          maxWidth: 860,
          borderRadius: 24,
          border: "1px solid rgba(103,232,249,0.12)",
          background:
            "linear-gradient(180deg, rgba(103,232,249,0.05), rgba(255,255,255,0.015))",
          padding: "40px 32px",
        }}
      >
        <div
          style={{
            width: 120,
            height: 10,
            borderRadius: 9999,
            background: "rgba(103,232,249,0.18)",
            marginBottom: 18,
          }}
        />
        <div
          style={{
            width: "68%",
            height: 28,
            borderRadius: 10,
            background: "rgba(255,255,255,0.08)",
            marginBottom: 14,
          }}
        />
        <div
          style={{
            width: "100%",
            height: 14,
            borderRadius: 9999,
            background: "rgba(255,255,255,0.06)",
            marginBottom: 10,
          }}
        />
        <div
          style={{
            width: "92%",
            height: 14,
            borderRadius: 9999,
            background: "rgba(255,255,255,0.06)",
          }}
        />
      </div>
    </div>
  );
}

export function App() {
  const { pathname } = useLocation();

  /* Scroll to top on every route change */
  useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: "instant" });
  }, [pathname]);

  return (
    <Layout>
      <Suspense fallback={<RouteFallback />}>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/benchmarks" element={<BenchmarksPage />} />
          <Route path="/science" element={<SciencePage />} />
          <Route path="/vision" element={<VisionPage />} />
          <Route path="/roadmap" element={<RoadmapPage />} />
          <Route path="/docs" element={<DocsPage />} />
        </Routes>
      </Suspense>
    </Layout>
  );
}
