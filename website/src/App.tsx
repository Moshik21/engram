import { useEffect } from "react";
import { Routes, Route, useLocation } from "react-router-dom";
import { Layout } from "./components/Layout";

/* --------------------------------------------------------------------------
   Lazy-ish page imports — kept as direct imports for simplicity since
   react-router-dom v7 handles code-splitting at the route level if needed.
   Stub placeholder components until the real pages are built.
   -------------------------------------------------------------------------- */

import { HomePage } from "./pages/HomePage";
import { BenchmarksPage } from "./pages/BenchmarksPage";
import SciencePage from "./pages/SciencePage";
import { VisionPage } from "./pages/VisionPage";
import { RoadmapPage } from "./pages/RoadmapPage";
import { DocsPage } from "./pages/DocsPage";

export function App() {
  const { pathname } = useLocation();

  /* Scroll to top on every route change */
  useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: "instant" });
  }, [pathname]);

  return (
    <Layout>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/benchmarks" element={<BenchmarksPage />} />
        <Route path="/science" element={<SciencePage />} />
        <Route path="/vision" element={<VisionPage />} />
        <Route path="/roadmap" element={<RoadmapPage />} />
        <Route path="/docs" element={<DocsPage />} />
      </Routes>
    </Layout>
  );
}
