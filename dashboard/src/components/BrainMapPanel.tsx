import { Suspense, lazy } from "react";
import { useEngramStore } from "../store";
import { AtlasView } from "./graph/AtlasView";
import { RegionView } from "./graph/RegionView";

const GraphExplorer = lazy(() =>
  import("./GraphExplorer").then((module) => ({
    default: module.GraphExplorer,
  })),
);

export function BrainMapPanel() {
  const brainMapScope = useEngramStore((s) => s.brainMapScope);

  if (brainMapScope === "atlas") {
    return <AtlasView />;
  }

  if (brainMapScope === "region") {
    return <RegionView />;
  }

  return (
    <Suspense fallback={null}>
      <GraphExplorer />
    </Suspense>
  );
}
