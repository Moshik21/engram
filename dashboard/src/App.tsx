import { useEffect } from "react";
import { DashboardShell } from "./components/DashboardShell";
import { useEngramStore } from "./store";

export default function App() {
  const darkMode = useEngramStore((s) => s.darkMode);
  const loadInitialGraph = useEngramStore((s) => s.loadInitialGraph);

  useEffect(() => {
    loadInitialGraph();
  }, [loadInitialGraph]);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", darkMode);
  }, [darkMode]);

  return <DashboardShell />;
}
