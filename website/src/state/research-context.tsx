import { createContext, useContext, useState, useCallback, useEffect, type ReactNode } from "react";
import type { DatasetId, OptimizerId, InterventionId, FusionId, ScaleId } from "../data/research";

/** The single scenario the whole app explores. */
export interface Scenario {
  dataset: DatasetId;
  optimizer: OptimizerId;
  intervention: InterventionId;
  scale: ScaleId;
  fusion: FusionId;
  seed: number;
}

const DEFAULT: Scenario = {
  dataset: "synthetic",
  optimizer: "adam",
  intervention: "forward",
  scale: 100,
  fusion: "sum",
  seed: 1,
};

interface Ctx extends Scenario {
  set: <K extends keyof Scenario>(key: K, value: Scenario[K]) => void;
  setScenario: (partial: Partial<Scenario>) => void;
  reset: () => void;
}

const ResearchCtx = createContext<Ctx | null>(null);
const KEY = "fog-scenario";

function load(): Scenario {
  try {
    const raw = sessionStorage.getItem(KEY);
    if (raw) return { ...DEFAULT, ...JSON.parse(raw) };
  } catch {
    /* ignore */
  }
  return DEFAULT;
}

export function ResearchProvider({ children }: { children: ReactNode }) {
  const [scenario, setState] = useState<Scenario>(load);

  useEffect(() => {
    try {
      sessionStorage.setItem(KEY, JSON.stringify(scenario));
    } catch {
      /* ignore */
    }
  }, [scenario]);

  const set = useCallback(
    <K extends keyof Scenario>(key: K, value: Scenario[K]) =>
      setState((s) => {
        const next = { ...s, [key]: value };
        // CREMA-D and UTD-MHAD were only run under Adam; keep scenarios valid.
        if (key === "dataset" && value !== "synthetic") next.optimizer = "adam";
        return next;
      }),
    [],
  );
  const setScenario = useCallback((partial: Partial<Scenario>) => setState((s) => ({ ...s, ...partial })), []);
  const reset = useCallback(() => setState(DEFAULT), []);

  return <ResearchCtx.Provider value={{ ...scenario, set, setScenario, reset }}>{children}</ResearchCtx.Provider>;
}

export function useResearch() {
  const ctx = useContext(ResearchCtx);
  if (!ctx) throw new Error("useResearch must be used within ResearchProvider");
  return ctx;
}
