import { createContext, useContext, useState, type ReactNode } from "react";
import type { PredictionResponse } from "./types";

/** Holds the most recent real prediction so other pages (Methodology) can trace it. In-memory only. */
interface StudioState {
  last: { response: PredictionResponse; inputs: string[] } | null;
  setLast: (v: { response: PredictionResponse; inputs: string[] } | null) => void;
}

const Ctx = createContext<StudioState>({ last: null, setLast: () => {} });

export function StudioProvider({ children }: { children: ReactNode }) {
  const [last, setLast] = useState<StudioState["last"]>(null);
  return <Ctx.Provider value={{ last, setLast }}>{children}</Ctx.Provider>;
}

export const useStudio = () => useContext(Ctx);
