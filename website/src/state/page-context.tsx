/**
 * Lets a page publish ITS OWN evidence context (dataset, optimizer, intervention, scale,
 * seeds) to the bottom context bar, so the bar never shows the Experiment Lab's global
 * scenario on a page whose figures come from a different, fixed experiment.
 */
import { createContext, useContext, useEffect, useState, type ReactNode } from "react";

export interface ContextChip {
  k: string;
  v: string;
  tone?: "ink" | "forward" | "gradient";
}

export interface PageContextValue {
  title: string;
  chips: ContextChip[];
  note?: string;
}

const Ctx = createContext<{ value: PageContextValue | null; set: (v: PageContextValue | null) => void } | null>(null);

export function PageContextProvider({ children }: { children: ReactNode }) {
  const [value, set] = useState<PageContextValue | null>(null);
  return <Ctx.Provider value={{ value, set }}>{children}</Ctx.Provider>;
}

/** Publish this page's context while it is mounted (re-published whenever it changes). */
export function usePageContext(value: PageContextValue) {
  const ctx = useContext(Ctx);
  const key = JSON.stringify(value);
  useEffect(() => {
    ctx?.set(JSON.parse(key));
    return () => ctx?.set(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
}

export function usePublishedPageContext(): PageContextValue | null {
  return useContext(Ctx)?.value ?? null;
}
