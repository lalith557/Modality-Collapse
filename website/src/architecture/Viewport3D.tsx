/** Lazily-loaded React wrapper around ArchitectureScene (this module pulls in three.js). */
import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";
import { useTheme } from "../context/theme-context";
import { ArchitectureScene, type SceneState, type SceneTheme, type ViewPreset } from "./ArchitectureScene";
import type { ArchGraph } from "./model";

export interface ViewportHandle {
  resetCamera: () => void;
  fitView: () => void;
  /** Fly to a node id, a figure component ("concept:…") or a stage ("stage:…"). */
  focus: (id: string) => void;
  setView: (v: ViewPreset) => void;
  setAutoRotate: (on: boolean) => void;
}

interface Props {
  graph: ArchGraph;
  state: SceneState;
  onHover: (id: string | null, x: number, y: number) => void;
  onSelect: (id: string | null) => void;
  onError: (message: string) => void;
  onOrbitChange?: (on: boolean) => void;
}

function luminance(hex: string) {
  const m = hex.replace("#", "").match(/^([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})/i);
  if (!m) return 1;
  const [r, g, b] = m.slice(1).map((x) => parseInt(x, 16) / 255);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function readTheme(): SceneTheme {
  const css = getComputedStyle(document.documentElement);
  const v = (k: string, fallback: string) => css.getPropertyValue(k).trim() || fallback;
  const surface = v("--surface", "#fcfbf8");
  return {
    ink: v("--ink", "#17232b"),
    muted: v("--muted", "#687278"),
    line: v("--line", "#dcd9d1"),
    surface,
    a: v("--fig-3d-a", "#5b8bb8"),
    b: v("--fig-3d-b", "#cf8a4e"),
    joint: v("--neutral", "#7a8589"),
    forward: v("--forward", "#557a95"),
    gradient: v("--gradient", "#9a7358"),
    warning: v("--warning", "#b08354"),
    arrow: v("--fig-arrow", "#5d4636"),
    sphere: v("--fig-sphere", "#c48ba8"),
    panel: v("--fig-panel", "#f6efe8"),
    border: v("--fig-border", "#e2d3c4"),
    card: v("--fig-card", "#fdfaf6"),
    zoneA: v("--fig-a", "#e8eef5"),
    zoneB: v("--fig-b", "#f9ede1"),
    zoneInt: v("--fig-int", "#ecd9df"),
    zoneFus: v("--fig-fus", "#e9efe1"),
    intInk: v("--fig-int-ink", "#8a5a70"),
    fusInk: v("--fig-fus-ink", "#5f7a52"),
    chip: v("--fig-chip", "#b98258"),
    dark: luminance(surface) < 0.4,
  };
}

const Viewport3D = forwardRef<ViewportHandle, Props>(function Viewport3D(
  { graph, state, onHover, onSelect, onError, onOrbitChange },
  ref,
) {
  const host = useRef<HTMLDivElement>(null);
  const scene = useRef<ArchitectureScene | null>(null);
  const cbs = useRef({ onHover, onSelect, onOrbitChange });
  cbs.current = { onHover, onSelect, onOrbitChange };
  const { theme } = useTheme();

  useImperativeHandle(ref, () => ({
    resetCamera: () => scene.current?.resetCamera(),
    fitView: () => scene.current?.fitView(),
    focus: (id) => scene.current?.focus(id),
    setView: (v) => scene.current?.setView(v),
    setAutoRotate: (on) => scene.current?.setAutoRotate(on),
  }));

  useEffect(() => {
    if (!host.current) return;
    try {
      scene.current = new ArchitectureScene(host.current, {
        onHover: (id, x, y) => cbs.current.onHover(id, x, y),
        onSelect: (id) => cbs.current.onSelect(id),
        onOrbitChange: (on) => cbs.current.onOrbitChange?.(on),
      });
    } catch (e) {
      onError(e instanceof Error ? e.message : "WebGL could not be initialised.");
      return;
    }
    return () => {
      scene.current?.dispose();
      scene.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    scene.current?.setGraph(graph, readTheme());
    scene.current?.setState(state);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [graph]);

  useEffect(() => {
    scene.current?.setState(state);
  }, [state]);

  useEffect(() => {
    // the theme class is toggled by the provider's effect; read the variables after it ran
    const id = requestAnimationFrame(() => scene.current?.setTheme(readTheme()));
    return () => cancelAnimationFrame(id);
  }, [theme]);

  return (
    <div
      ref={host}
      className="relative h-full w-full touch-none select-none"
      data-testid="viewport-3d"
      role="img"
      aria-label={`Interactive 360° 3D view of the ${graph.name} model workflow. Use the stage buttons, the layer list or the 2D view for keyboard access.`}
    />
  );
});

export default Viewport3D;
