import { useEffect, useRef, useState } from "react";

/**
 * Tweens a numeric value toward `target` over `duration` ms with an ease-out.
 * Respects prefers-reduced-motion (snaps instantly). Used so measurement
 * values visibly move when the scenario changes, rather than jumping.
 */
export function useAnimatedNumber(target: number, duration = 450): number {
  const [value, setValue] = useState(target);
  const fromRef = useRef(target);
  const startRef = useRef(0);
  const rafRef = useRef(0);

  useEffect(() => {
    const reduce =
      typeof window !== "undefined" &&
      window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce || duration <= 0) {
      setValue(target);
      return;
    }
    fromRef.current = value;
    startRef.current = performance.now();
    const from = fromRef.current;
    const delta = target - from;

    const tick = (now: number) => {
      const t = Math.min(1, (now - startRef.current) / duration);
      const eased = 1 - Math.pow(1 - t, 3); // ease-out cubic
      setValue(from + delta * eased);
      if (t < 1) rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target, duration]);

  return value;
}
