import { useEffect, useRef } from "react";
import { useTheme } from "../../context/theme-context";

interface Dot {
  x: number;
  y: number;
  t: number;
}

export function CursorTrail({
  fadeDuration = 900,
  dotSize = 5,
}: {
  fadeDuration?: number;
  dotSize?: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dotsRef = useRef<Dot[]>([]);
  const rafRef = useRef<number>(0);
  const { theme } = useTheme();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const trailColor = theme === "dark" ? { r: 107, g: 154, b: 181 } : { r: 85, g: 122, b: 149 };

    function resize() {
      if (!canvas) return;
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    }
    resize();

    const ro = new ResizeObserver(resize);
    ro.observe(document.documentElement);

    function onMove(e: MouseEvent) {
      dotsRef.current.push({ x: e.clientX, y: e.clientY, t: Date.now() });
      if (dotsRef.current.length > 120) dotsRef.current.shift();
    }
    window.addEventListener("mousemove", onMove, { passive: true });

    function draw() {
      if (!canvas || !ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const now = Date.now();
      dotsRef.current = dotsRef.current.filter((d) => now - d.t < fadeDuration);

      for (const d of dotsRef.current) {
        const age = now - d.t;
        const frac = 1 - age / fadeDuration;
        const radius = dotSize * frac * 0.9;
        ctx.beginPath();
        ctx.arc(d.x, d.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${trailColor.r},${trailColor.g},${trailColor.b},${frac * 0.5})`;
        ctx.fill();
      }
      rafRef.current = requestAnimationFrame(draw);
    }
    draw();

    return () => {
      ro.disconnect();
      window.removeEventListener("mousemove", onMove);
      cancelAnimationFrame(rafRef.current);
    };
  }, [fadeDuration, dotSize, theme]);

  return (
    <canvas
      ref={canvasRef}
      className="pointer-events-none fixed inset-0 z-[9998]"
      style={{ mixBlendMode: theme === "dark" ? "screen" : "multiply" }}
    />
  );
}
