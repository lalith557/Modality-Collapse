import { useEffect, useRef } from "react";
import { useTheme } from "../../context/theme-context";

const DITHER = [
  [0, 8, 2, 10],
  [12, 4, 14, 6],
  [3, 11, 1, 9],
  [15, 7, 13, 5],
];

function noise(x: number, y: number, t: number): number {
  return (
    Math.sin(x * 0.8 + t * 0.5) * Math.cos(y * 0.6 + t * 0.3) * 0.5 +
    Math.sin(x * 0.3 - y * 0.4 + t * 0.7) * 0.3 +
    Math.sin((x + y) * 0.5 + t * 0.4) * 0.2
  );
}

interface ShaderBackgroundProps {
  className?: string;
  scale?: number;
}

export function ShaderBackground({ className = "", scale = 0.35 }: ShaderBackgroundProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const { theme } = useTheme();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let W = 1, H = 1;
    let imgData: ImageData = ctx.createImageData(1, 1);

    function resize() {
      const pw = canvas!.parentElement?.clientWidth ?? window.innerWidth;
      const ph = canvas!.parentElement?.clientHeight ?? window.innerHeight;
      W = Math.max(1, Math.floor(pw * scale));
      H = Math.max(1, Math.floor(ph * scale));
      canvas!.width = W;
      canvas!.height = H;
      imgData = ctx!.createImageData(W, H);
    }
    resize();

    const ro = new ResizeObserver(resize);
    if (canvas.parentElement) ro.observe(canvas.parentElement);

    let t = 0;
    const isDark = theme === "dark";

    function draw() {
      t += 0.01;
      const data = imgData.data;

      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const n = noise((x / W) * 6, (y / H) * 6, t);
          const v = (n + 1) * 0.5;
          const threshold = DITHER[y & 3][x & 3] / 16;
          const hi = v > threshold;
          const px = (y * W + x) * 4;

          if (isDark) {
            data[px]     = hi ? Math.floor(18 + n * 10) : Math.floor(10 + n * 5);
            data[px + 1] = hi ? Math.floor(32 + n * 14) : Math.floor(18 + n * 7);
            data[px + 2] = hi ? Math.floor(50 + n * 20) : Math.floor(28 + n * 10);
          } else {
            // Lighter, muted teal-cream palette for light mode
            data[px]     = hi ? Math.floor(220 + n * 12) : Math.floor(200 + n * 8);
            data[px + 1] = hi ? Math.floor(225 + n * 10) : Math.floor(208 + n * 7);
            data[px + 2] = hi ? Math.floor(230 + n * 8)  : Math.floor(215 + n * 6);
          }
          data[px + 3] = 255;
        }
      }

      ctx!.putImageData(imgData, 0, 0);
      rafRef.current = requestAnimationFrame(draw);
    }

    draw();

    return () => {
      ro.disconnect();
      cancelAnimationFrame(rafRef.current);
    };
  }, [scale, theme]);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className={`absolute inset-0 h-full w-full ${className}`}
      style={{ imageRendering: "pixelated" }}
    />
  );
}
