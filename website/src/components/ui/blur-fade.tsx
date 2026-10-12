import { useEffect, useRef, useState } from "react";

interface BlurFadeProps {
  children: React.ReactNode;
  delay?: number;
  inView?: boolean;
  className?: string;
  yOffset?: number;
}

export function BlurFade({
  children,
  delay = 0,
  inView = false,
  className = "",
  yOffset = 12,
}: BlurFadeProps) {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(!inView);

  useEffect(() => {
    if (!inView) return;
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true);
          obs.disconnect();
        }
      },
      { threshold: 0.05 }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, [inView]);

  return (
    <div
      ref={ref}
      className={className}
      style={{
        opacity: visible ? 1 : 0,
        transform: visible ? "translateY(0) scale(1)" : `translateY(${yOffset}px) scale(0.99)`,
        filter: visible ? "blur(0px)" : "blur(8px)",
        transition: `opacity 0.55s ease-out ${delay}s, transform 0.55s ease-out ${delay}s, filter 0.55s ease-out ${delay}s`,
        willChange: "opacity, transform, filter",
      }}
    >
      {children}
    </div>
  );
}
