import { useTheme } from "../../context/theme-context";

interface AnimatedBorderProps {
  children: React.ReactNode;
  className?: string;
  active?: boolean;
}

export function AnimatedBorder({ children, className = "", active = true }: AnimatedBorderProps) {
  const { theme } = useTheme();
  const trackBg = theme === "dark" ? "rgba(20,27,34,0.95)" : "rgba(252,251,248,0.95)";

  return (
    <div className={`relative rounded-md ${className}`} style={{ isolation: "isolate" }}>
      {active && (
        <div
          className="pointer-events-none absolute -inset-[1.5px] z-0 rounded-[inherit]"
          style={{
            background: `conic-gradient(from var(--spin-angle, 0deg), transparent 0%, transparent 45%, var(--forward) 65%, var(--gradient) 82%, transparent 100%)`,
            animation: "spin-border 6s linear infinite",
          }}
        />
      )}
      <div
        className="relative z-10 rounded-[inherit]"
        style={{ background: active ? trackBg : undefined }}
      >
        {children}
      </div>
    </div>
  );
}
