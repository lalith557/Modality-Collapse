import { useRef, useState } from "react";

interface MorphingPopoverProps {
  trigger: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  side?: "top" | "bottom";
}

export function MorphingPopover({
  trigger,
  children,
  className = "",
  side = "top",
}: MorphingPopoverProps) {
  const [open, setOpen] = useState(false);
  const timeoutRef = useRef<ReturnType<typeof setTimeout>>();

  const show = () => {
    clearTimeout(timeoutRef.current);
    setOpen(true);
  };
  const hide = () => {
    timeoutRef.current = setTimeout(() => setOpen(false), 100);
  };

  const posClass =
    side === "top"
      ? "bottom-full mb-2.5"
      : "top-full mt-2.5";
  const originClass = side === "top" ? "origin-bottom" : "origin-top";
  const arrowClass =
    side === "top"
      ? "bottom-[-5px] border-b border-r"
      : "top-[-5px] border-t border-l";

  return (
    <span
      className="relative inline-block"
      onMouseEnter={show}
      onMouseLeave={hide}
      onFocus={show}
      onBlur={hide}
    >
      {trigger}
      <span
        role="tooltip"
        className={[
          "absolute left-1/2 z-50 w-64 -translate-x-1/2",
          posClass,
          "rounded-lg border border-line bg-surface p-3 shadow-panel",
          "text-2xs font-normal leading-relaxed text-ink/85",
          originClass,
          "transition-all duration-200 ease-out",
          open
            ? "pointer-events-auto opacity-100 scale-100 blur-none translate-y-0"
            : "pointer-events-none opacity-0 scale-95 blur-sm translate-y-1",
          className,
        ].join(" ")}
        style={{ filter: open ? "blur(0)" : "blur(2px)" }}
      >
        {children}
        <span
          className={`absolute left-1/2 -translate-x-1/2 h-2.5 w-2.5 rotate-45 border-line bg-surface ${arrowClass}`}
        />
      </span>
    </span>
  );
}
