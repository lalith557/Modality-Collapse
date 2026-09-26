import type { ReactNode } from "react";

export interface TabDef {
  id: string;
  label: string;
}

/** Underline tab bar (used for Adam/SGD, CREMA-D/UTD-MHAD, etc.). */
export function Tabs({
  tabs,
  active,
  onChange,
  className = "",
}: {
  tabs: TabDef[];
  active: string;
  onChange: (id: string) => void;
  className?: string;
}) {
  return (
    <div className={`flex items-center gap-6 border-b border-line ${className}`} role="tablist">
      {tabs.map((t) => (
        <button
          key={t.id}
          role="tab"
          aria-selected={active === t.id}
          data-active={active === t.id}
          onClick={() => onChange(t.id)}
          className="tab"
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}

export function TabPanel({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={`animate-fade-in ${className}`}>{children}</div>;
}
