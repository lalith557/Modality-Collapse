import { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { Sun, Moon, Menu, X } from "lucide-react";
import { useTheme } from "../../context/theme-context";
import { nav } from "../../data/nav";

export function TopNavbar() {
  const { pathname } = useLocation();
  const { theme, toggle } = useTheme();
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <>
      {/* Desktop pill navbar */}
      <header className="pointer-events-none fixed inset-x-0 top-0 z-40 flex items-start justify-center px-6 pt-3">
        <nav
          className="pointer-events-auto hidden items-center gap-0.5 rounded-full border border-line bg-surface/90 px-2 py-1.5 shadow-panel backdrop-blur-md lg:flex dark:bg-surface/80 dark:border-line"
          style={{ transition: "background 0.3s ease, border-color 0.3s ease" }}
        >
          {nav.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.path;
            return (
              <Link
                key={item.path}
                to={item.path}
                title={item.label}
                className={[
                  "relative flex items-center gap-1.5 rounded-full px-2.5 py-1.5 text-[0.72rem] font-medium transition-all duration-200",
                  isActive
                    ? "bg-ink text-canvas shadow-sm dark:bg-forward dark:text-canvas"
                    : "text-muted hover:text-ink hover:bg-black/[0.05] dark:hover:bg-white/[0.06] dark:hover:text-ink",
                ].join(" ")}
              >
                <Icon size={12} />
                <span>{item.label}</span>
              </Link>
            );
          })}
          <div className="mx-1 h-4 w-px bg-line" />
          <button
            onClick={toggle}
            aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
            className="flex h-7 w-7 items-center justify-center rounded-full text-muted transition-all duration-200 hover:text-ink hover:bg-black/[0.06] dark:hover:bg-white/[0.07] dark:hover:text-ink"
          >
            {theme === "dark" ? <Sun size={13} /> : <Moon size={13} />}
          </button>
        </nav>

        {/* Mobile top bar */}
        <div className="pointer-events-auto flex w-full items-center justify-between rounded-full border border-line bg-surface/90 px-3 py-2 shadow-panel backdrop-blur-md lg:hidden dark:bg-surface/80">
          <Link to="/" className="flex items-center gap-2">
            <span className="flex h-7 w-7 items-center justify-center rounded-full border border-line bg-navy font-display text-[0.6rem] font-semibold">
              <span className="text-forward">F</span>
              <span className="text-white/30">/</span>
              <span className="text-[#b08a6a]">G</span>
            </span>
            <span className="text-sm font-semibold text-ink">Forward or Gradient?</span>
          </Link>
          <div className="flex items-center gap-2">
            <button
              onClick={toggle}
              aria-label="Toggle dark mode"
              className="flex h-7 w-7 items-center justify-center rounded-full text-muted hover:text-ink"
            >
              {theme === "dark" ? <Sun size={14} /> : <Moon size={14} />}
            </button>
            <button
              aria-label="Open navigation"
              onClick={() => setMobileOpen(true)}
              className="flex h-7 w-7 items-center justify-center rounded-full border border-line text-ink"
            >
              <Menu size={15} />
            </button>
          </div>
        </div>
      </header>

      {/* Mobile drawer */}
      {mobileOpen && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div
            className="absolute inset-0 bg-ink/40 backdrop-blur-sm"
            onClick={() => setMobileOpen(false)}
          />
          <div className="absolute inset-y-0 right-0 w-72 bg-surface dark:bg-surface overflow-y-auto shadow-xl">
            <div className="flex items-center justify-between border-b border-line px-4 py-3">
              <span className="font-mono text-2xs uppercase tracking-label text-muted">Navigation</span>
              <button
                onClick={() => setMobileOpen(false)}
                className="rounded p-1.5 text-muted hover:text-ink"
              >
                <X size={16} />
              </button>
            </div>
            <nav className="p-3">
              {nav.map((item) => {
                const Icon = item.icon;
                const isActive = pathname === item.path;
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    onClick={() => setMobileOpen(false)}
                    className={[
                      "flex items-center gap-3 rounded-md px-3 py-2.5 text-sm font-medium transition-colors",
                      isActive
                        ? "bg-ink/[0.06] text-ink dark:bg-forward/20 dark:text-forward"
                        : "text-muted hover:bg-black/[0.04] hover:text-ink dark:hover:bg-white/[0.04]",
                    ].join(" ")}
                  >
                    <Icon size={15} />
                    {item.label}
                  </Link>
                );
              })}
            </nav>
          </div>
        </div>
      )}
    </>
  );
}
