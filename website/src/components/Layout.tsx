import { useState, useEffect } from "react";
import { Outlet, useLocation, Link } from "react-router-dom";
import { Menu, X } from "lucide-react";
import { Sidebar, SidebarContent } from "./Sidebar";
import { nav } from "../data/nav";

function Breadcrumb() {
  const { pathname } = useLocation();
  const current = nav.find((n) => n.path === pathname);
  return (
    <div className="flex items-center gap-2 font-mono text-2xs uppercase tracking-label text-muted">
      <Link to="/" className="transition-colors hover:text-ink">
        Forward / Gradient
      </Link>
      <span className="text-line">—</span>
      <span className="text-ink">{current?.label ?? "Home"}</span>
    </div>
  );
}

export function Layout() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const { pathname } = useLocation();

  // Scroll to top + close drawer on route change.
  useEffect(() => {
    setMobileOpen(false);
    window.scrollTo({ top: 0, behavior: "auto" });
  }, [pathname]);

  return (
    <div className="min-h-screen bg-canvas">
      <Sidebar />

      {/* Mobile top bar */}
      <div className="sticky top-0 z-40 flex items-center justify-between border-b border-line bg-canvas/95 px-4 py-3 backdrop-blur lg:hidden">
        <Link to="/" className="flex items-center gap-2">
          <span className="flex h-8 w-8 items-center justify-center rounded border border-ink/15 bg-navy font-display text-xs font-semibold">
            <span className="text-forward">F</span>
            <span className="text-white/30">/</span>
            <span className="text-gradient">G</span>
          </span>
          <span className="font-display text-sm font-semibold text-ink">Forward or Gradient?</span>
        </Link>
        <button
          aria-label="Open navigation"
          onClick={() => setMobileOpen(true)}
          className="rounded border border-line p-2 text-ink"
        >
          <Menu size={18} />
        </button>
      </div>

      {/* Mobile drawer */}
      {mobileOpen && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div className="absolute inset-0 bg-ink/40" onClick={() => setMobileOpen(false)} />
          <div className="absolute inset-y-0 left-0 w-[264px] animate-fade-in bg-navy">
            <button
              aria-label="Close navigation"
              onClick={() => setMobileOpen(false)}
              className="absolute right-3 top-4 z-10 rounded p-1.5 text-white/60 hover:text-white"
            >
              <X size={18} />
            </button>
            <SidebarContent onNavigate={() => setMobileOpen(false)} />
          </div>
        </div>
      )}

      {/* Main content */}
      <div className="lg:pl-[240px]">
        <div className="mx-auto max-w-content px-5 py-6 sm:px-8 lg:px-12 lg:py-9">
          <div className="mb-7 hidden lg:block">
            <Breadcrumb />
          </div>
          <Outlet />
          <footer className="mt-20 border-t border-line pt-6">
            <div className="flex flex-col gap-2 font-mono text-2xs text-muted sm:flex-row sm:items-center sm:justify-between">
              <span>Forward or Gradient? — interactive research artifact</span>
              <span>Values transcribed from the manuscript · not a live experiment</span>
            </div>
          </footer>
        </div>
      </div>
    </div>
  );
}
