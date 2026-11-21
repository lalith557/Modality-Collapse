import { useEffect } from "react";
import { Outlet, useLocation } from "react-router-dom";
import { TopNavbar } from "./ui/top-navbar";
import { ScenarioBar } from "./ScenarioBar";

export function Layout() {
  const { pathname, search } = useLocation();
  const studio = pathname === "/studio";

  useEffect(() => {
    // pages opened at a section (?section=...) scroll themselves (useSectionScroll)
    if (!new URLSearchParams(search).has("section")) window.scrollTo({ top: 0, behavior: "auto" });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pathname]);

  return (
    <div
      className="min-h-screen transition-colors duration-300"
      style={{ backgroundColor: "var(--canvas)", color: "var(--ink)" }}
    >
      <TopNavbar />

      {/* Main content — padded below fixed navbar */}
      <main className="mx-auto max-w-[1600px] px-4 pb-28 pt-20 sm:px-6 lg:px-10 lg:pt-[72px]">
        <Outlet />
        <footer className="mt-20 border-t pt-6" style={{ borderColor: "var(--line)" }}>
          <div
            className="flex flex-col gap-2 font-mono text-2xs sm:flex-row sm:items-center sm:justify-between"
            style={{ color: "var(--muted)" }}
          >
            <span>Feature Scale as a Hidden Learning Rate — interactive research artifact</span>
            <span>
              {studio
                ? "Live inference from saved baseline checkpoints · research prototype"
                : pathname === "/architecture"
                  ? "Architecture generated from the model code and checkpoints · no inference on this page"
                  : "Values transcribed from the manuscript and frozen result artifacts · not a live experiment"}
            </span>
          </div>
        </footer>
      </main>
      {!studio && <ScenarioBar />}
    </div>
  );
}
