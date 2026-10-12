import { useEffect } from "react";
import { Outlet, useLocation } from "react-router-dom";
import { TopNavbar } from "./ui/top-navbar";
import { ScenarioBar } from "./ScenarioBar";

export function Layout() {
  const { pathname } = useLocation();

  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "auto" });
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
            <span>Forward or Gradient? — interactive research artifact</span>
            <span>Values transcribed from the manuscript · not a live experiment</span>
          </div>
        </footer>
      </main>
      <ScenarioBar />
    </div>
  );
}
