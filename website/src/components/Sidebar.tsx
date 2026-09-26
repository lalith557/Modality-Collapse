import { NavLink } from "react-router-dom";
import { nav, groupLabels, type NavItem } from "../data/nav";

const ORDER: NavItem["group"][] = ["overview", "experiments", "evidence", "reference"];

export function SidebarContent({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <div className="flex h-full flex-col">
      {/* Wordmark */}
      <div className="px-5 pb-6 pt-6">
        <div className="flex items-center gap-2.5">
          <span className="flex h-9 w-9 items-center justify-center rounded border border-white/15 font-display text-sm font-semibold tracking-tight text-canvas">
            <span className="text-forward">F</span>
            <span className="px-[1px] text-white/30">/</span>
            <span className="text-gradient">G</span>
          </span>
          <div className="leading-tight">
            <div className="font-display text-[0.95rem] font-semibold text-canvas">Forward or Gradient?</div>
            <div className="font-mono text-[0.6rem] uppercase tracking-label text-white/40">
              Modality contribution
            </div>
          </div>
        </div>
      </div>

      <nav className="flex-1 space-y-6 overflow-y-auto px-3 pb-6">
        {ORDER.map((group) => (
          <div key={group}>
            <div className="px-2 pb-2 font-mono text-[0.6rem] uppercase tracking-label text-white/35">
              {groupLabels[group]}
            </div>
            <ul className="space-y-0.5">
              {nav
                .filter((n) => n.group === group)
                .map((item) => {
                  const Icon = item.icon;
                  return (
                    <li key={item.path}>
                      <NavLink
                        to={item.path}
                        end={item.path === "/"}
                        onClick={onNavigate}
                        className={({ isActive }) =>
                          [
                            "group flex items-center gap-3 rounded px-2.5 py-2 text-[0.85rem] transition-colors",
                            isActive
                              ? "bg-white/[0.07] text-canvas"
                              : "text-white/60 hover:bg-white/[0.035] hover:text-white/90",
                          ].join(" ")
                        }
                      >
                        {({ isActive }) => (
                          <>
                            <Icon
                              size={16}
                              strokeWidth={1.75}
                              className={isActive ? "text-forward" : "text-white/45 group-hover:text-white/70"}
                            />
                            <span>{item.label}</span>
                          </>
                        )}
                      </NavLink>
                    </li>
                  );
                })}
            </ul>
          </div>
        ))}
      </nav>

      <div className="border-t border-white/10 px-5 py-4">
        <div className="font-mono text-[0.6rem] leading-relaxed text-white/35">
          n = 5 seeds · fixed splits
          <br />
          within-split reproducibility
        </div>
      </div>
    </div>
  );
}

export function Sidebar() {
  return (
    <aside className="fixed inset-y-0 left-0 z-30 hidden w-[240px] border-r border-black/20 bg-navy lg:block">
      <SidebarContent />
    </aside>
  );
}
