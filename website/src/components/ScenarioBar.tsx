import { useLocation } from "react-router-dom";
import { useResearch } from "../state/research-context";
import { LABELS } from "../data/research";
import { usePublishedPageContext } from "../state/page-context";

function Chip({
  k,
  v,
  tone = "ink",
}: {
  k: string;
  v: string;
  tone?: "ink" | "forward" | "gradient";
}) {
  const c = {
    ink: "text-ink",
    forward: "text-forward",
    gradient: "text-gradient",
  }[tone];
  return (
    <span className="flex items-baseline gap-1.5 whitespace-nowrap">
      <span className="text-[0.6rem] uppercase tracking-wider text-muted/70">{k}</span>
      <span className={`font-mono text-2xs font-medium ${c}`}>{v}</span>
    </span>
  );
}

const SHELL =
  "fixed inset-x-0 bottom-0 z-30 border-t border-line bg-surface/95 backdrop-blur";
const INNER =
  "mx-auto flex max-w-content items-center gap-x-4 gap-y-1 overflow-x-auto px-5 py-2 sm:px-8 lg:px-12";

/**
 * Persistent bottom strip.
 *
 * A page whose figures come from fixed frozen experiments publishes its own context
 * (usePageContext) and the bar shows exactly that. Otherwise the bar shows the
 * Experiment Lab's scenario, labelled as such so it is never read as this page's data.
 */
export function ScenarioBar() {
  const s = useResearch();
  const { pathname } = useLocation();
  const page = usePublishedPageContext();

  if (page) {
    return (
      <div className={SHELL} data-testid="context-bar">
        <div className={INNER}>
          <span className="hidden shrink-0 items-center gap-1.5 font-mono text-[0.6rem] uppercase tracking-label text-muted sm:flex">
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-forward" />
            {page.title}
          </span>
          <div className="flex flex-1 items-center gap-x-4 gap-y-1">
            {page.chips.map((c, i) => (
              <span key={c.k} className="contents">
                {i > 0 && <span className="text-line">·</span>}
                <Chip k={c.k} v={c.v} tone={c.tone} />
              </span>
            ))}
            {page.note && (
              <>
                <span className="hidden text-line sm:inline">·</span>
                <span className="hidden font-mono text-2xs text-muted/60 sm:inline">{page.note}</span>
              </>
            )}
          </div>
        </div>
      </div>
    );
  }

  if (pathname === "/real-world") {
    return (
      <div className={SHELL}>
        <div className={INNER}>
          <span className="hidden shrink-0 items-center gap-1.5 font-mono text-[0.6rem] uppercase tracking-label text-muted sm:flex">
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-forward" />
            Analysis Context
          </span>
          <div className="flex flex-1 items-center gap-x-4 gap-y-1">
            <Chip k="Datasets" v="CREMA-D + UTD-MHAD" />
            <span className="text-line">·</span>
            <Chip k="Optimizer" v="Adam" tone="forward" />
            <span className="text-line">·</span>
            <Chip k="Evidence" v="5 seeds" />
            <span className="text-line">·</span>
            <Chip k="Arms" v="Forward · Gradient · Ordinary" />
            <span className="hidden text-line sm:inline">·</span>
            <span className="hidden font-mono text-2xs text-muted/60 sm:inline">
              Frozen experimental results · No live training
            </span>
          </div>
        </div>
      </div>
    );
  }

  const interTone =
    s.intervention === "forward"
      ? "forward"
      : s.intervention === "gradient"
      ? "gradient"
      : "ink";

  return (
    <div className={SHELL} data-testid="context-bar">
      <div className={INNER}>
        <span className="hidden shrink-0 items-center gap-1.5 font-mono text-[0.6rem] uppercase tracking-label text-muted sm:flex">
          <span className="inline-block h-1.5 w-1.5 rounded-full bg-forward" />
          {pathname === "/lab" ? "Scenario" : "Experiment Lab scenario"}
        </span>
        <div className="flex flex-1 items-center gap-x-4 gap-y-1">
          <Chip k="Data" v={LABELS.dataset[s.dataset]} />
          <span className="text-line">·</span>
          <Chip k="Opt" v={LABELS.optimizer[s.optimizer]} />
          <span className="text-line">·</span>
          <Chip
            k="Arm"
            v={LABELS.intervention[s.intervention]}
            tone={interTone}
          />
          <span className="text-line">·</span>
          <Chip k="α" v={`${s.scale}×`} />
          <span className="hidden text-line sm:inline">·</span>
          <span className="hidden sm:contents">
            <Chip k="Fusion" v={LABELS.fusion[s.fusion]} />
            <span className="text-line">·</span>
            <Chip k="Seed" v={String(s.seed)} />
          </span>
        </div>
      </div>
    </div>
  );
}
