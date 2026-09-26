import { Microscope, FlaskConical, Compass, AlertTriangle } from "lucide-react";

interface Finding {
  n: string;
  title: string;
  observation: string;
  evidence: string;
  interpretation: string;
  limitation: string;
}

const FIELDS = [
  { key: "observation", label: "Observation", icon: Microscope, color: "#17232B" },
  { key: "evidence", label: "Evidence", icon: FlaskConical, color: "#557A95" },
  { key: "interpretation", label: "Interpretation", icon: Compass, color: "#547A67" },
  { key: "limitation", label: "Limitation", icon: AlertTriangle, color: "#B08354" },
] as const;

/**
 * A finding presented as a scientific article entry — Observation / Evidence /
 * Interpretation / Limitation — never a marketing claim.
 */
export function FindingBlock({ finding }: { finding: Finding }) {
  return (
    <article className="animate-fade-up border-t border-ink/15 pt-7">
      <div className="grid gap-6 lg:grid-cols-[auto_1fr]">
        <div className="measure text-[2.5rem] font-semibold leading-none text-line lg:w-16">{finding.n}</div>
        <div>
          <h3 className="max-w-2xl font-display text-xl font-semibold leading-snug text-ink">{finding.title}</h3>
          <dl className="mt-5 grid gap-x-8 gap-y-5 sm:grid-cols-2">
            {FIELDS.map((f) => {
              const Icon = f.icon;
              return (
                <div key={f.key}>
                  <dt className="mb-1.5 flex items-center gap-2">
                    <Icon size={13} style={{ color: f.color }} strokeWidth={2} />
                    <span className="label" style={{ color: f.color }}>
                      {f.label}
                    </span>
                  </dt>
                  <dd className="text-[0.9rem] leading-relaxed text-ink/85">{finding[f.key]}</dd>
                </div>
              );
            })}
          </dl>
        </div>
      </div>
    </article>
  );
}
