interface PathwayDiagramProps {
  animated?: boolean;
  alpha?: string;
  className?: string;
}

/**
 * The central conceptual decomposition:
 *   FEATURE SCALE α → { forward representation magnitude | gradient / update }
 *   → modality contribution.
 * Muted blue = forward, muted brown = gradient. Thin scientific connectors.
 * This is a conceptual diagram, not an identified causal graph.
 */
export function PathwayDiagram({ animated = true, alpha, className = "" }: PathwayDiagramProps) {
  const flow = animated ? "pw-flow" : "";
  return (
    <svg
      viewBox="0 0 560 440"
      className={`h-auto w-full ${className}`}
      role="img"
      aria-label="Feature scale splits into a forward representation-magnitude pathway and a gradient/parameter-update pathway, which recombine into modality contribution."
    >
      {/* connectors ---------------------------------------------------- */}
      <g fill="none" stroke="#B9B5AA" strokeWidth="1.25">
        {/* alpha down to split */}
        <path d="M280 92 V128" />
        {/* split */}
        <path d="M280 128 H150 V158" />
        <path d="M280 128 H410 V158" />
        {/* forward + gradient down to merge */}
        <path d="M150 232 V300 H280" />
        <path d="M410 232 V300 H280" />
        {/* merge into contribution */}
        <path d="M280 300 V336" />
      </g>
      {/* animated signal flow overlay */}
      <g fill="none" strokeWidth="1.75">
        <path className={flow} d="M280 92 V128" stroke="#7A8589" />
        <path className={flow} d="M280 128 H150 V158" stroke="#557A95" />
        <path className={flow} d="M150 232 V300 H280" stroke="#557A95" />
        <path className={flow} d="M280 128 H410 V158" stroke="#9A7358" />
        <path className={flow} d="M410 232 V300 H280" stroke="#9A7358" />
        <path className={flow} d="M280 300 V336" stroke="#17232B" />
      </g>

      {/* Feature scale node ------------------------------------------- */}
      <text x="280" y="34" textAnchor="middle" className="fill-[#687278] font-mono" fontSize="10.5" letterSpacing="2">
        FEATURE SCALE
      </text>
      <circle cx="280" cy="66" r="24" fill="#FCFBF8" stroke="#17232B" strokeWidth="1.5" />
      <text x="280" y="72" textAnchor="middle" className="fill-[#17232B] font-mono" fontSize="18">
        {alpha ? "α" : "α"}
      </text>
      {alpha && (
        <text x="280" y="66" dx="34" textAnchor="start" className="fill-[#687278] font-mono" fontSize="11" dominantBaseline="middle">
          = {alpha}
        </text>
      )}

      {/* Forward node -------------------------------------------------- */}
      <g>
        <rect x="66" y="158" width="168" height="74" rx="6" fill="#557A95" opacity="0.08" stroke="#557A95" strokeWidth="1.4" />
        <text x="150" y="184" textAnchor="middle" className="fill-[#557A95] font-mono" fontSize="10" letterSpacing="1.5">
          FORWARD PATHWAY
        </text>
        <text x="150" y="205" textAnchor="middle" className="fill-[#17232B]" fontSize="13" fontWeight="600">
          Representation
        </text>
        <text x="150" y="221" textAnchor="middle" className="fill-[#17232B]" fontSize="13" fontWeight="600">
          magnitude
        </text>
      </g>

      {/* Gradient node ------------------------------------------------- */}
      <g>
        <rect x="326" y="158" width="168" height="74" rx="6" fill="#9A7358" opacity="0.08" stroke="#9A7358" strokeWidth="1.4" />
        <text x="410" y="184" textAnchor="middle" className="fill-[#9A7358] font-mono" fontSize="10" letterSpacing="1.5">
          GRADIENT PATHWAY
        </text>
        <text x="410" y="205" textAnchor="middle" className="fill-[#17232B]" fontSize="13" fontWeight="600">
          Parameter
        </text>
        <text x="410" y="221" textAnchor="middle" className="fill-[#17232B]" fontSize="13" fontWeight="600">
          update
        </text>
      </g>

      {/* Contribution node -------------------------------------------- */}
      <g>
        <rect x="164" y="336" width="232" height="64" rx="6" fill="#17232B" />
        <text x="280" y="362" textAnchor="middle" className="fill-white/50 font-mono" fontSize="9.5" letterSpacing="1.5">
          OUTCOME
        </text>
        <text x="280" y="383" textAnchor="middle" className="fill-[#F5F3EE]" fontSize="14.5" fontWeight="600">
          Modality contribution
        </text>
      </g>
    </svg>
  );
}
