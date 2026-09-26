/**
 * Thin topographic/field contour lines used as a restrained backdrop —
 * evokes a measurement field, not decorative "AI" waves.
 */
export function Contour({ className = "" }: { className?: string }) {
  const lines = Array.from({ length: 9 }, (_, i) => 40 + i * 26);
  return (
    <div className={`contour-bg ${className}`} aria-hidden="true">
      <svg viewBox="0 0 400 520" fill="none" preserveAspectRatio="xMidYMid meet">
        <g stroke="#557A95" strokeOpacity="0.16" strokeWidth="1">
          {lines.map((r, i) => (
            <path
              key={i}
              d={`M ${200 - r} 260
                  C ${200 - r} ${260 - r * 0.62}, ${200 - r * 0.36} ${140 - i * 6}, 200 ${150 - i * 4}
                  C ${200 + r * 0.42} ${140 - i * 6}, ${200 + r} ${260 - r * 0.62}, ${200 + r} 260
                  C ${200 + r} ${260 + r * 0.62}, ${200 + r * 0.42} ${380 + i * 6}, 200 ${372 + i * 4}
                  C ${200 - r * 0.36} ${380 + i * 6}, ${200 - r} ${260 + r * 0.62}, ${200 - r} 260 Z`}
            />
          ))}
        </g>
      </svg>
    </div>
  );
}
