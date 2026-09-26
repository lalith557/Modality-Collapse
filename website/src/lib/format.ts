/** Formatting helpers so numbers read as scientific measurements. */

/** Signed fixed-decimal, e.g. +0.259 / −0.070. Uses a real minus sign. */
export function signed(x: number, digits = 3): string {
  const s = x.toFixed(digits);
  if (x > 0) return "+" + s;
  return s.replace("-", "−");
}

/** Fixed decimal with a typographic minus. */
export function fixed(x: number, digits = 3): string {
  return x.toFixed(digits).replace("-", "−");
}

/** p-value in compact scientific form: 0.30, 1.1×10⁻³, 8×10⁻⁵. */
export function pValue(p: number): string {
  if (p >= 0.01) return p.toFixed(2);
  const exp = Math.floor(Math.log10(p));
  const mant = p / Math.pow(10, exp);
  const mantStr = Math.abs(mant - Math.round(mant)) < 0.05 ? String(Math.round(mant)) : mant.toFixed(1);
  return `${mantStr}×10${superscript(exp)}`;
}

/** Multiplier, e.g. ≈ 282× or 1.61×. */
export function times(x: number, digits = x >= 100 ? 0 : 2): string {
  return `${x.toFixed(digits)}×`;
}

const SUP: Record<string, string> = {
  "-": "⁻",
  "0": "⁰",
  "1": "¹",
  "2": "²",
  "3": "³",
  "4": "⁴",
  "5": "⁵",
  "6": "⁶",
  "7": "⁷",
  "8": "⁸",
  "9": "⁹",
};

export function superscript(n: number): string {
  return String(n)
    .split("")
    .map((c) => SUP[c] ?? c)
    .join("");
}
