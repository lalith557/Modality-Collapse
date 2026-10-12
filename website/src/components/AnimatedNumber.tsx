import { useAnimatedNumber } from "../lib/useAnimatedNumber";

interface Props {
  value: number;
  /** decimal places */
  digits?: number;
  /** show an explicit +/− sign (real minus glyph) */
  signed?: boolean;
  prefix?: string;
  suffix?: string;
  className?: string;
}

/** A measurement value that tweens smoothly when `value` changes. */
export function AnimatedNumber({ value, digits = 3, signed = false, prefix = "", suffix = "", className = "" }: Props) {
  const v = useAnimatedNumber(value);
  let s = Math.abs(v).toFixed(digits);
  const sign = v < 0 ? "−" : signed ? "+" : "";
  return (
    <span className={`measure ${className}`}>
      {prefix}
      {sign}
      {s}
      {suffix}
    </span>
  );
}
