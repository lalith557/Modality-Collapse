import { useEffect, useRef, useState } from "react";

const DIGITS = "0123456789";

function DigitColumn({ digit, prevDigit }: { digit: string; prevDigit: string }) {
  const isNum = DIGITS.includes(digit);
  const toIdx = DIGITS.indexOf(digit);
  const fromIdx = DIGITS.indexOf(prevDigit === "" ? digit : prevDigit);

  const [offset, setOffset] = useState(toIdx * -1.2);

  useEffect(() => {
    if (!isNum) return;
    if (fromIdx === toIdx) {
      setOffset(toIdx * -1.2);
      return;
    }
    setOffset(fromIdx * -1.2);
    const id = requestAnimationFrame(() =>
      requestAnimationFrame(() => setOffset(toIdx * -1.2))
    );
    return () => cancelAnimationFrame(id);
  }, [digit, fromIdx, toIdx, isNum]);

  if (!isNum) return <span>{digit}</span>;

  return (
    <span
      className="inline-block overflow-hidden"
      style={{ height: "1.2em", verticalAlign: "bottom" }}
    >
      <span
        className="flex flex-col"
        style={{
          transform: `translateY(${offset}em)`,
          transition: "transform 0.4s cubic-bezier(0.22, 1, 0.36, 1)",
        }}
      >
        {DIGITS.split("").map((d) => (
          <span key={d} className="block" style={{ height: "1.2em", lineHeight: "1.2em" }}>
            {d}
          </span>
        ))}
      </span>
    </span>
  );
}

interface SlidingNumberProps {
  value: number;
  digits?: number;
  signed?: boolean;
  className?: string;
}

export function SlidingNumber({
  value,
  digits = 0,
  signed = false,
  className = "",
}: SlidingNumberProps) {
  const formatted =
    digits > 0 ? Math.abs(value).toFixed(digits) : String(Math.abs(Math.round(value)));

  const prevRef = useRef(formatted);
  const [prevStr, setPrevStr] = useState(formatted);

  useEffect(() => {
    setPrevStr(prevRef.current);
    prevRef.current = formatted;
  }, [formatted]);

  const chars = formatted.split("");
  const prevChars = prevStr.split("");

  return (
    <span className={`inline-flex ${className}`} aria-label={String(value)}>
      {signed && <span className="mr-[0.05em]">{value < 0 ? "−" : "+"}</span>}
      {chars.map((ch, i) => (
        <DigitColumn key={i} digit={ch} prevDigit={prevChars[i] ?? ""} />
      ))}
    </span>
  );
}
