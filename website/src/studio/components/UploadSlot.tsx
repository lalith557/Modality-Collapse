import { useRef, useState } from "react";
import { Upload, X } from "lucide-react";
import { ACCEPT, LIMITS_MB, formatBytes, type Role } from "../validation";

export function UploadSlot({
  role,
  label,
  hint,
  file,
  error,
  required,
  onChange,
}: {
  role: Role;
  label: string;
  hint: string;
  file: File | null;
  error: string | null;
  required?: boolean;
  onChange: (f: File | null) => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  const id = `upload-${role}`;
  return (
    <div>
      <div className="mb-2 flex items-baseline justify-between">
        <label htmlFor={id} className="label">
          {label} {required ? <span className="text-critical">*</span> : <span className="normal-case">(optional)</span>}
        </label>
        <span className="font-mono text-2xs text-muted">
          {ACCEPT[role].join(" ")} · ≤{LIMITS_MB[role]} MB
        </span>
      </div>
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          const f = e.dataTransfer.files?.[0];
          if (f) onChange(f);
        }}
        data-testid={`slot-${role}`}
        className={`rounded-md border border-dashed p-4 transition-colors ${
          error ? "border-critical/60 bg-critical/[0.04]" : drag ? "border-forward bg-forward/[0.05]" : "border-line bg-surface"
        }`}
      >
        {file ? (
          <div className="flex items-center justify-between gap-3">
            <div className="min-w-0">
              <div className="truncate text-sm font-medium text-ink">{file.name}</div>
              <div className="font-mono text-2xs text-muted">
                {formatBytes(file.size)} · {file.type || "unknown type"}
              </div>
            </div>
            <button
              type="button"
              aria-label={`Remove ${label}`}
              className="btn-ghost !px-2 !py-1"
              onClick={() => {
                onChange(null);
                if (input.current) input.current.value = "";
              }}
            >
              <X size={14} />
            </button>
          </div>
        ) : (
          <button
            type="button"
            className="flex w-full items-center gap-3 text-left"
            onClick={() => input.current?.click()}
          >
            <Upload size={16} className="shrink-0 text-muted" />
            <span className="text-sm text-muted">{hint}</span>
          </button>
        )}
        <input
          id={id}
          ref={input}
          type="file"
          className="hidden"
          accept={ACCEPT[role].join(",")}
          onChange={(e) => onChange(e.target.files?.[0] ?? null)}
        />
      </div>
      {error && (
        <p role="alert" className="mt-1.5 text-2xs text-critical">
          {error}
        </p>
      )}
    </div>
  );
}
