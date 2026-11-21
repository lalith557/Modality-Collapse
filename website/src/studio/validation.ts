/**
 * Client-side PRE-validation only (fast feedback before upload). The server is the
 * authority: it re-checks every file by its actual content (magic bytes) and size.
 * Limits mirror inference/input_validator.py LIMITS.
 */

export type Role = "audio" | "video" | "depth" | "inertial";

export const LIMITS_MB: Record<Role, number> = { audio: 25, video: 100, depth: 64, inertial: 4 };

export const ACCEPT: Record<Role, string[]> = {
  audio: [".wav", ".flac", ".ogg", ".mp3", ".m4a", ".webm", ".opus", ".mp4"],
  video: [".mp4", ".webm", ".avi", ".flv", ".mov", ".m4v"],
  depth: [".mat", ".npy"],
  inertial: [".mat", ".csv"],
};

/** Human-readable size; small files keep their byte count so a placeholder stub is obvious. */
export function formatBytes(n: number): string {
  if (n < 1000) return `${n} B`;
  if (n < 1e6) return `${(n / 1e3).toFixed(1)} KB`;
  return `${(n / 1e6).toFixed(2)} MB`;
}

export function extensionOf(name: string): string {
  const i = name.lastIndexOf(".");
  return i >= 0 ? name.slice(i).toLowerCase() : "";
}

/** Returns a human-readable problem, or null when the file may be sent. */
export function preValidate(file: File | null, role: Role): string | null {
  if (!file) return null;
  if (file.size === 0) return "The file is empty.";
  if (file.size > LIMITS_MB[role] * 1024 * 1024) {
    return `The file is ${(file.size / 1e6).toFixed(1)} MB; the ${role} limit is ${LIMITS_MB[role]} MB.`;
  }
  const ext = extensionOf(file.name);
  if (!ACCEPT[role].includes(ext)) {
    return `"${ext || "no extension"}" is not an accepted ${role} format (${ACCEPT[role].join(", ")}).`;
  }
  return null;
}

export type CremadPlan =
  | { kind: "none" }
  | { kind: "audio"; path: string }
  | { kind: "video"; path: string }
  | { kind: "multimodal"; path: string; audioSource: "file" | "video_track" };

/** Which endpoint a CREMA-D submission maps to. Multimodal only when both modalities are supplied. */
export function planCremad(audio: File | null, video: File | null, useVideoAudio: boolean): CremadPlan {
  if (audio && video) return { kind: "multimodal", path: "/predict/cremad/multimodal", audioSource: "file" };
  if (video && useVideoAudio) return { kind: "multimodal", path: "/predict/cremad/multimodal", audioSource: "video_track" };
  if (audio) return { kind: "audio", path: "/predict/cremad/audio" };
  if (video) return { kind: "video", path: "/predict/cremad/video" };
  return { kind: "none" };
}

export const STATUS_LABEL: Record<string, string> = {
  compatible: "Compatible input",
  potential_distribution_shift: "Potential distribution shift",
  low_quality: "Low-quality input",
  unsupported: "Unsupported input",
};
