import type { HealthResponse, PredictionResponse, StudioErrorBody, VideoPreviewResponse } from "./types";

/** Backend base URL. Override with VITE_STUDIO_API when the API runs elsewhere. */
export const API_BASE: string =
  (import.meta.env.VITE_STUDIO_API as string | undefined)?.replace(/\/$/, "") ?? "http://127.0.0.1:8000";

export class StudioError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
    public field: string | null = null,
    public requestId: string | null = null,
  ) {
    super(message);
  }
}

async function parse<T>(res: Response): Promise<T> {
  let body: any = null;
  try {
    body = await res.json();
  } catch {
    /* non-JSON body */
  }
  if (!res.ok) {
    const e: StudioErrorBody | undefined = body?.error;
    throw new StudioError(
      res.status,
      e?.code ?? `http_${res.status}`,
      e?.message ?? `The server responded with HTTP ${res.status}.`,
      e?.field ?? null,
      body?.request_id ?? res.headers.get("X-Request-ID"),
    );
  }
  return body as T;
}

function networkError(err: unknown): StudioError {
  if (err instanceof StudioError) return err;
  if (err instanceof DOMException && err.name === "AbortError") {
    return new StudioError(0, "aborted", "The request was cancelled.");
  }
  return new StudioError(
    0,
    "backend_unreachable",
    `Could not reach the Prediction Studio API at ${API_BASE}. Is the backend running?`,
  );
}

export async function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  try {
    return await parse<HealthResponse>(await fetch(`${API_BASE}/health`, { signal }));
  } catch (e) {
    throw networkError(e);
  }
}

export async function getModels(signal?: AbortSignal): Promise<any> {
  try {
    return await parse<any>(await fetch(`${API_BASE}/models`, { signal }));
  } catch (e) {
    throw networkError(e);
  }
}

export type CremadEndpoint = "audio" | "video" | "multimodal";

export async function predict(
  path: string,
  fields: Record<string, File | string | undefined>,
  signal?: AbortSignal,
): Promise<PredictionResponse> {
  const form = new FormData();
  for (const [k, v] of Object.entries(fields)) {
    if (v === undefined) continue;
    if (typeof v === "string") form.append(k, v);
    else form.append(k, v, v.name);
  }
  try {
    return await parse<PredictionResponse>(await fetch(`${API_BASE}${path}`, { method: "POST", body: form, signal }));
  } catch (e) {
    throw networkError(e);
  }
}

/** Server-decoded preview frames for videos the browser cannot play (e.g. CREMA-D's .flv). */
export async function previewVideo(file: File, signal?: AbortSignal): Promise<VideoPreviewResponse> {
  const form = new FormData();
  form.append("video", file, file.name);
  try {
    return await parse<VideoPreviewResponse>(await fetch(`${API_BASE}/preview/video`, { method: "POST", body: form, signal }));
  } catch (e) {
    throw networkError(e);
  }
}
