/**
 * Video preview fallback. The server is mocked with REAL /preview/video responses captured
 * from the running API (src/studio/__fixtures__/preview_*.json).
 */
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { VideoPreview } from "./WaveformPreview";
import previewFlv from "../__fixtures__/preview_flv.json";
import previewCorrupt from "../__fixtures__/preview_error_corrupt.json";

const json = (body: unknown, status = 200) =>
  Promise.resolve(new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }));
const file = (name: string, type = "") => new File([new Uint8Array(64)], name, { type });

let fetchMock: ReturnType<typeof vi.fn>;
let canPlay: MockInstance<HTMLMediaElement["canPlayType"]>;
beforeEach(() => {
  fetchMock = vi.fn();
  vi.stubGlobal("fetch", fetchMock);
  // jsdom answers "" for every type; tests set what the "browser" claims it can play
  canPlay = vi.spyOn(HTMLMediaElement.prototype, "canPlayType").mockReturnValue("maybe");
});
afterEach(() => {
  cleanup();
  canPlay.mockRestore();
  vi.unstubAllGlobals();
});

const previewCalls = () => fetchMock.mock.calls.filter(([url]) => String(url).endsWith("/preview/video"));

describe("VideoPreview", () => {
  it("plays a browser-playable file natively without contacting the server", () => {
    render(<VideoPreview file={file("clip.mp4", "video/mp4")} />);
    expect(screen.getByTestId("native-video")).toBeTruthy();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("sends .flv straight to the server and shows the frames it decoded", async () => {
    let release!: (r: Response) => void;
    fetchMock.mockImplementation(() => new Promise<Response>((res) => (release = res)));
    render(<VideoPreview file={file("1001_DFA_ANG_XX.flv")} />);
    // genuine loading state while the request is in flight; no <video> element is tried
    expect(screen.getByTestId("preview-loading").textContent).toMatch(/cannot play \.flv files/);
    expect(screen.queryByTestId("native-video")).toBeNull();
    expect(previewCalls()).toHaveLength(1);
    const [, init] = previewCalls()[0];
    expect(init.method).toBe("POST");
    expect((init.body as FormData).get("video")).toBeInstanceOf(File);

    release(new Response(JSON.stringify(previewFlv), { status: 200 }));
    const panel = await screen.findByTestId("server-preview");
    // every frame and every number comes from the server response
    const imgs = panel.querySelectorAll("img");
    expect(imgs).toHaveLength(previewFlv.frames.length);
    previewFlv.frames.forEach((f, k) => {
      expect(imgs[k].getAttribute("src")).toBe(f.image);
      expect(imgs[k].getAttribute("alt")).toBe(`Frame ${f.index}`);
    });
    expect(panel.textContent).toMatch(`#${previewFlv.frames[1].index} · ${previewFlv.frames[1].time_s!.toFixed(2)} s`);
    expect(screen.getByTestId("server-preview-meta").textContent).toBe(
      `${previewFlv.width}×${previewFlv.height} · ${previewFlv.fps.toFixed(2)} fps · ${previewFlv.n_frames} frames · ` +
        `${previewFlv.duration_s.toFixed(2)} s · no audio track`,
    );
    // the centre-crop overlay uses the server's crop box (640×480 → x 12.5 %, w 75 %)
    const crop = panel.querySelector<HTMLElement>(".border-dashed")!;
    expect(crop.style.left).toBe(`${previewFlv.crop.x * 100}%`);
    expect(crop.style.width).toBe(`${previewFlv.crop.w * 100}%`);
  });

  it("treats .avi like .flv", () => {
    fetchMock.mockImplementation(() => new Promise<Response>(() => {}));
    render(<VideoPreview file={file("trial.AVI", "video/x-msvideo")} />);
    expect(screen.getByTestId("preview-loading")).toBeTruthy();
    expect(previewCalls()).toHaveLength(1);
  });

  it("falls back to the server when the browser reports it cannot play the MIME type", () => {
    canPlay.mockReturnValue("");
    fetchMock.mockImplementation(() => new Promise<Response>(() => {}));
    render(<VideoPreview file={file("clip.mkv", "video/x-matroska")} />);
    expect(screen.getByTestId("preview-loading").textContent).toMatch(/cannot play \.mkv files/);
  });

  it("falls back when native playback fails to decode", () => {
    fetchMock.mockImplementation(() => new Promise<Response>(() => {}));
    render(<VideoPreview file={file("clip.mp4", "video/mp4")} />);
    fireEvent.error(screen.getByTestId("native-video"));
    expect(screen.getByTestId("preview-loading").textContent).toMatch(/could not decode this video/);
    expect(previewCalls()).toHaveLength(1);
  });

  it("falls back when the video track has no picture (no decoder for the codec)", () => {
    fetchMock.mockImplementation(() => new Promise<Response>(() => {}));
    render(<VideoPreview file={file("clip.mp4", "video/mp4")} />);
    const v = screen.getByTestId("native-video") as HTMLVideoElement;
    Object.defineProperty(v, "videoWidth", { value: 0, configurable: true });
    fireEvent.loadedMetadata(v);
    expect(screen.getByTestId("preview-loading").textContent).toMatch(/no decoder for this video codec/);
  });

  it("shows the server's error code and message when the preview also fails", async () => {
    fetchMock.mockImplementation(() => json(previewCorrupt, 422));
    render(<VideoPreview file={file("broken.flv")} />);
    const failed = await screen.findByTestId("preview-failed");
    expect(failed.textContent).toMatch(previewCorrupt.error.code);
    expect(failed.textContent).toMatch(previewCorrupt.error.message);
    expect(screen.queryByTestId("server-preview")).toBeNull();
  });

  it("reports an unreachable backend instead of a blank preview", async () => {
    fetchMock.mockRejectedValue(new TypeError("Failed to fetch"));
    render(<VideoPreview file={file("clip.flv")} />);
    expect((await screen.findByTestId("preview-failed")).textContent).toMatch(/backend_unreachable/);
  });

  it("aborts the in-flight preview request when the file is removed", () => {
    fetchMock.mockImplementation(() => new Promise<Response>(() => {}));
    const { unmount } = render(<VideoPreview file={file("clip.flv")} />);
    const signal = previewCalls()[0][1].signal as AbortSignal;
    expect(signal.aborted).toBe(false);
    unmount();
    expect(signal.aborted).toBe(true);
  });
});
