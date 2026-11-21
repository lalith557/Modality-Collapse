/**
 * Studio UI states. The backend is mocked with REAL responses captured from the running
 * API (src/studio/__fixtures__/*.json), so the UI is tested against the true contract.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { PredictionStudio } from "../pages/PredictionStudio";
import { StudioProvider } from "./StudioContext";
import health from "./__fixtures__/health.json";
import external from "./__fixtures__/external_multimodal.json";
import unsupported from "./__fixtures__/unsupported_tone.json";
import missingAudio from "./__fixtures__/error_missing_audio.json";

const json = (body: unknown, status = 200) =>
  Promise.resolve(new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }));

function mount() {
  return render(
    <MemoryRouter>
      <StudioProvider>
        <PredictionStudio />
      </StudioProvider>
    </MemoryRouter>,
  );
}

const runButton = () => screen.getByRole("button", { name: /run prediction/i }) as HTMLButtonElement;
const attach = (id: string, name: string, type = "") =>
  fireEvent.change(document.getElementById(id)!, { target: { files: [new File([new Uint8Array(64)], name, { type })] } });

let fetchMock: ReturnType<typeof vi.fn>;
beforeEach(() => {
  fetchMock = vi.fn();
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("Prediction Studio", () => {
  it("shows an offline state and disables Run when the backend is unreachable", async () => {
    fetchMock.mockRejectedValue(new TypeError("Failed to fetch"));
    mount();
    expect(await screen.findByText(/backend offline/i)).toBeTruthy();
    expect(runButton().disabled).toBe(true);
  });

  it("starts empty with Run disabled until a valid input is added", async () => {
    fetchMock.mockImplementation(() => json(health));
    mount();
    expect(await screen.findByText(/backend online/i)).toBeTruthy();
    expect(screen.getByTestId("empty-state")).toBeTruthy();
    expect(runButton().disabled).toBe(true);
    attach("upload-audio", "notes.txt");
    expect(screen.getByRole("alert").textContent).toMatch(/not an accepted audio format/);
    expect(runButton().disabled).toBe(true);
    attach("upload-audio", "speech.wav", "audio/wav");
    await waitFor(() => expect(runButton().disabled).toBe(false));
  });

  it("shows a genuine loading state, then renders the server's actual prediction", async () => {
    let release!: (r: Response) => void;
    fetchMock.mockImplementation((url: string) =>
      url.endsWith("/health") ? json(health) : new Promise<Response>((res) => (release = res)),
    );
    mount();
    await screen.findByText(/backend online/i);
    attach("upload-video", "face_with_speech.mp4", "video/mp4");
    await waitFor(() => expect(runButton().disabled).toBe(false));
    fireEvent.click(runButton());
    expect(await screen.findByTestId("loading-state")).toBeTruthy();
    release(new Response(JSON.stringify(external), { status: 200 }));
    expect(await screen.findByTestId("result-panel")).toBeTruthy();
    // values come from the captured backend response, not from the UI
    expect(screen.getByTestId("predicted-code").textContent).toBe(external.primary!.predicted_code);
    expect(screen.getAllByText(external.primary!.max_probability.toFixed(3)).length).toBeGreaterThan(0);
    expect(screen.getByText(/potential distribution shift/i, { selector: "span" })).toBeTruthy();
    expect(screen.getByTestId("architecture-trace").querySelectorAll("tbody tr").length).toBe(external.trace.layers.length);
    const [, init] = fetchMock.mock.calls.find(([u]) => String(u).includes("/predict/"))!;
    expect(String(fetchMock.mock.calls.find(([u]) => String(u).includes("/predict/"))![0])).toMatch(/\/predict\/cremad\/multimodal$/);
    expect((init as RequestInit).method).toBe("POST");
  });

  it("renders an unsupported result without any prediction", async () => {
    fetchMock.mockImplementation((url: string) => (url.endsWith("/health") ? json(health) : json(unsupported)));
    mount();
    await screen.findByText(/backend online/i);
    attach("upload-audio", "tone.wav", "audio/wav");
    await waitFor(() => expect(runButton().disabled).toBe(false));
    fireEvent.click(runButton());
    expect(await screen.findByText(/no prediction was made/i)).toBeTruthy();
    expect(screen.queryByTestId("predicted-code")).toBeNull();
    expect(screen.getByText(/no forward pass was executed/i)).toBeTruthy();
  });

  it("renders the server's error code and message", async () => {
    fetchMock.mockImplementation((url: string) => (url.endsWith("/health") ? json(health) : json(missingAudio, 422)));
    mount();
    await screen.findByText(/backend online/i);
    attach("upload-video", "silent_video.webm", "video/webm");
    await waitFor(() => expect(runButton().disabled).toBe(false));
    fireEvent.click(runButton());
    const err = await screen.findByTestId("error-state");
    expect(err.textContent).toMatch(/HTTP 422/);
    expect(err.textContent).toMatch(/missing_audio_stream/);
    expect(err.textContent).toMatch(missingAudio.error.message);
  });

  it("requires both files for UTD-MHAD", async () => {
    fetchMock.mockImplementation(() => json(health));
    mount();
    await screen.findByText(/backend online/i);
    fireEvent.click(screen.getByRole("radio", { name: /UTD-MHAD/i }));
    attach("upload-depth", "clip.npy");
    expect(runButton().disabled).toBe(true);
    expect(screen.getByTestId("blocker").textContent).toMatch(/both a depth file and an inertial file/);
    attach("upload-inertial", "imu.csv", "text/csv");
    await waitFor(() => expect(runButton().disabled).toBe(false));
  });
});
