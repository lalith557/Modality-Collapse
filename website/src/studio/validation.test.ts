import { describe, expect, it } from "vitest";
import { formatBytes, planCremad, preValidate } from "./validation";

describe("formatBytes", () => {
  it("keeps tiny files in bytes so placeholder stubs are visible", () => {
    expect(formatBytes(131)).toBe("131 B");
    expect(formatBytes(119630)).toBe("119.6 KB");
    expect(formatBytes(2_500_000)).toBe("2.50 MB");
  });
});

const f = (name: string, bytes = 10, type = "") => new File([new Uint8Array(bytes)], name, { type });

describe("preValidate", () => {
  it("accepts supported extensions within limits", () => {
    expect(preValidate(f("a.wav"), "audio")).toBeNull();
    expect(preValidate(f("v.webm"), "video")).toBeNull();
    expect(preValidate(f("d.npy"), "depth")).toBeNull();
    expect(preValidate(f("i.csv"), "inertial")).toBeNull();
  });
  it("rejects empty, oversized and unsupported files", () => {
    expect(preValidate(f("a.wav", 0), "audio")).toMatch(/empty/);
    expect(preValidate(f("a.txt"), "audio")).toMatch(/not an accepted audio format/);
    expect(preValidate(f("v.mp4"), "depth")).toMatch(/not an accepted depth format/);
    const big = { name: "a.wav", size: 26 * 1024 * 1024 } as File;
    expect(preValidate(big, "audio")).toMatch(/limit is 25 MB/);
  });
});

describe("planCremad", () => {
  it("only plans multimodal when both modalities are supplied", () => {
    expect(planCremad(null, null, true).kind).toBe("none");
    expect(planCremad(f("a.wav"), null, true)).toMatchObject({ kind: "audio", path: "/predict/cremad/audio" });
    expect(planCremad(null, f("v.mp4"), false)).toMatchObject({ kind: "video", path: "/predict/cremad/video" });
    expect(planCremad(null, f("v.mp4"), true)).toMatchObject({ kind: "multimodal", audioSource: "video_track" });
    expect(planCremad(f("a.wav"), f("v.mp4"), false)).toMatchObject({ kind: "multimodal", audioSource: "file" });
  });
});
