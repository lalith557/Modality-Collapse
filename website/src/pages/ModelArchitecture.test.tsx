/**
 * Model Architecture page interactions. jsdom has no WebGL, so the page takes its real
 * no-WebGL path (2D view + notice) — the same code a browser without WebGL runs.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, screen, within } from "@testing-library/react";
import { currentLocation, renderAt } from "../test-utils";
import { GRAPHS } from "../architecture/model";

const radio = (group: string, name: string) =>
  within(screen.getByRole("radiogroup", { name: group })).getByRole("radio", { name: new RegExp(`^${name}$`, "i") });
const fieldValue = (k: string) =>
  [...screen.getByTestId("inspector-fields").children].find((d) => d.textContent?.toUpperCase().startsWith(k.toUpperCase()))?.querySelector("dd")?.textContent;

beforeEach(() => vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new TypeError("offline")))));
afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("Model Architecture page", () => {
  it("falls back to the 2D view with a notice when WebGL is unavailable", () => {
    renderAt("/architecture");
    expect(screen.getByRole("status").textContent).toMatch(/WebGL is not available/);
    expect(screen.getByTestId("diagram-2d")).toBeTruthy();
    expect(radio("View", "3D")).toHaveProperty("disabled", true);
    expect(screen.getByTestId("arch-provenance").textContent).toMatch(/32,710 parameters.*strict=True/);
  });

  it("selecting a layer opens its verified details; selecting another updates them", () => {
    renderAt("/architecture");
    fireEvent.click(screen.getByTestId("layer-a_block2"));
    expect(screen.getByTestId("inspector-title").textContent).toBe("Conv block 2");
    expect(fieldValue("Input shape")).toBe("[16, 32, 94]");
    expect(fieldValue("Output shape")).toBe("[32, 16, 47]");
    expect(fieldValue("Kernel size")).toBe("3 × 3");
    expect(fieldValue("Parameters")).toBe("4,640");
    expect(fieldValue("Pooling")).toBe("MaxPool2d(kernel=2, stride=2)");
    // 2D node is pressed and the other branch is dimmed (connected-path highlight)
    const svg = screen.getByTestId("diagram-2d");
    expect(svg.querySelector('[data-node="a_block2"]')!.getAttribute("aria-pressed")).toBe("true");
    expect(svg.querySelector('[data-node="b_block2"]')!.getAttribute("opacity")).toBe("0.25");

    fireEvent.click(screen.getByTestId("layer-fusion"));
    expect(screen.getByTestId("inspector-title").textContent).toBe("Sum fusion");
    expect(fieldValue("Input shape")).toBe("h_A [64] + h_B [64]");
    expect(fieldValue("Parameters")).toBe("0");
    expect(fieldValue("Kernel size")).toBeUndefined(); // not applicable → not shown
  });

  it("2D nodes are keyboard-selectable", () => {
    renderAt("/architecture");
    const node = screen.getByTestId("diagram-2d").querySelector('[data-node="head"]')!;
    fireEvent.keyDown(node, { key: "Enter" });
    expect(screen.getByTestId("inspector-title").textContent).toBe("Linear classifier");
    expect(fieldValue("Parameters")).toBe("390");
  });

  it("switching to UTD-MHAD replaces the whole architecture and resets the selection", () => {
    renderAt("/architecture");
    fireEvent.click(screen.getByTestId("layer-a_logmel"));
    fireEvent.click(radio("Model", "UTD-MHAD"));
    expect(currentLocation.search).toBe("?dataset=utd_mhad");
    expect(screen.getByTestId("inspector-empty")).toBeTruthy();
    const list = screen.getByTestId("layer-list").textContent!;
    expect(list).toMatch(/Inertial signal/);
    expect(list).toMatch(/Depth sequence/);
    expect(list).not.toMatch(/Audio recording|log-mel/);
    expect(screen.getByTestId("arch-provenance").textContent).toMatch(/35,467 parameters/);
    fireEvent.click(screen.getByTestId("layer-b_block1"));
    expect(fieldValue("Layer type")).toBe("Conv1d → ReLU → MaxPool1d");
    expect(fieldValue("Kernel size")).toBe("5");
    fireEvent.click(screen.getByTestId("layer-output"));
    expect(fieldValue("Classes")).toMatch(/a1–a27 \(action names unavailable/);
  });

  it("data flow: next / previous / restart / stage select / play / pause", () => {
    vi.useFakeTimers();
    renderAt("/architecture");
    fireEvent.click(radio("Mode", "Data flow"));
    const steps = GRAPHS.cremad.forwardOrder.length;
    expect(screen.getByTestId("flow-panel").textContent).toMatch(new RegExp(`step 1 of ${steps}`));
    expect(screen.getByTestId("flow-stage").textContent).toBe("Audio recording");
    fireEvent.click(screen.getByLabelText("Next stage"));
    fireEvent.click(screen.getByLabelText("Next stage"));
    expect(screen.getByTestId("flow-stage").textContent).toBe("64-bin log-mel");
    expect(screen.getByTestId("flow-shape").textContent).toBe("[1, 48000] → [1, 64, 188]");
    fireEvent.click(screen.getByLabelText("Previous stage"));
    expect(screen.getByTestId("flow-stage").textContent).toBe("Mono 16 kHz, 3.0 s");
    fireEvent.change(screen.getByLabelText("Select a processing stage"), { target: { value: String(steps - 3) } });
    expect(screen.getByTestId("flow-stage").textContent).toBe("Sum fusion");
    expect(screen.getByTestId("flow-shape").textContent).toBe("[64] + [64] → [64]");
    expect(screen.getByTestId("flow-next").textContent).toBe("Linear classifier");
    // the active stage is highlighted and its incoming edge animated in the view
    const svg = screen.getByTestId("diagram-2d");
    expect(svg.querySelector('[data-node="fusion"]')!.getAttribute("aria-pressed")).toBe("true");
    fireEvent.click(screen.getByLabelText("Play data flow"));
    act(() => vi.advanceTimersByTime(2700));
    expect(screen.getByTestId("flow-stage").textContent).toBe("Linear classifier");
    fireEvent.click(screen.getByLabelText("Pause"));
    act(() => vi.advanceTimersByTime(6000));
    expect(screen.getByTestId("flow-stage").textContent).toBe("Linear classifier");
    fireEvent.click(screen.getByLabelText("Restart"));
    expect(screen.getByTestId("flow-stage").textContent).toBe("Audio recording");
    expect(screen.getByTestId("context-bar").textContent).toMatch(/data flow/);
  });

  it("the conceptual overlay is labelled as conceptual and marks the scaled branch", () => {
    renderAt("/architecture?dataset=utd_mhad");
    expect(screen.queryByTestId("overlay-2d")).toBeNull();
    fireEvent.click(screen.getByTestId("overlay-toggle"));
    expect(screen.getByTestId("overlay-2d")).toBeTruthy();
    const ex = screen.getByTestId("overlay-explainer").textContent!;
    expect(ex).toMatch(/Actual model architecture/);
    expect(ex).toMatch(/Experiment-specific intervention/);
    expect(ex).toMatch(/Conceptual research mechanism/);
    expect(ex).toMatch(/h_B \(inertial\)/);
    expect(ex).toMatch(/not the model's computational graph/);
    expect(screen.getByTestId("context-bar").textContent).toMatch(/conceptual \(on\)/);
  });

  it("comparison mode shows both models side by side", () => {
    renderAt("/architecture");
    fireEvent.click(radio("Mode", "Compare"));
    const t = screen.getByTestId("compare-table").textContent!;
    expect(t).toMatch(/32,710/);
    expect(t).toMatch(/35,467/);
    expect(t).toMatch(/InertialCNN/);
    expect(screen.queryByTestId("diagram-2d")).toBeNull();
  });

  it("links to the Prediction Studio with the selected dataset, and back", () => {
    renderAt("/architecture?dataset=utd_mhad");
    expect(screen.getByTestId("try-in-studio").getAttribute("href")).toBe("/studio?dataset=utd_mhad");
    cleanup();
    renderAt("/studio?dataset=utd_mhad");
    expect(within(screen.getByRole("radiogroup", { name: "Dataset" })).getByRole("radio", { checked: true }).textContent).toMatch(/UTD-MHAD/);
    expect(screen.getByTestId("inspect-architecture").getAttribute("href")).toBe("/architecture?dataset=utd_mhad");
  });

  it("never claims to have run the model", () => {
    renderAt("/architecture");
    fireEvent.click(screen.getByTestId("layer-output"));
    const page = document.body.textContent!;
    expect(page).toMatch(/nothing is executed on this page/);
    expect(page).not.toMatch(/predicted class:|confidence:/i);
  });
});

describe("figure components in the layer list", () => {
  it("can be selected from the keyboard list and open the component inspector", () => {
    renderAt("/architecture?dataset=utd_mhad");
    fireEvent.click(screen.getByTestId("layer-concept-intervention"));
    expect(screen.getByTestId("inspector-title").textContent).toBe("Feature-scale intervention (α)");
    expect(screen.getByTestId("inspector").textContent).toMatch(/Not a layer of the baseline checkpoint/);
    expect(fieldValue("Scaled branch")).toBe("inertial (h_B)");
    fireEvent.click(screen.getByTestId("layer-concept-fusion_gated"));
    expect(screen.getByTestId("inspector-title").textContent).toBe("Gated fusion");
    expect(screen.getByTestId("inspector").textContent).toMatch(/synthetic experiment only/);
    // selecting a figure component never dims the model path
    expect(screen.getByTestId("diagram-2d").querySelector('[data-node="a_block1"]')!.getAttribute("opacity")).toBe("1");
  });
});

describe("workflow navigation", () => {
  it("the stage navigator opens each stage and marks it as the current step", () => {
    renderAt("/architecture");
    fireEvent.click(screen.getByTestId("stage-fusion"));
    expect(screen.getByTestId("inspector-title").textContent).toBe("Sum fusion");
    expect(screen.getByTestId("stage-fusion").getAttribute("aria-current")).toBe("step");
    expect(screen.getByTestId("stage-inputs").getAttribute("aria-current")).toBeNull();
    fireEvent.click(screen.getByTestId("stage-intervention"));
    expect(screen.getByTestId("inspector-title").textContent).toBe("Feature-scale intervention (α)");
    // in the data-flow tour a stage jumps to its step
    fireEvent.click(radio("Mode", "Data flow"));
    fireEvent.click(screen.getByTestId("stage-classifier"));
    expect(screen.getByTestId("flow-stage").textContent).toBe("Linear classifier");
  });

  it("the research strip shows the variants table and opens each component", () => {
    renderAt("/architecture?dataset=utd_mhad");
    const strip = screen.getByTestId("research-strip");
    expect(strip.textContent).toMatch(/Forward-only/);
    expect(strip.textContent).toMatch(/Gradient-only/);
    expect(strip.textContent).toMatch(/Ordinary \(both\)/);
    // UTD-MHAD scaled the inertial branch, so the table is written for h_B
    expect(within(screen.getByTestId("research-variants")).getAllByText("B").length).toBeGreaterThan(0);
    expect(strip.textContent).toMatch(/r_g 100\.9× → r_Δθ 1\.00×/);
    expect(strip.textContent).toMatch(/r_g 318\.5× → r_Δθ 95\.9×/);
    fireEvent.click(screen.getByTestId("research-optimizer"));
    expect(screen.getByTestId("inspector-title").textContent).toBe("Optimizer analysis");
    fireEvent.click(screen.getByTestId("research-variants"));
    expect(screen.getByTestId("inspector-title").textContent).toBe("Intervention variants");
  });

  it("the 2D view draws the intervention markers and the synthetic-only fusions as selectable parts", () => {
    renderAt("/architecture");
    const svg = screen.getByTestId("diagram-2d");
    const marker = svg.querySelector('[data-concept="concept:intervention"]')!;
    expect(marker.getAttribute("aria-label")).toMatch(/α on audio/);
    fireEvent.keyDown(marker, { key: "Enter" });
    expect(screen.getByTestId("inspector-title").textContent).toBe("Feature-scale intervention (α)");
    fireEvent.click(svg.querySelector('[data-concept="concept:fusion_concat"]')!);
    expect(screen.getByTestId("inspector-title").textContent).toBe("Concatenation fusion");
    // class tiles show the checkpoint's class codes and no scores
    expect(svg.textContent).toMatch(/ANG/);
    expect(svg.textContent).not.toMatch(/\d\.\d{3}/);
  });
});

describe("hover explanations", () => {
  it("hovering a part pops up what it does, from the generated graph", () => {
    renderAt("/architecture");
    const node = screen.getByTestId("diagram-2d").querySelector('[data-node="a_logmel"]')!;
    fireEvent.mouseEnter(node, { clientX: 200, clientY: 200 });
    const card = screen.getByTestId("hover-explain");
    const logmel = GRAPHS.cremad.nodes.find((n) => n.id === "a_logmel")!;
    expect(card.textContent).toContain(logmel.simple);
    expect(card.textContent).toMatch(/\[1, 48000\] → \[1, 64, 188\]/);
    expect(card.textContent).toMatch(/no learned parameters/);
    fireEvent.mouseLeave(node);
    expect(screen.queryByTestId("hover-explain")).toBeNull();
    const fc = screen.getByTestId("diagram-2d").querySelector('[data-node="b_fc"]')!;
    fireEvent.mouseEnter(fc, { clientX: 200, clientY: 200 });
    expect(screen.getByTestId("hover-explain").textContent).toMatch(/Encoder ℰ_B · video/);
    expect(screen.getByTestId("hover-explain").textContent).toMatch(/2,112 learned parameters/);
  });
});
