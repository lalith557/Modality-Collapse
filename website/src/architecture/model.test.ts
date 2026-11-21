/** The architecture graph shown on the page is internally consistent and matches the generated data. */
import { describe, expect, it } from "vitest";
import generated from "../data/architecture.generated.json";
import { GRAPHS, compareRows, connectedPath, flowSteps, nodeById } from "./model";

const ds = ["cremad", "utd_mhad"] as const;

describe("architecture graph", () => {
  it.each(ds)("%s: parameter counts sum to the checkpoint total", (d) => {
    const g = GRAPHS[d];
    const sum = (br: string) => g.nodes.filter((n) => n.branch === br).reduce((s, n) => s + n.params, 0);
    expect(sum("a")).toBe(g.params.encoder_a);
    expect(sum("b")).toBe(g.params.encoder_b);
    expect(nodeById(g, "head")!.params).toBe(g.params.head);
    expect(g.params.encoder_a + g.params.encoder_b + g.params.head).toBe(g.params.total);
    expect(g.params.total).toBe(d === "cremad" ? 32710 : 35467);
    expect(g.checkpoint.strict_state_dict_load).toBe(true);
  });

  it.each(ds)("%s: every connection carries the shape the next stage expects", (d) => {
    const g = GRAPHS[d];
    for (const e of g.edges) {
      const a = nodeById(g, e.from)!;
      const b = nodeById(g, e.to)!;
      if (!a.out_shape || !b.in_shape || b.id === "fusion") continue;
      if (b.id.endsWith("_tmean")) {
        expect(b.in_shape).toEqual([a.per_frame, ...a.out_shape]); // per-frame vectors stacked over frames
      } else if (b.per_frame && !a.per_frame) {
        expect([b.per_frame, ...b.in_shape], `${e.from} → ${e.to}`).toEqual(a.out_shape); // frames enter the per-frame CNN
      } else {
        expect(b.in_shape, `${e.from} → ${e.to}`).toEqual(a.out_shape);
      }
    }
    const fusion = nodeById(g, "fusion")!;
    expect(fusion.in_shapes!.a).toEqual(nodeById(g, "a_fc")!.out_shape);
    expect(fusion.in_shapes!.b).toEqual(nodeById(g, "b_fc")!.out_shape);
    expect(nodeById(g, "output")!.out_shape).toEqual([g.nClasses]);
  });

  it("each dataset shows only its own modalities", () => {
    const text = (d: (typeof ds)[number]) => JSON.stringify(GRAPHS[d].nodes.map((n) => [n.label, n.simple, n.technical]));
    expect(GRAPHS.cremad.modalities).toEqual({ a: "audio", b: "video" });
    expect(GRAPHS.utd_mhad.modalities).toEqual({ a: "depth", b: "inertial" });
    expect(text("cremad")).not.toMatch(/inertial|depth|Kinect/i);
    expect(text("utd_mhad")).not.toMatch(/log-mel|spectrogram|grayscale/i);
  });

  it("node data is copied from the generated JSON, not retyped", () => {
    for (const d of ds) {
      const gen = (generated.datasets as any)[d].nodes as any[];
      expect(GRAPHS[d].nodes.map((n) => n.id)).toEqual(gen.map((n) => n.id));
      GRAPHS[d].nodes.forEach((n, i) => {
        expect(n.in_shape).toEqual(gen[i].in_shape);
        expect(n.out_shape).toEqual(gen[i].out_shape);
        expect(n.params).toBe(gen[i].params);
        expect(n.simple.length).toBeGreaterThan(20);
        expect(`${n.simple} ${n.technical}`).not.toMatch(/undefined|NaN|\[object/);
      });
    }
  });

  it("selecting a layer highlights exactly its connected data-flow path", () => {
    const g = GRAPHS.cremad;
    const p = connectedPath(g, "a_block2");
    expect(p.nodes.has("a_raw")).toBe(true);
    expect(p.nodes.has("output")).toBe(true);
    expect(p.nodes.has("b_block2")).toBe(false); // the other branch is not on this path
    expect(p.edges.has("a_fc->fusion")).toBe(true);
  });

  it("data-flow steps follow the forward() call order and end at the prediction", () => {
    for (const d of ds) {
      const steps = flowSteps(GRAPHS[d]);
      expect(steps).toHaveLength(GRAPHS[d].nodes.length);
      expect(steps.map((s) => s.node.id)).toEqual((generated.datasets as any)[d].forward_order);
      expect(steps[steps.length - 1].node.id).toBe("output");
      expect(steps[steps.length - 1].next).toHaveLength(0);
    }
  });

  it("comparison rows come from the two graphs", () => {
    const rows = Object.fromEntries(compareRows().map((r) => [r.field, r]));
    expect(rows["Total parameters"]).toMatchObject({ cremad: "32,710", utd_mhad: "35,467" });
    expect(rows["Model input B"]).toMatchObject({ cremad: "[8, 1, 64, 64]", utd_mhad: "[6, 128]" });
    expect(rows["Fusion"].cremad).toBe(rows["Fusion"].utd_mhad);
  });

  it("the overlay marks the branch each real-data experiment scaled", () => {
    expect(GRAPHS.cremad.scaledBranch).toBe("a"); // audio
    expect(GRAPHS.utd_mhad.scaledBranch).toBe("b"); // inertial
  });
});

import media from "../data/architecture.media.json";
import { decomposition, fusion } from "../data/research";
import { pValue } from "../lib/format";
import { conceptNodes } from "./model";

describe("figure components (not checkpoint layers)", () => {
  const ev = { fusion: fusion.mechanisms as never, ratios: decomposition.ratios, pValue };

  it("intervention marks the branch each dataset's experiment scaled, with α = 1 in the checkpoint", () => {
    const c = conceptNodes(GRAPHS.cremad, ev)["concept:intervention"];
    expect(c.facts.find((f) => f.k === "Scaled branch")!.v).toBe("audio (h_A)");
    expect(c.facts.find((f) => f.k === "α in this checkpoint")!.v).toMatch(/^1 /);
    const u = conceptNodes(GRAPHS.utd_mhad, ev)["concept:intervention"];
    expect(u.facts.find((f) => f.k === "Scaled branch")!.v).toBe("inertial (h_B)");
  });

  it("concat / gated are labelled synthetic-only and quote research.ts values and the implemented formula", () => {
    const c = conceptNodes(GRAPHS.cremad, ev);
    for (const key of ["concat", "gated"] as const) {
      const n = c[`concept:fusion_${key}`];
      const m = fusion.mechanisms.find((x) => x.key === key)!;
      expect(n.tag).toBe("synthetic experiment only");
      expect(n.technical).toContain(m.formula);
      expect(n.technical).toMatch(/checkpoint uses sum fusion/);
      expect(n.facts.find((f) => f.k === "Forward d_z · p")!.v).toBe(`${m.forwardD.toFixed(2)} · ${pValue(m.p)}`);
    }
    expect(fusion.mechanisms.find((x) => x.key === "gated")!.formula).toBe("z = σ(W_A h_A + b_A) ⊙ s(h_A) + σ(W_B h_B + b_B) ⊙ h_B");
    expect(c["concept:fusion_gated"].facts.find((f) => f.k === "Gate values")!.v).toMatch(/not logged/);
  });

  it("optimizer panel uses the paper's multipliers and the labelled synthetic ratios", () => {
    const o = conceptNodes(GRAPHS.cremad, ev)["concept:optimizer"];
    expect(o.technical).toMatch(/m = s_f·s_g under SGD and m = s_f under small-ε Adam/);
    expect(o.facts[0].v).toContain(`${decomposition.ratios.adam.rG.toFixed(1)}×`);
    expect(o.facts[1].v).toContain(`${decomposition.ratios.sgd.rTheta.toFixed(1)}×`);
  });

  it("the classifier is a single linear layer in both checkpoints (no hidden FC→ReLU)", () => {
    for (const d of ds) {
      const h = nodeById(GRAPHS[d], "head")!;
      expect(h.op).toBe("Linear");
      expect(h.params).toBe(h.in_shape![0] * h.out_shape![0] + h.out_shape![0]);
    }
  });

  it("input thumbnails have the real model-input shapes", () => {
    expect(media.logmel_shape).toEqual(nodeById(GRAPHS.cremad, "a_logmel")!.out_shape!.slice(1));
    expect(media.frames_shape).toEqual([8, 64, 64]);
    expect(media.depth_shape).toEqual([16, 32, 32]);
    expect(media.inertial_z).toHaveLength(6);
    expect((media.inertial_z as number[][])[0]).toHaveLength(nodeById(GRAPHS.utd_mhad, "b_zscore")!.out_shape![1]);
  });
});

describe("workflow stages", () => {
  it("every node maps to a stage, and every stage opens on a node of that stage", async () => {
    const { STAGE_ORDER, stageEntry, stageOf } = await import("./model");
    for (const g of Object.values(GRAPHS)) {
      for (const n of g.nodes) expect(stageOf(g, n.id)).not.toBeNull();
      for (const s of STAGE_ORDER) expect(stageOf(g, stageEntry(g, s))).toBe(s);
    }
    expect(stageOf(GRAPHS.cremad, "a_raw")).toBe("inputs");
    expect(stageOf(GRAPHS.cremad, "a_logmel")).toBe("preprocessing");
    expect(stageOf(GRAPHS.utd_mhad, "b_block2")).toBe("encoding");
    expect(stageOf(GRAPHS.cremad, "concept:fusion_gated")).toBe("fusion");
  });
});

describe("sections", () => {
  it("every node and figure component belongs to one raisable section", async () => {
    const { sectionOf } = await import("./model");
    const C = GRAPHS.cremad;
    const U = GRAPHS.utd_mhad;
    for (const g of [C, U]) for (const n of g.nodes) expect(sectionOf(g, n.id)).not.toBeNull();
    expect(sectionOf(C, "a_block2")).toBe("enc_a");
    expect(sectionOf(C, "b_tmean")).toBe("enc_b");
    expect(sectionOf(C, "a_wave")).toBe("preprocessing");
    expect(sectionOf(C, "head")).toBe("classifier");
    expect(sectionOf(C, "concept:variants")).toBe("intervention");
    expect(sectionOf(C, "concept:fusion_gated")).toBe("fusion");
    expect(sectionOf(C, "section:output")).toBe("output");
    // the optimizer marker sits over the branch each dataset scaled
    expect(sectionOf(C, "concept:optimizer")).toBe("enc_a");
    expect(sectionOf(U, "concept:optimizer")).toBe("enc_b");
  });
});
