/**
 * Scientific-integrity tests: the hand-transcribed values in research.ts must agree with
 * the values read from the frozen result artifacts (evidence.verified.json, written by
 * scripts/export_site_evidence.py), and each value must sit under the right label.
 */
import { describe, expect, it } from "vitest";
import ev from "./evidence.verified.json";
import { decomposition, decoupled, findings, fusion, sgdStable } from "./research";

/** The site value must equal the artifact value rounded to the precision the site shows. */
function sameAtShownPrecision(site: number, art: number) {
  const s = String(site);
  const decimals = s.includes("e") ? 12 : (s.split(".")[1] ?? "").length;
  const tol = 0.5 * Math.pow(10, -decimals) + 1e-9;
  expect(Math.abs(site - art), `site ${site} vs artifact ${art}`).toBeLessThanOrEqual(tol);
}

/** p-values: ≥ 0.01 shown to 2 decimals; smaller ones to the significant digits the site gives. */
function sameP(site: number, art: number) {
  if (site >= 0.01) {
    expect(art.toFixed(2), `p ${site} vs ${art}`).toBe(site.toFixed(2));
  } else {
    const sig = site.toExponential().split("e")[0].replace(".", "").replace(/0+$/, "").length || 1;
    expect(Number(art.toPrecision(sig)), `p ${site} vs ${art}`).toBe(site);
  }
}

describe("research.ts agrees with the frozen artifacts", () => {
  const sets = [
    ["adamSynthetic", decomposition.adamSynthetic],
    ["sgdSynthetic", decomposition.sgdSynthetic],
    ["adamCremad", decomposition.adamCremad],
  ] as const;

  it.each(sets)("decomposition %s (α = 100, 5 seeds)", (key, site) => {
    const art = ev.decomposition[key];
    expect(art.seeds).toBe(5);
    expect(art.alpha).toBe(100);
    for (const arm of ["forward", "gradient", "ordinary"] as const) {
      sameAtShownPrecision(site[arm].dDom, art[arm].dDom);
      sameAtShownPrecision(site[arm].d, art[arm].d);
      sameP(site[arm].p, art[arm].p);
    }
  });

  it("gradient-only ratios are shown under the dataset/optimizer they were measured on", () => {
    sameAtShownPrecision(decomposition.ratios.adam.rG, ev.ratios.adamSynthetic.rG);
    expect(Math.abs(decomposition.ratios.adam.rTheta - ev.ratios.adamSynthetic.rTheta)).toBeLessThan(0.01);
    sameAtShownPrecision(decomposition.ratios.sgd.rG, ev.ratios.sgdSynthetic.rG);
    sameAtShownPrecision(decomposition.ratios.sgd.rTheta, ev.ratios.sgdSynthetic.rTheta);
    sameAtShownPrecision(decomposition.ratiosCremad.rG, ev.ratios.adamCremad.rG);
    sameAtShownPrecision(decomposition.ratiosCremad.rTheta, ev.ratios.adamCremad.rTheta);
    sameAtShownPrecision(decomposition.ratiosCremad.baselineRTheta, ev.ratios.adamCremad.baselineRTheta);
    // regression guard: the CREMA-D numbers must never again appear under the synthetic label
    expect(decomposition.ratios.adam.rG).not.toBe(282.7);
    expect(decomposition.ratios.adam.rTheta).not.toBe(1.61);
  });

  it("fusion mechanisms (synthetic, Adam, α = 100, 5 seeds)", () => {
    expect(ev.fusion.optimizer).toBe("adam");
    expect(ev.fusion.alpha).toBe(100);
    expect(ev.fusion.seeds).toBe(5);
    for (const m of fusion.mechanisms) {
      const a = ev.fusion.mechanisms[m.key];
      sameAtShownPrecision(m.forwardD, a.forward.d);
      sameP(m.p, a.forward.p);
      sameAtShownPrecision(m.forwardDDom, a.forward.dDom);
      sameAtShownPrecision(m.gradient.dDom, a.gradient.dDom);
      sameAtShownPrecision(m.gradient.d, a.gradient.d);
      sameP(m.gradient.p, a.gradient.p);
      sameAtShownPrecision(m.ordinaryDDom, a.ordinaryShift);
      sameAtShownPrecision(m.ordinaryVsForward.d, a.ordinaryVsForward.d);
      sameP(m.ordinaryVsForward.p, a.ordinaryVsForward.p);
      sameAtShownPrecision(m.gradientArm.rG, a.gradientArmRG);
      expect(Math.abs(m.gradientArm.rTheta - a.gradientArmRTheta)).toBeLessThan(0.01);
      expect(a.gateValuesLogged).toBe(false);
      expect(m.significant).toBe(a.forward.p < 0.05);
      expect(Math.abs(m.gradient.dDom)).toBeLessThanOrEqual(0.001); // "|ΔD| ≤ 0.001"
    }
  });

  it("the fusion run's sum arm reproduces the Adam decomposition exactly", () => {
    expect(ev.fusion.mechanisms.sum.forward.dDom).toBeCloseTo(ev.decomposition.adamSynthetic.forward.dDom, 12);
    expect(ev.fusion.mechanisms.sum.gradient.dDom).toBeCloseTo(ev.decomposition.adamSynthetic.gradient.dDom, 12);
  });

  it("findings quote the same Student-t intervals as the decoupled table", () => {
    const f1 = findings[0].evidence;
    for (const ci of [
      decoupled.syntheticCIs.masked,
      decoupled.syntheticCIs.probe,
      decoupled.cremadCIs.masked,
      decoupled.cremadCIs.probe,
      decoupled.utdCIs.masked,
    ]) {
      expect(f1).toContain(`±${ci.toFixed(3)}`);
    }
    const f3 = findings[2].evidence;
    expect(f3).toContain(`±${decoupled.cremadSgdClipCIs.masked.toFixed(3)}`);
    expect(f3).toContain(`±${sgdStable.gradientArmMaskedCI.toFixed(3)}`);
    // the retired normal-approximation intervals are gone
    for (const stale of ["±0.010,", "±0.031", "±0.034", "±0.056", "±0.035"]) {
      for (const f of findings) expect(f.evidence).not.toContain(stale);
    }
  });
});
