/** Navigation, retired-page redirects, and the merged Disentanglement / Overview pages. */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, screen, within } from "@testing-library/react";
import { nav } from "../data/nav";
import { REDIRECTS } from "../App";
import { decomposition, fusion, findings } from "../data/research";
import { pValue } from "../lib/format";
import { currentLocation, renderAt } from "../test-utils";

beforeEach(() => vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new TypeError("offline")))));
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("primary navigation", () => {
  it("has exactly the nine intended destinations, in order", () => {
    expect(nav.map((n) => n.label)).toEqual([
      "Home",
      "Research Overview",
      "Experiment Lab",
      "Prediction Studio",
      "Model Architecture",
      "Disentanglement",
      "Real-World Evidence",
      "Methodology",
      "Paper",
    ]);
    expect(nav.map((n) => n.path)).not.toEqual(expect.arrayContaining(["/findings", "/optimizer", "/fusion"]));
  });

  it("the navbar renders those items and no retired ones", () => {
    renderAt("/");
    const links = [...document.querySelectorAll("header nav a")].map((a) => a.textContent!.trim());
    expect(links).toEqual(nav.map((n) => n.label));
    expect(document.body.textContent).not.toMatch(/Optimizer Analysis|Fusion & Utility/);
  });

  it.each(nav.map((n) => [n.label, n.path]))("%s (%s) renders", (_label, path) => {
    renderAt(path);
    expect(document.querySelector("h1")?.textContent?.length).toBeGreaterThan(3);
    expect(currentLocation.pathname).toBe(path);
  });
});

describe("retired URLs redirect to the section that now holds their content", () => {
  it.each(Object.entries(REDIRECTS))("%s → %s", (from, to) => {
    renderAt(from);
    expect(currentLocation.pathname + currentLocation.search).toBe(to);
    const section = new URLSearchParams(to.split("?")[1]).get("section")!;
    expect(document.getElementById(section)).toBeTruthy();
  });

  it("unknown paths go home", () => {
    renderAt("/no-such-page");
    expect(currentLocation.pathname).toBe("/");
  });
});

describe("Disentanglement (with the former Optimizer and Fusion pages)", () => {
  const armP = (tone: string) => screen.getByTestId(`arm-${tone}-p`).textContent;

  it("Adam/SGD switch updates every optimizer-dependent element together", () => {
    renderAt("/disentanglement");
    const a = decomposition.adamSynthetic;
    expect(armP("gradient")).toBe(`p ${pValue(a.gradient.p)}`);
    expect(screen.getByTestId("regime").textContent).toBe("Forward-dominant");
    expect(screen.getByTestId("flow-chain").textContent).toMatch(/≈ 100\.9×.*≈ 1\.00×/);
    expect(screen.getByTestId("context-bar").textContent).toMatch(/Pathway sectionsAdam/);

    const toggles = screen.getAllByRole("radiogroup", { name: "Select optimizer" });
    fireEvent.click(within(toggles[0]).getByRole("radio", { name: "SGD" }));
    const s = decomposition.sgdSynthetic;
    expect(armP("forward")).toBe(`p ${pValue(s.forward.p)}`);
    expect(armP("gradient")).toBe(`p ${pValue(s.gradient.p)}`);
    expect(screen.getByTestId("regime").textContent).toBe("Both pathways contribute");
    expect(screen.getByTestId("flow-chain").textContent).toMatch(/≈ 318\.5×.*≈ 95\.9×/);
    const selectedCells = [...screen.getByTestId("optimizer-comparison").querySelectorAll('td[data-col="sgd"]')];
    expect(selectedCells.every((c) => c.className.includes("font-semibold"))).toBe(true);
    expect(screen.getByTestId("meaning-card").textContent).toMatch(/SGD/);
    // both toggles are bound to the same state
    expect(within(toggles[1]).getByRole("radio", { name: "SGD" }).getAttribute("aria-checked")).toBe("true");
    expect(screen.getByTestId("context-bar").textContent).toMatch(/Pathway sectionsSGD/);
  });

  it("never shows the CREMA-D ratios under the synthetic label", () => {
    renderAt("/disentanglement");
    expect(screen.getByTestId("flow-chain").textContent).not.toMatch(/282\.7|1\.61/);
    expect(screen.getByTestId("cremad-ratio-note").textContent).toMatch(/CREMA-D.*282\.7×.*1\.61×.*1\.65×/);
  });

  it("fusion selector updates diagram formula, metrics and explanation together", () => {
    renderAt("/disentanglement");
    for (const m of fusion.mechanisms) {
      fireEvent.click(screen.getByTestId(`fusion-${m.key}`));
      expect(screen.getByTestId("fusion-formula").textContent).toBe(m.formula);
      expect(screen.getByTestId("fusion-forward-p").textContent).toBe(`p = ${pValue(m.p)}`);
      expect(screen.getByTestId("fusion-detail").textContent).toMatch(m.key === "gated" ? /not individually significant/ : /gradient-only arm stays null/);
      const active = screen.getByTestId("gradient-only-table").querySelector("tr.font-semibold") as HTMLElement;
      expect(active.dataset.fusion).toBe(m.key);
      expect(screen.getByTestId("context-bar").textContent).toMatch(new RegExp(`Fusion sectionAdam · ${m.name}`));
    }
    expect(screen.getByTestId("fusion-detail").textContent).toMatch(/gate activations were not logged/);
  });

  it("has no decorative controls that invent values", () => {
    renderAt("/disentanglement");
    expect(document.querySelector('input[type="range"]')).toBeNull();
  });
});

describe("Research Overview holds the findings", () => {
  it("shows all four findings with metrics, evidence, limitation and links", () => {
    renderAt("/overview?section=findings");
    expect(document.getElementById("findings")).toBeTruthy();
    for (const f of findings) {
      const card = screen.getByTestId(`finding-${f.n}`);
      expect(card.textContent).toContain(f.title);
      expect(card.textContent).toContain(f.interpretation);
      expect(within(card).getByTestId(`finding-${f.n}-metrics`).children.length).toBe(3);
      fireEvent.click(within(card).getByRole("button", { name: /Evidence and scope/ }));
      const ev = within(card).getByTestId(`finding-${f.n}-evidence`).textContent!;
      expect(ev).toContain(f.evidence);
      expect(ev).toContain(f.limitation);
      const hrefs = [...card.querySelectorAll("a")].map((a) => a.getAttribute("href")!);
      expect(hrefs.length).toBeGreaterThan(0);
      for (const h of hrefs) expect(h).toMatch(/^\/(disentanglement|real-world)\?/);
    }
    expect(screen.getByTestId("not-claimed").textContent).toMatch(/We do not claim/);
  });
});
