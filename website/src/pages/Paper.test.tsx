/** The Paper page serves the CA research paper PDF and describes that paper. */
import { afterEach, describe, expect, it } from "vitest";
import { cleanup, fireEvent, screen } from "@testing-library/react";
import { currentLocation, renderAt } from "../test-utils";
import { paper } from "../data/research";

afterEach(cleanup);

describe("Paper page", () => {
  it("links to the CA research paper PDF and shows its title and authors", () => {
    renderAt("/paper");
    expect(screen.getByTestId("paper-pdf").getAttribute("href")).toMatch(/ca_research_paper\.pdf$/);
    expect(screen.getByTestId("paper-preview-title").textContent).toMatch(/^When Scale Silences a Modality/);
    expect(document.body.textContent).toMatch(/Komal Lamba · Ankita Maji · Prabhutee Neb · Premananda Sahu/);
    expect(document.body.textContent).toMatch(/7 pages · 15 references/);
    expect(paper.pdfPath).toBe("ca_research_paper.pdf");
  });

  it("section links stay on the Paper page (hash-router safe)", () => {
    renderAt("/paper");
    fireEvent.click(screen.getByRole("link", { name: "Results" }));
    expect(currentLocation.pathname).toBe("/paper");
    expect(currentLocation.search).toBe("?section=results");
  });
});
