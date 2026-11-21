/** Experiment Lab: the logged per-epoch training curves (no invented trajectories). */
import { afterEach, describe, expect, it } from "vitest";
import { cleanup, fireEvent, screen } from "@testing-library/react";
import { renderAt } from "../test-utils";
import traj from "../data/trajectories.generated.json";

afterEach(cleanup);

describe("Experiment Lab training curves", () => {
  it("shows the logged curves of the selected setting, and says so when a scale was not logged", async () => {
    renderAt("/lab");
    expect(await screen.findByTestId("trajectory-chart")).toBeTruthy();
    expect(screen.getByTestId("trajectory-chart").textContent).toMatch(/results\/disentangle\/disentangle_trajectories\.csv/);
    expect(screen.getByTestId("trajectory-chart").textContent).toMatch(/validation split/);
    // α = 5 was never run in the decomposition: no curve is drawn for it
    fireEvent.change(document.querySelector('input[type="range"]')!, { target: { value: "1" } });
    expect((await screen.findByTestId("traj-missing")).textContent).toMatch(/Logged scales: 1×, 10×, 100×/);
  });

  it("the generated data holds one series per logged seed, epoch by epoch", () => {
    const s = traj.series.synthetic_adam;
    expect(s.alphas).toEqual([1, 10, 100]);
    const runs = s.data["100"].forward.val_dominance;
    expect(runs).toHaveLength(5);
    runs.forEach((r) => expect(r).toHaveLength(40));
    expect(traj.series.cremad_adam.epochs).toBe(30);
    expect(Object.keys(traj.series)).toEqual(["synthetic_adam", "synthetic_sgd", "cremad_adam"]);
  });
});
