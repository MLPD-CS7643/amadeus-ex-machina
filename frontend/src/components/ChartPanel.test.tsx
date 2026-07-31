import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import ChartPanel from "./ChartPanel";
import * as client from "../api/client";
import type { ChartResponse } from "../types";

vi.mock("../api/client");

const CHART: ChartResponse = {
  jobId: "job-1",
  title: "song.mp3",
  key: "C",
  capo: null,
  capoShapes: [],
  duration: 4,
  chords: [
    { name: "C", label: "C:maj", frets: [null, 3, 2, 0, 1, 0], start: 0, end: 2, isNoChord: false },
    { name: "G", label: "G:maj", frets: [3, 2, 0, 0, 0, 3], start: 2, end: 4, isNoChord: false },
  ],
  capoChords: [],
};

describe("ChartPanel", () => {
  beforeEach(() => vi.resetAllMocks());

  it("fetches by job id and renders a diagram per chord", async () => {
    vi.mocked(client.getChart).mockResolvedValue(CHART);
    render(<ChartPanel jobId="job-1" />);
    expect(client.getChart).toHaveBeenCalledWith("job-1");
    await screen.findByTestId("progression");
    // C and G appear in both the progression strip and the "Chords used" grid.
    expect(screen.getAllByLabelText("C chord diagram").length).toBeGreaterThan(0);
    expect(screen.getAllByLabelText("G chord diagram").length).toBeGreaterThan(0);
    expect(screen.getByText("Key: C")).toBeInTheDocument();
  });

  it("shows an audio player only when an audio URL is provided", async () => {
    vi.mocked(client.getChart).mockResolvedValue(CHART);
    const { rerender } = render(<ChartPanel jobId="job-1" />);
    await screen.findByTestId("progression");
    expect(screen.queryByTestId("chart-audio")).not.toBeInTheDocument();

    rerender(<ChartPanel jobId="job-1" audioUrl="blob:audio" />);
    expect(await screen.findByTestId("chart-audio")).toBeInTheDocument();
  });

  it("surfaces an error when the chart cannot load", async () => {
    vi.mocked(client.getChart).mockRejectedValue(new Error("nope"));
    vi.mocked(client.extractErrorMessage).mockReturnValue("Could not load the chord chart.");
    render(<ChartPanel jobId="job-1" />);
    await screen.findByRole("alert");
    expect(screen.getByRole("alert")).toHaveTextContent("Could not load the chord chart.");
  });
});
