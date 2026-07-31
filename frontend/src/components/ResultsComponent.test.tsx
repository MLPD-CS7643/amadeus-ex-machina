import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import ResultsComponent from "./ResultsComponent";
import * as client from "../api/client";
import type { ChartResponse, ResultsResponse } from "../types";

vi.mock("../api/client");

function results(count: number, filename = "song.wav"): ResultsResponse {
  return {
    jobId: "job-1",
    filename,
    duration: count,
    annotations: Array.from({ length: count }, (_, i) => ({
      startTime: i,
      endTime: i + 1,
      chordLabel: `C:${i}`,
    })),
  };
}

const SAMPLE_CHART: ChartResponse = {
  jobId: "job-1",
  title: "song.wav",
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

describe("ResultsComponent", () => {
  beforeEach(() => {
    vi.resetAllMocks();
    // ChartPanel fetches the chart on mount; give it a default so it never rejects.
    vi.mocked(client.getChart).mockResolvedValue(SAMPLE_CHART);
  });

  it("previews the first 10 annotations and notes the remainder (Property 11)", async () => {
    render(<ResultsComponent results={results(15)} />);
    expect(screen.getAllByTestId("preview-row")).toHaveLength(10);
    expect(screen.getByText(/\+ 5 more/)).toBeInTheDocument();
    await screen.findByTestId("progression"); // flush the panel's mount-time fetch
  });

  it("shows all annotations when fewer than 10", async () => {
    render(<ResultsComponent results={results(3)} />);
    expect(screen.getAllByTestId("preview-row")).toHaveLength(3);
    expect(screen.queryByText(/more/)).not.toBeInTheDocument();
    await screen.findByTestId("progression"); // flush the panel's mount-time fetch
  });

  it("downloads with the {stem}_chords.lab name (Property 10)", async () => {
    vi.mocked(client.downloadAnnotations).mockResolvedValue();
    render(<ResultsComponent results={results(2, "My_Song.wav")} />);
    fireEvent.click(screen.getByRole("button", { name: /Download \.lab/i }));
    await waitFor(() =>
      expect(client.downloadAnnotations).toHaveBeenCalledWith(
        "job-1",
        "My_Song_chords.lab"
      )
    );
  });

  it("downloads the guitar tab with the {stem}_chords.txt name", async () => {
    vi.mocked(client.downloadTab).mockResolvedValue();
    render(<ResultsComponent results={results(2, "My_Song.wav")} />);
    fireEvent.click(screen.getByRole("button", { name: /Guitar Tab/i }));
    await waitFor(() =>
      expect(client.downloadTab).toHaveBeenCalledWith("job-1", "My_Song_chords.txt")
    );
  });

  it("renders the chord chart with SVG diagrams", async () => {
    render(<ResultsComponent results={results(2, "My_Song.wav")} />);
    expect(client.getChart).toHaveBeenCalledWith("job-1");
    await screen.findByTestId("progression");
    expect((await screen.findAllByLabelText(/chord diagram/)).length).toBeGreaterThan(0);
  });
});
