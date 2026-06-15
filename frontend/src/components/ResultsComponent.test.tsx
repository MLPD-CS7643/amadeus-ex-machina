import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import ResultsComponent from "./ResultsComponent";
import * as client from "../api/client";
import type { ResultsResponse } from "../types";

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

describe("ResultsComponent", () => {
  beforeEach(() => vi.resetAllMocks());

  it("previews the first 10 annotations and notes the remainder (Property 11)", () => {
    render(<ResultsComponent results={results(15)} />);
    expect(screen.getAllByTestId("preview-row")).toHaveLength(10);
    expect(screen.getByText(/\+ 5 more/)).toBeInTheDocument();
  });

  it("shows all annotations when fewer than 10", () => {
    render(<ResultsComponent results={results(3)} />);
    expect(screen.getAllByTestId("preview-row")).toHaveLength(3);
    expect(screen.queryByText(/more/)).not.toBeInTheDocument();
  });

  it("downloads with the {stem}_chords.lab name (Property 10)", async () => {
    vi.mocked(client.downloadAnnotations).mockResolvedValue();
    render(<ResultsComponent results={results(2, "My_Song.wav")} />);
    fireEvent.click(screen.getByRole("button", { name: /Download/i }));
    await waitFor(() =>
      expect(client.downloadAnnotations).toHaveBeenCalledWith(
        "job-1",
        "My_Song_chords.lab"
      )
    );
  });
});
