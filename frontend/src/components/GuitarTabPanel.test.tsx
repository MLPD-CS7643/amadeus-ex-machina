import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import GuitarTabPanel from "./GuitarTabPanel";
import * as client from "../api/client";

vi.mock("../api/client");

const SAMPLE_TAB = "Guitar chord chart\n\ne|---0---3---\nB|---1---0---\n";

describe("GuitarTabPanel", () => {
  beforeEach(() => vi.resetAllMocks());

  it("shows a loading hint before the tab resolves", () => {
    vi.mocked(client.fetchTab).mockReturnValue(new Promise(() => {}));
    render(<GuitarTabPanel jobId="job-1" />);
    expect(screen.getByText(/Rendering guitar tab/i)).toBeInTheDocument();
  });

  it("fetches by job id and renders the tab text", async () => {
    vi.mocked(client.fetchTab).mockResolvedValue(SAMPLE_TAB);
    render(<GuitarTabPanel jobId="job-42" />);
    expect(client.fetchTab).toHaveBeenCalledWith("job-42");
    const tab = await screen.findByTestId("guitar-tab");
    expect(tab).toHaveTextContent("e|---0---3---");
  });

  it("surfaces an error when the tab cannot be rendered", async () => {
    vi.mocked(client.fetchTab).mockRejectedValue(new Error("boom"));
    vi.mocked(client.extractErrorMessage).mockReturnValue("Could not render.");
    render(<GuitarTabPanel jobId="job-1" />);
    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent("Could not render.")
    );
    expect(screen.queryByTestId("guitar-tab")).not.toBeInTheDocument();
  });
});
