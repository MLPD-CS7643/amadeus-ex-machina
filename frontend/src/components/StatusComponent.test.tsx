import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import StatusComponent from "./StatusComponent";
import * as client from "../api/client";
import type { StatusResponse } from "../types";

vi.mock("../api/client");

function status(partial: Partial<StatusResponse>): StatusResponse {
  return {
    jobId: "x",
    status: "processing",
    progress: 0,
    estimatedTimeRemaining: null,
    message: "",
    ...partial,
  };
}

describe("StatusComponent", () => {
  beforeEach(() => vi.resetAllMocks());

  it("renders progress while processing", async () => {
    vi.mocked(client.getJobStatus).mockResolvedValue(
      status({ status: "processing", progress: 45, message: "Extracting features" })
    );
    render(
      <StatusComponent
        jobId="x"
        onComplete={() => {}}
        onFailed={() => {}}
        pollInterval={10_000}
      />
    );
    expect(await screen.findByText("45%")).toBeInTheDocument();
    expect(screen.getByTestId("status-label")).toHaveTextContent("Processing");
  });

  it("calls onComplete when the job completes (Property 7)", async () => {
    vi.mocked(client.getJobStatus).mockResolvedValue(
      status({ status: "complete", progress: 100, message: "Complete" })
    );
    const onComplete = vi.fn();
    render(<StatusComponent jobId="x" onComplete={onComplete} onFailed={() => {}} />);
    await waitFor(() => expect(onComplete).toHaveBeenCalledTimes(1));
  });

  it("calls onFailed when the job fails", async () => {
    vi.mocked(client.getJobStatus).mockResolvedValue(
      status({ status: "failed", message: "boom" })
    );
    const onFailed = vi.fn();
    render(<StatusComponent jobId="x" onComplete={() => {}} onFailed={onFailed} />);
    await waitFor(() => expect(onFailed).toHaveBeenCalledWith("boom"));
  });
});
