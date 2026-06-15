import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import UploadComponent from "./UploadComponent";
import * as client from "../api/client";

vi.mock("../api/client");

function makeFile(name: string, size = 1024, type = "audio/wav"): File {
  return new File([new Uint8Array(size)], name, { type });
}

describe("UploadComponent", () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it("rejects an unsupported file format", async () => {
    render(<UploadComponent onUploaded={() => {}} />);
    const input = screen.getByTestId("file-input");
    fireEvent.change(input, {
      target: { files: [makeFile("notes.txt", 16, "text/plain")] },
    });
    expect(await screen.findByRole("alert")).toHaveTextContent(/Unsupported format/i);
  });

  it("accepts a valid file and uploads it", async () => {
    vi.mocked(client.uploadFile).mockResolvedValue({
      jobId: "job-123",
      status: "queued",
      message: "ok",
    });
    const onUploaded = vi.fn();
    render(<UploadComponent onUploaded={onUploaded} />);

    fireEvent.change(screen.getByTestId("file-input"), {
      target: { files: [makeFile("song.wav", 4096, "audio/wav")] },
    });

    expect(await screen.findByText("song.wav")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Analyze chords/i }));

    await waitFor(() => expect(onUploaded).toHaveBeenCalledWith("job-123"));
    expect(client.uploadFile).toHaveBeenCalledTimes(1);
  });
});
