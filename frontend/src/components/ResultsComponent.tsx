import { useState } from "react";
import type { ResultsResponse } from "../types";
import {
  downloadAnnotations,
  downloadTab,
  extractErrorMessage,
} from "../api/client";
import { formatTime, stripExtension } from "../utils/format";
import ChartPanel from "./ChartPanel";

interface Props {
  results: ResultsResponse;
  audioUrl?: string;
}

const PREVIEW_COUNT = 10;

type DownloadFormat = "lab" | "tab";

export default function ResultsComponent({ results, audioUrl }: Props) {
  const [downloading, setDownloading] = useState<DownloadFormat | null>(null);
  const [error, setError] = useState<string | null>(null);

  const preview = results.annotations.slice(0, PREVIEW_COUNT);
  const remaining = Math.max(0, results.annotations.length - PREVIEW_COUNT);
  const stem = stripExtension(results.filename || "audio");
  const labName = `${stem}_chords.lab`;
  const tabName = `${stem}_chords.txt`;

  const handleDownload = async (format: DownloadFormat) => {
    setDownloading(format);
    setError(null);
    try {
      if (format === "tab") {
        await downloadTab(results.jobId, tabName);
      } else {
        await downloadAnnotations(results.jobId, labName);
      }
    } catch (err) {
      setError(extractErrorMessage(err, "Download failed. Please try again."));
    } finally {
      setDownloading(null);
    }
  };

  return (
    <div className="space-y-4 rounded-xl bg-white p-6 shadow ring-1 ring-gray-200">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="min-w-0">
          <h2 className="truncate font-semibold text-gray-800">{results.filename}</h2>
          <p className="text-sm text-gray-500">
            {results.annotations.length} chord segments ·{" "}
            {formatTime(results.duration)}
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => handleDownload("tab")}
            disabled={downloading !== null}
            title="ASCII guitar tab (chord chart)"
            className="rounded-lg bg-white px-4 py-2 text-sm font-semibold text-indigo-700 ring-1 ring-indigo-300 hover:bg-indigo-50 disabled:opacity-50"
          >
            {downloading === "tab" ? "Downloading…" : "Guitar Tab"}
          </button>
          <button
            type="button"
            onClick={() => handleDownload("lab")}
            disabled={downloading !== null}
            className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-700 disabled:opacity-50"
          >
            {downloading === "lab" ? "Downloading…" : "Download .lab"}
          </button>
        </div>
      </div>

      <div>
        <h3 className="mb-2 text-sm font-medium text-gray-600">
          Preview {preview.length > 0 ? `(first ${preview.length})` : ""}
        </h3>
        <div className="overflow-hidden rounded-lg ring-1 ring-gray-200">
          <table className="w-full text-left text-sm">
            <thead className="bg-gray-50 text-gray-500">
              <tr>
                <th className="px-3 py-2 font-medium">#</th>
                <th className="px-3 py-2 font-medium">Start</th>
                <th className="px-3 py-2 font-medium">End</th>
                <th className="px-3 py-2 font-medium">Chord</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {preview.map((a, i) => (
                <tr key={i} data-testid="preview-row">
                  <td className="px-3 py-1.5 text-gray-400">{i + 1}</td>
                  <td className="px-3 py-1.5 tabular-nums">{a.startTime.toFixed(2)}s</td>
                  <td className="px-3 py-1.5 tabular-nums">{a.endTime.toFixed(2)}s</td>
                  <td className="px-3 py-1.5 font-medium text-gray-800">
                    {a.chordLabel}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {remaining > 0 && (
          <p className="mt-2 text-sm text-gray-400">
            + {remaining} more — download the .lab file for the full annotation.
          </p>
        )}
      </div>

      <ChartPanel jobId={results.jobId} audioUrl={audioUrl} />

      {error && (
        <p role="alert" className="text-sm text-red-600">
          {error}
        </p>
      )}
    </div>
  );
}
