import { useCallback, useState } from "react";
import UploadComponent from "./components/UploadComponent";
import StatusComponent from "./components/StatusComponent";
import VisualizationComponent from "./components/VisualizationComponent";
import ResultsComponent from "./components/ResultsComponent";
import ErrorBoundary from "./components/ErrorBoundary";
import { extractErrorMessage, getResults } from "./api/client";
import type { ResultsResponse } from "./types";

type Phase = "idle" | "processing" | "complete" | "failed";

export default function App() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [jobId, setJobId] = useState<string | null>(null);
  const [results, setResults] = useState<ResultsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const reset = useCallback(() => {
    setPhase("idle");
    setJobId(null);
    setResults(null);
    setError(null);
  }, []);

  const handleUploaded = useCallback((id: string) => {
    setJobId(id);
    setResults(null);
    setError(null);
    setPhase("processing");
  }, []);

  const handleComplete = useCallback(async () => {
    if (!jobId) return;
    try {
      const res = await getResults(jobId);
      setResults(res);
      setPhase("complete");
    } catch (err) {
      setError(extractErrorMessage(err, "Could not load results."));
      setPhase("failed");
    }
  }, [jobId]);

  const handleFailed = useCallback((message: string) => {
    setError(message);
    setPhase("failed");
  }, []);

  return (
    <ErrorBoundary>
      <div className="min-h-full bg-gray-100 text-gray-900">
        <header className="border-b border-gray-200 bg-white">
          <div className="mx-auto max-w-3xl px-4 py-5">
            <h1 className="text-xl font-bold">🎵 Chord Recognition</h1>
            <p className="text-sm text-gray-500">
              Upload audio and get a time-stamped chord annotation.
            </p>
          </div>
        </header>

        <main className="mx-auto max-w-3xl space-y-6 px-4 py-8">
          {phase === "idle" && <UploadComponent onUploaded={handleUploaded} />}

          {phase === "processing" && jobId && (
            <StatusComponent
              jobId={jobId}
              onComplete={handleComplete}
              onFailed={handleFailed}
            />
          )}

          {phase === "complete" && results && (
            <>
              <VisualizationComponent
                annotations={results.annotations}
                duration={results.duration}
              />
              <ResultsComponent results={results} />
              <div className="text-center">
                <button
                  type="button"
                  onClick={reset}
                  className="rounded-lg bg-white px-4 py-2 text-sm font-medium text-gray-700 shadow ring-1 ring-gray-300 hover:bg-gray-50"
                >
                  Analyze another file
                </button>
              </div>
            </>
          )}

          {phase === "failed" && (
            <div className="rounded-xl bg-red-50 p-6 text-center ring-1 ring-red-200">
              <p className="font-medium text-red-700">
                {error ?? "Processing failed."}
              </p>
              <button
                type="button"
                onClick={reset}
                className="mt-3 rounded-lg bg-red-600 px-4 py-2 text-sm font-semibold text-white hover:bg-red-700"
              >
                Try another file
              </button>
            </div>
          )}
        </main>
      </div>
    </ErrorBoundary>
  );
}
