import { useEffect, useRef, useState } from "react";
import { extractErrorMessage, getJobStatus } from "../api/client";
import { POLL_INTERVAL_MS } from "../constants";
import type { JobStatus, StatusResponse } from "../types";
import { formatTime } from "../utils/format";

interface Props {
  jobId: string;
  onComplete: () => void;
  onFailed: (message: string) => void;
  pollInterval?: number;
}

const MAX_CONSECUTIVE_FAILURES = 5;

const STATUS_LABEL: Record<JobStatus, string> = {
  queued: "Queued",
  processing: "Processing audio",
  complete: "Complete",
  failed: "Failed",
};

export default function StatusComponent({
  jobId,
  onComplete,
  onFailed,
  pollInterval = POLL_INTERVAL_MS,
}: Props) {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [netError, setNetError] = useState<string | null>(null);

  // Keep callbacks in refs so identity changes don't restart polling.
  const onCompleteRef = useRef(onComplete);
  const onFailedRef = useRef(onFailed);
  onCompleteRef.current = onComplete;
  onFailedRef.current = onFailed;

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;
    let failures = 0;

    const poll = async () => {
      try {
        const next = await getJobStatus(jobId);
        if (cancelled) return;
        failures = 0;
        setNetError(null);
        setStatus(next);
        if (next.status === "complete") {
          onCompleteRef.current();
          return;
        }
        if (next.status === "failed") {
          onFailedRef.current(next.message || "Processing failed.");
          return;
        }
        timer = window.setTimeout(poll, pollInterval);
      } catch (err) {
        if (cancelled) return;
        failures += 1;
        if (failures >= MAX_CONSECUTIVE_FAILURES) {
          onFailedRef.current("Lost connection to the server.");
          return;
        }
        setNetError(extractErrorMessage(err, "Connection issue. Retrying…"));
        timer = window.setTimeout(poll, pollInterval);
      }
    };

    poll();
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [jobId, pollInterval]);

  const progress = status?.progress ?? 0;
  const eta = status?.estimatedTimeRemaining;

  return (
    <div className="space-y-4 rounded-xl bg-white p-6 shadow ring-1 ring-gray-200">
      <div className="flex items-center gap-3">
        <span
          className="inline-block h-3 w-3 animate-pulse rounded-full bg-indigo-500"
          aria-hidden
        />
        <span className="font-medium text-gray-800" data-testid="status-label">
          {status ? STATUS_LABEL[status.status] : "Starting…"}
        </span>
      </div>

      <div>
        <div className="mb-1 flex justify-between text-sm text-gray-500">
          <span>{status?.message ?? "Working…"}</span>
          <span>{progress}%</span>
        </div>
        <div className="h-2 w-full overflow-hidden rounded-full bg-gray-200">
          <div
            className="h-full bg-indigo-600 transition-all"
            style={{ width: `${progress}%` }}
            role="progressbar"
            aria-valuenow={progress}
            aria-valuemin={0}
            aria-valuemax={100}
          />
        </div>
      </div>

      {eta != null && eta > 0 && (
        <p className="text-sm text-gray-500">
          Estimated time remaining: {formatTime(eta)}
        </p>
      )}

      {netError && (
        <p role="alert" className="text-sm text-amber-600">
          {netError}
        </p>
      )}
    </div>
  );
}
