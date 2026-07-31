import { useEffect, useState } from "react";
import { fetchTab, extractErrorMessage } from "../api/client";

interface Props {
  jobId: string;
}

/**
 * Fetches the server-rendered ASCII guitar tab for a completed job and shows it
 * inline in a monospace block. The tab is derived from the same chord annotations
 * as the .lab download, so it stays consistent with the rest of the results.
 */
export default function GuitarTabPanel({ jobId }: Props) {
  const [tab, setTab] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setTab(null);
    setError(null);
    fetchTab(jobId)
      .then((text) => {
        if (active) setTab(text);
      })
      .catch((err) => {
        if (active)
          setError(extractErrorMessage(err, "Could not render the guitar tab."));
      });
    return () => {
      active = false;
    };
  }, [jobId]);

  return (
    <div>
      <h3 className="mb-2 text-sm font-medium text-gray-600">Guitar tab</h3>
      {error ? (
        <p role="alert" className="text-sm text-red-600">
          {error}
        </p>
      ) : tab === null ? (
        <p className="text-sm text-gray-400">Rendering guitar tab…</p>
      ) : (
        <pre
          data-testid="guitar-tab"
          className="max-h-96 overflow-auto rounded-lg bg-gray-900 p-4 font-mono text-xs leading-relaxed text-gray-100"
        >
          {tab}
        </pre>
      )}
    </div>
  );
}
