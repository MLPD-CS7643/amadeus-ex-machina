import { useCallback, useState } from "react";
import { useDropzone, type FileRejection } from "react-dropzone";
import { ACCEPTED_EXTENSIONS, ACCEPTED_MIME, MAX_FILE_SIZE } from "../constants";
import { extractErrorMessage, uploadFile } from "../api/client";
import { formatBytes } from "../utils/format";

interface Props {
  onUploaded: (jobId: string) => void;
}

function extensionOf(name: string): string {
  const idx = name.lastIndexOf(".");
  return idx >= 0 ? name.slice(idx).toLowerCase() : "";
}

function validateFile(file: File): string | null {
  if (!(ACCEPTED_EXTENSIONS as readonly string[]).includes(extensionOf(file.name))) {
    return `Unsupported format. Supported: ${ACCEPTED_EXTENSIONS.join(", ")}.`;
  }
  if (file.size === 0) return "The selected file is empty.";
  if (file.size > MAX_FILE_SIZE) {
    return `File is too large. Maximum size is ${formatBytes(MAX_FILE_SIZE)}.`;
  }
  return null;
}

export default function UploadComponent({ onUploaded }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);

  const onDrop = useCallback(
    (accepted: File[], rejections: FileRejection[]) => {
      setError(null);
      const candidate = accepted[0] ?? rejections[0]?.file;
      if (!candidate) return;
      const validationError = validateFile(candidate);
      if (validationError) {
        setError(validationError);
        setFile(null);
        return;
      }
      setFile(candidate);
    },
    []
  );

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop,
    accept: ACCEPTED_MIME,
    maxSize: MAX_FILE_SIZE,
    multiple: false,
    noClick: true,
  });

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setError(null);
    setProgress(0);
    try {
      const res = await uploadFile(file, { onProgress: setProgress });
      onUploaded(res.jobId);
    } catch (err) {
      setError(extractErrorMessage(err, "Upload failed. Please try again."));
      setUploading(false);
    }
  };

  return (
    <div className="space-y-4">
      <div
        {...getRootProps()}
        className={`rounded-xl border-2 border-dashed p-10 text-center transition-colors ${
          isDragActive
            ? "border-indigo-500 bg-indigo-50"
            : "border-gray-300 bg-gray-50 hover:border-gray-400"
        }`}
      >
        <input {...getInputProps()} data-testid="file-input" />
        <p className="text-gray-700">
          {isDragActive ? "Drop the audio file here…" : "Drag & drop an audio file here"}
        </p>
        <p className="mt-1 text-sm text-gray-500">
          Supported: {ACCEPTED_EXTENSIONS.join(", ")} · up to {formatBytes(MAX_FILE_SIZE)}
        </p>
        <button
          type="button"
          onClick={open}
          disabled={uploading}
          className="mt-4 rounded-lg bg-white px-4 py-2 text-sm font-medium text-gray-800 shadow ring-1 ring-gray-300 hover:bg-gray-100 disabled:opacity-50"
        >
          Browse files
        </button>
      </div>

      {file && (
        <div className="flex items-center justify-between rounded-lg bg-white p-3 shadow ring-1 ring-gray-200">
          <div className="min-w-0">
            <p className="truncate font-medium text-gray-800">{file.name}</p>
            <p className="text-sm text-gray-500">{formatBytes(file.size)}</p>
          </div>
          <button
            type="button"
            onClick={handleUpload}
            disabled={uploading}
            className="ml-4 shrink-0 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-700 disabled:opacity-50"
          >
            {uploading ? "Uploading…" : "Analyze chords"}
          </button>
        </div>
      )}

      {uploading && (
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
      )}

      {error && (
        <p role="alert" className="rounded-lg bg-red-50 p-3 text-sm text-red-700">
          {error}
        </p>
      )}
    </div>
  );
}
