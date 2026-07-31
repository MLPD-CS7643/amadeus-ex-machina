import axios, { AxiosError } from "axios";
import type {
  ApiErrorBody,
  ChartResponse,
  ResultsResponse,
  StatusResponse,
  UploadResponse,
} from "../types";

// Empty base => same-origin "/api" (Vite dev proxy forwards to the backend).
const baseURL = import.meta.env.VITE_API_BASE_URL ?? "";

export const api = axios.create({ baseURL });

export interface UploadOptions {
  onProgress?: (percent: number) => void;
  signal?: AbortSignal;
}

export async function uploadFile(
  file: File,
  opts: UploadOptions = {}
): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const resp = await api.post<UploadResponse>("/api/upload", form, {
    signal: opts.signal,
    onUploadProgress: (event) => {
      if (opts.onProgress && event.total) {
        opts.onProgress(Math.round((event.loaded / event.total) * 100));
      }
    },
  });
  return resp.data;
}

export async function getJobStatus(jobId: string): Promise<StatusResponse> {
  const resp = await api.get<StatusResponse>(`/api/status/${jobId}`);
  return resp.data;
}

export async function getResults(jobId: string): Promise<ResultsResponse> {
  const resp = await api.get<ResultsResponse>(`/api/results/${jobId}`);
  return resp.data;
}

/** Fetch the structured chord chart (shapes + timing) for rich rendering. */
export async function getChart(jobId: string): Promise<ChartResponse> {
  const resp = await api.get<ChartResponse>(`/api/chart/${jobId}`);
  return resp.data;
}

async function downloadBlob(path: string, downloadName: string): Promise<void> {
  const resp = await api.get(path, { responseType: "blob" });
  const url = URL.createObjectURL(resp.data as Blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = downloadName;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

/** Fetch the chords rendered as ASCII guitar tab, as plain text for inline display. */
export async function fetchTab(jobId: string): Promise<string> {
  const resp = await api.get(`/api/tab/${jobId}`, { responseType: "text" });
  return resp.data as string;
}

/** Download the chord annotations as a .lab file. */
export async function downloadAnnotations(
  jobId: string,
  downloadName: string
): Promise<void> {
  await downloadBlob(`/api/download/${jobId}`, downloadName);
}

/** Download the chords rendered as an ASCII guitar-tab (.tab) file. */
export async function downloadTab(
  jobId: string,
  downloadName: string
): Promise<void> {
  await downloadBlob(`/api/download/${jobId}?format=tab`, downloadName);
}

/** Pull a user-facing message out of an Axios error, falling back gracefully. */
export function extractErrorMessage(
  err: unknown,
  fallback = "Something went wrong. Please try again."
): string {
  const axiosErr = err as AxiosError<ApiErrorBody>;
  return axiosErr?.response?.data?.error?.message ?? axiosErr?.message ?? fallback;
}
