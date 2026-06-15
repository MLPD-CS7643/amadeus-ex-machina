import axios, { AxiosError } from "axios";
import type {
  ApiErrorBody,
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

export async function downloadAnnotations(
  jobId: string,
  downloadName: string
): Promise<void> {
  const resp = await api.get(`/api/download/${jobId}`, { responseType: "blob" });
  const url = URL.createObjectURL(resp.data as Blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = downloadName;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

/** Pull a user-facing message out of an Axios error, falling back gracefully. */
export function extractErrorMessage(
  err: unknown,
  fallback = "Something went wrong. Please try again."
): string {
  const axiosErr = err as AxiosError<ApiErrorBody>;
  return axiosErr?.response?.data?.error?.message ?? axiosErr?.message ?? fallback;
}
