export type JobStatus = "queued" | "processing" | "complete" | "failed";

export interface ChordAnnotation {
  startTime: number;
  endTime: number;
  chordLabel: string;
}

export interface UploadResponse {
  jobId: string;
  status: JobStatus;
  message: string;
}

export interface StatusResponse {
  jobId: string;
  status: JobStatus;
  progress: number;
  estimatedTimeRemaining: number | null;
  message: string;
}

export interface ResultsResponse {
  jobId: string;
  filename: string;
  duration: number;
  annotations: ChordAnnotation[];
}

export interface ApiErrorBody {
  error: {
    code: string;
    message: string;
    details: unknown;
  };
}
