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

export interface ChartChord {
  name: string;
  label: string;
  frets: (number | null)[]; // low E -> high e; null = muted string
  start: number;
  end: number;
  isNoChord: boolean;
}

export interface ChartResponse {
  jobId: string;
  title: string | null;
  key: string | null;
  capo: number | null;
  capoShapes: string[];
  duration: number;
  chords: ChartChord[];
  capoChords: ChartChord[];
}

export interface ApiErrorBody {
  error: {
    code: string;
    message: string;
    details: unknown;
  };
}
