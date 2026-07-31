export const ACCEPTED_EXTENSIONS = [".wav", ".mp3", ".flac"] as const;

export const ACCEPTED_MIME: Record<string, string[]> = {
  "audio/wav": [".wav"],
  "audio/x-wav": [".wav"],
  "audio/mpeg": [".mp3"],
  "audio/flac": [".flac"],
  "audio/x-flac": [".flac"],
};

export const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50 MB

export const POLL_INTERVAL_MS = 2000;

// Audio longer than this gets a horizontally scrollable timeline.
export const SCROLL_THRESHOLD_SECONDS = 60;
