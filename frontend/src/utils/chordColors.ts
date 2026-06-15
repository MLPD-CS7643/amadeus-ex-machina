/**
 * Deterministic chord -> color mapping for the timeline.
 *
 * Color is driven primarily by the chord ROOT (12 distinct hues, musically
 * meaningful) and secondarily by the QUALITY (deterministic lightness/saturation),
 * so different chord types generally get different colors. Combined with the chord
 * label text rendered on each segment, this satisfies the "visually distinct" goal
 * (design Property 13).
 */
const PITCH_CLASSES: Record<string, number> = {
  C: 0, "B#": 0,
  "C#": 1, Db: 1,
  D: 2,
  "D#": 3, Eb: 3,
  E: 4, Fb: 4,
  F: 5, "E#": 5,
  "F#": 6, Gb: 6,
  G: 7,
  "G#": 8, Ab: 8,
  A: 9,
  "A#": 10, Bb: 10,
  B: 11, Cb: 11,
};

const NO_CHORD_BG = "hsl(0, 0%, 82%)";

function hashString(value: string): number {
  let hash = 0;
  for (let i = 0; i < value.length; i++) {
    hash = (hash * 31 + value.charCodeAt(i)) >>> 0;
  }
  return hash;
}

export function isNoChord(label: string): boolean {
  return label === "" || label === "N" || label === "X";
}

export function parseChord(label: string): { root: string; quality: string } {
  const idx = label.indexOf(":");
  if (idx === -1) return { root: label, quality: "" };
  return { root: label.slice(0, idx), quality: label.slice(idx + 1) };
}

function lightnessFor(quality: string): number {
  return 45 + ((hashString(quality) >> 3) % 25); // 45–69
}

export function getChordColor(label: string): string {
  if (isNoChord(label)) return NO_CHORD_BG;
  const { root, quality } = parseChord(label);
  const pc = PITCH_CLASSES[root];
  const hue = pc === undefined ? hashString(root) % 360 : pc * 30;
  const sat = 55 + (hashString(quality) % 30); // 55–84
  return `hsl(${hue}, ${sat}%, ${lightnessFor(quality)}%)`;
}

/** Pick a readable text color for a chord segment background. */
export function getChordTextColor(label: string): string {
  if (isNoChord(label)) return "#374151";
  return lightnessFor(parseChord(label).quality) < 58 ? "#ffffff" : "#111827";
}
