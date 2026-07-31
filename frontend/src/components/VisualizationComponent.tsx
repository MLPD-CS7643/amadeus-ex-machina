import { useMemo } from "react";
import type { ChordAnnotation } from "../types";
import { getChordColor, getChordTextColor } from "../utils/chordColors";
import { formatTime } from "../utils/format";

interface Props {
  annotations: ChordAnnotation[];
  duration: number;
}

const PX_PER_SEC = 24; // long clips (>~26s) overflow -> horizontal scroll
const MIN_WIDTH = 640;
const TRACK_HEIGHT = 64;

function buildTicks(duration: number): number[] {
  if (duration <= 0) return [0];
  const steps = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
  const target = duration / 10;
  const step = steps.find((s) => s >= target) ?? 600;
  const ticks: number[] = [];
  for (let t = 0; t <= duration + 1e-6; t += step) ticks.push(Math.round(t));
  return ticks;
}

export default function VisualizationComponent({ annotations, duration }: Props) {
  const lastEnd =
    annotations.length > 0 ? annotations[annotations.length - 1].endTime : 0;
  const totalDuration = (duration > 0 ? duration : lastEnd) || 1;
  const width = Math.max(totalDuration * PX_PER_SEC, MIN_WIDTH);
  const ticks = useMemo(() => buildTicks(totalDuration), [totalDuration]);

  const legend = useMemo(() => {
    const seen = new Map<string, string>();
    for (const a of annotations) {
      if (!seen.has(a.chordLabel)) seen.set(a.chordLabel, getChordColor(a.chordLabel));
    }
    return Array.from(seen.entries());
  }, [annotations]);

  return (
    <div className="space-y-3 rounded-xl bg-white p-6 shadow ring-1 ring-gray-200">
      <div className="flex items-center justify-between">
        <h2 className="font-semibold text-gray-800">Chord timeline</h2>
        <span className="text-sm text-gray-500">
          Duration: {formatTime(totalDuration)}
        </span>
      </div>

      <div className="overflow-x-auto" data-testid="timeline-scroll">
        <div
          style={{ width, height: TRACK_HEIGHT, position: "relative" }}
          className="rounded-md ring-1 ring-gray-200"
          role="img"
          aria-label={`Chord timeline with ${annotations.length} segments`}
        >
          {annotations.map((a, i) => {
            const leftPct = (a.startTime / totalDuration) * 100;
            const widthPct = ((a.endTime - a.startTime) / totalDuration) * 100;
            const segPx = (widthPct / 100) * width;
            return (
              <div
                key={`${i}-${a.startTime}`}
                data-testid="chord-segment"
                data-chord={a.chordLabel}
                data-start={a.startTime}
                data-end={a.endTime}
                title={`${a.chordLabel}  (${formatTime(a.startTime)}–${formatTime(
                  a.endTime
                )})`}
                style={{
                  position: "absolute",
                  left: `${leftPct}%`,
                  width: `${widthPct}%`,
                  height: "100%",
                  backgroundColor: getChordColor(a.chordLabel),
                  color: getChordTextColor(a.chordLabel),
                }}
                className="flex items-center justify-center overflow-hidden border-r border-white/40 text-xs font-medium"
              >
                {segPx > 30 ? a.chordLabel : ""}
              </div>
            );
          })}
        </div>

        {/* time axis */}
        <div style={{ width, position: "relative", height: 22 }} className="mt-1">
          {ticks.map((t) => (
            <div
              key={t}
              style={{ position: "absolute", left: `${(t / totalDuration) * 100}%` }}
              className="flex flex-col items-start"
            >
              <span className="block h-2 border-l border-gray-300" />
              <span className="text-[10px] text-gray-400">{formatTime(t)}</span>
            </div>
          ))}
        </div>
      </div>

      {legend.length > 0 && (
        <div className="flex flex-wrap gap-x-4 gap-y-1 pt-2" data-testid="legend">
          {legend.map(([label, color]) => (
            <span key={label} className="flex items-center gap-1.5 text-xs text-gray-600">
              <span
                className="inline-block h-3 w-3 rounded-sm ring-1 ring-black/10"
                style={{ backgroundColor: color }}
              />
              {label}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
