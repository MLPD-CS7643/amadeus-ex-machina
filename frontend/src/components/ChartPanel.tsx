import { useEffect, useRef, useState, type ReactNode } from "react";
import { getChart, extractErrorMessage } from "../api/client";
import type { ChartChord, ChartResponse } from "../types";
import { formatTime } from "../utils/format";
import ChordDiagram from "./ChordDiagram";
import GuitarTabPanel from "./GuitarTabPanel";

interface Props {
  jobId: string;
  /** Object URL for the uploaded audio; enables synced playback when present. */
  audioUrl?: string;
}

function uniqueChords(chords: ChartChord[]): ChartChord[] {
  const seen = new Set<string>();
  const out: ChartChord[] = [];
  for (const c of chords) {
    if (c.isNoChord || seen.has(c.name)) continue;
    seen.add(c.name);
    out.push(c);
  }
  return out;
}

function Badge({ children }: { children: ReactNode }) {
  return (
    <span className="rounded-full bg-gray-100 px-2 py-0.5 text-xs font-medium text-gray-600">
      {children}
    </span>
  );
}

/**
 * Rich chord-chart view: SVG chord diagrams plus a progression strip that, when the
 * uploaded audio is available, highlights the current chord during playback and lets
 * you click a chord to seek to it. Falls back to a text tab on demand.
 */
export default function ChartPanel({ jobId, audioUrl }: Props) {
  const [chart, setChart] = useState<ChartResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeIndex, setActiveIndex] = useState(-1);
  const [showText, setShowText] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    let active = true;
    setChart(null);
    setError(null);
    getChart(jobId)
      .then((c) => {
        if (active) setChart(c);
      })
      .catch((e) => {
        if (active)
          setError(extractErrorMessage(e, "Could not load the chord chart."));
      });
    return () => {
      active = false;
    };
  }, [jobId]);

  if (error)
    return (
      <p role="alert" className="text-sm text-red-600">
        {error}
      </p>
    );
  if (!chart)
    return <p className="text-sm text-gray-400">Building chord chart…</p>;

  const activeName =
    activeIndex >= 0 ? chart.chords[activeIndex]?.name : undefined;

  const onTimeUpdate = () => {
    const t = audioRef.current?.currentTime ?? 0;
    setActiveIndex(chart.chords.findIndex((c) => t >= c.start && t < c.end));
  };
  const seek = (c: ChartChord) => {
    const a = audioRef.current;
    if (!a) return;
    a.currentTime = c.start;
    void a.play();
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="text-sm font-medium text-gray-600">Chord chart</h3>
        {chart.key && <Badge>Key: {chart.key}</Badge>}
        {chart.capo ? <Badge>Capo {chart.capo}</Badge> : null}
        <button
          type="button"
          onClick={() => setShowText((s) => !s)}
          className="ml-auto text-xs font-medium text-indigo-600 hover:underline"
        >
          {showText ? "Hide text tab" : "Show text tab"}
        </button>
      </div>

      {audioUrl && (
        <audio
          ref={audioRef}
          src={audioUrl}
          controls
          onTimeUpdate={onTimeUpdate}
          className="w-full"
          data-testid="chart-audio"
        />
      )}

      <div>
        <p className="mb-1 text-xs font-medium text-gray-500">
          Progression
          {audioUrl ? " — play to follow along; click a chord to jump" : ""}
        </p>
        <div className="flex gap-2 overflow-x-auto pb-2" data-testid="progression">
          {chart.chords.map((c, i) => {
            const len = Math.max(c.end - c.start, 0.1);
            const isActive = i === activeIndex;
            return (
              <button
                key={i}
                type="button"
                onClick={() => seek(c)}
                disabled={!audioUrl}
                title={`${c.name}  ${formatTime(c.start)}`}
                style={{ flex: `${len} 1 68px` }}
                className={`flex min-w-[68px] flex-col items-center rounded-lg border p-1.5 transition-colors ${
                  isActive
                    ? "border-indigo-500 bg-indigo-50"
                    : "border-gray-200 bg-white"
                } ${audioUrl ? "cursor-pointer hover:border-indigo-300" : ""}`}
              >
                {c.isNoChord ? (
                  <span className="py-6 text-xs text-gray-400">N.C.</span>
                ) : (
                  <ChordDiagram
                    frets={c.frets}
                    name={c.name}
                    active={isActive}
                    width={64}
                  />
                )}
                <span className="mt-0.5 text-[10px] tabular-nums text-gray-400">
                  {formatTime(c.start)}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      <div>
        <p className="mb-1 text-xs font-medium text-gray-500">Chords used</p>
        <div className="flex flex-wrap gap-3">
          {uniqueChords(chart.chords).map((c) => (
            <ChordDiagram
              key={c.name}
              frets={c.frets}
              name={c.name}
              active={c.name === activeName}
            />
          ))}
        </div>
      </div>

      {chart.capo && chart.capoChords.length > 0 && (
        <div>
          <p className="mb-1 text-xs font-medium text-gray-500">
            Capo {chart.capo} — easier open shapes
            {chart.capoShapes.length ? `: ${chart.capoShapes.join(" ")}` : ""}
          </p>
          <div className="flex flex-wrap gap-3">
            {uniqueChords(chart.capoChords).map((c) => (
              <ChordDiagram key={c.name} frets={c.frets} name={c.name} />
            ))}
          </div>
        </div>
      )}

      {showText && <GuitarTabPanel jobId={jobId} />}
    </div>
  );
}
