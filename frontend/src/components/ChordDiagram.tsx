interface Props {
  /** Frets low E -> high e; null = muted string, 0 = open. */
  frets: (number | null)[];
  name?: string;
  active?: boolean;
  /** Rendered pixel width; height scales with it. */
  width?: number;
}

const STRINGS = 6;
const FRET_ROWS = 5; // frets drawn in the window
const PAD_X = 9;
const TOP = 20; // room for the x/o markers above the nut
const STRING_GAP = 11;
const FRET_GAP = 15;

const GRID_W = (STRINGS - 1) * STRING_GAP;
const GRID_H = FRET_ROWS * FRET_GAP;
const VB_W = PAD_X * 2 + GRID_W + 8; // extra right pad for a position label
const VB_H = TOP + GRID_H + 6;

/**
 * A standard chord-box diagram drawn in SVG: nut, frets, finger dots, x/o markers,
 * barre bar, and a position label for shapes played up the neck. Purely presentational.
 */
export default function ChordDiagram({ frets, name, active = false, width = 78 }: Props) {
  const fretted = frets
    .map((f, i) => ({ f: f ?? -1, i }))
    .filter((s) => s.f > 0);
  const maxFret = fretted.length ? Math.max(...fretted.map((s) => s.f)) : 0;
  const open = maxFret <= FRET_ROWS;
  const baseFret = open ? 1 : Math.min(...fretted.map((s) => s.f));

  const minFret = fretted.length ? Math.min(...fretted.map((s) => s.f)) : 0;
  const atMin = fretted.filter((s) => s.f === minFret).map((s) => s.i);
  const barre =
    atMin.length >= 2
      ? { fret: minFret, from: Math.min(...atMin), to: Math.max(...atMin) }
      : null;

  const stringX = (i: number) => PAD_X + i * STRING_GAP;
  const fretY = (row: number) => TOP + (row - 0.5) * FRET_GAP; // dot centre for a fret
  const gridLine = "#9ca3af"; // gray-400
  const dotColor = active ? "#4f46e5" : "#1f2937"; // indigo-600 / gray-800

  return (
    <svg
      viewBox={`0 0 ${VB_W} ${VB_H}`}
      width={width}
      height={(width * VB_H) / VB_W}
      role="img"
      aria-label={name ? `${name} chord diagram` : "chord diagram"}
      className="shrink-0"
    >
      {name && (
        <text
          x={PAD_X + GRID_W / 2}
          y={9}
          textAnchor="middle"
          fontSize="11"
          fontWeight="600"
          fill={active ? "#4f46e5" : "#111827"}
        >
          {name}
        </text>
      )}

      {/* nut (thick when at the top of the neck) */}
      <line
        x1={stringX(0)}
        y1={TOP}
        x2={stringX(STRINGS - 1)}
        y2={TOP}
        stroke={gridLine}
        strokeWidth={open ? 3 : 1}
      />
      {Array.from({ length: FRET_ROWS }, (_, r) => (
        <line
          key={`f${r}`}
          x1={stringX(0)}
          y1={TOP + (r + 1) * FRET_GAP}
          x2={stringX(STRINGS - 1)}
          y2={TOP + (r + 1) * FRET_GAP}
          stroke={gridLine}
          strokeWidth={1}
        />
      ))}
      {Array.from({ length: STRINGS }, (_, i) => (
        <line
          key={`s${i}`}
          x1={stringX(i)}
          y1={TOP}
          x2={stringX(i)}
          y2={TOP + GRID_H}
          stroke={gridLine}
          strokeWidth={1}
        />
      ))}

      {!open && (
        <text x={VB_W - 4} y={fretY(1) + 3} textAnchor="end" fontSize="9" fill="#6b7280">
          {baseFret}fr
        </text>
      )}

      {/* x (muted) / o (open) markers above the nut */}
      {frets.map((f, i) =>
        f === null ? (
          <text
            key={`m${i}`}
            x={stringX(i)}
            y={TOP - 4}
            textAnchor="middle"
            fontSize="9"
            fill="#6b7280"
          >
            ×
          </text>
        ) : f === 0 ? (
          <circle
            key={`m${i}`}
            cx={stringX(i)}
            cy={TOP - 7}
            r={3}
            fill="none"
            stroke="#6b7280"
            strokeWidth={1}
          />
        ) : null
      )}

      {/* barre bar */}
      {barre && (
        <rect
          x={stringX(barre.from) - 4}
          y={fretY(barre.fret - baseFret + 1) - 4}
          width={stringX(barre.to) - stringX(barre.from) + 8}
          height={8}
          rx={4}
          fill={dotColor}
        />
      )}

      {/* finger dots */}
      {frets.map((f, i) =>
        f && f > 0 && !(barre && f === barre.fret) ? (
          <circle
            key={`d${i}`}
            cx={stringX(i)}
            cy={fretY(f - baseFret + 1)}
            r={4}
            fill={dotColor}
          />
        ) : null
      )}
    </svg>
  );
}
