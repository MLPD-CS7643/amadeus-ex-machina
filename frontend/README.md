# Chord Recognition Frontend

React + TypeScript single-page app for the chord-recognition web service. Upload an
audio file, watch processing status, and explore the predicted chord timeline with a
downloadable `.lab` annotation.

Built with **Vite 5 + React 18 + TypeScript + Tailwind CSS 3**, talking to the Flask
backend in `../backend`.

## Stack

| Concern | Choice |
| --- | --- |
| Build/dev server | Vite 5 |
| UI | React 18 + TypeScript |
| Styling | Tailwind CSS 3 |
| HTTP | Axios |
| File upload | react-dropzone |
| Chord timeline | Custom SVG/CSS timeline (see note) |
| Tests | Vitest + Testing Library + fast-check |

> **Timeline note:** the original design called for Recharts, but Recharts is a poor
> fit for Gantt-style chord *segments*. This uses a lightweight custom timeline
> (`VisualizationComponent`) that positions each segment by its start/end time and
> colors it by chord — satisfying the same design properties (12 positioning, 13
> distinctness) with no heavy charting dependency.

## Prerequisites

- Node.js 18+ (developed on Node 20)
- The backend running (see `../backend/README.md`)

## Setup & run

```bash
npm install
npm run dev          # dev server on http://localhost:5173
```

In dev, requests to `/api/*` are **proxied to the backend at `127.0.0.1:5000`**
(configured in `vite.config.ts`), so you don't need CORS or any extra config — just
start the backend and the frontend together.

To call a backend on a different origin instead of the proxy, copy `.env.example` to
`.env` and set `VITE_API_BASE_URL` (the backend already allows `:5173` via CORS):

```bash
cp .env.example .env   # VITE_API_BASE_URL=http://localhost:5000
```

## Scripts

| Command | Description |
| --- | --- |
| `npm run dev` | Start the Vite dev server (port 5173) |
| `npm run build` | Type-check (`tsc -b`) and build to `dist/` |
| `npm run preview` | Preview the production build |
| `npm run typecheck` | Type-check only |
| `npm test` | Run the Vitest suite once |
| `npm run test:watch` | Run Vitest in watch mode |

## Project structure

```
src/
├── api/client.ts            # Axios client: uploadFile, getJobStatus, getResults, downloadAnnotations
├── components/
│   ├── UploadComponent.tsx       # drag-and-drop + validation + upload progress
│   ├── StatusComponent.tsx       # polls /api/status, progress bar + ETA
│   ├── VisualizationComponent.tsx# scrollable chord timeline + legend
│   ├── ResultsComponent.tsx      # first-10 preview + .lab download
│   └── ErrorBoundary.tsx
├── utils/
│   ├── chordColors.ts       # deterministic chord -> color
│   └── format.ts            # time/byte formatting helpers
├── types/index.ts           # ChordAnnotation + API response types
├── constants.ts             # accepted formats, size limit, poll interval
├── App.tsx                  # idle -> processing -> complete/failed state machine
└── main.tsx
```

## App flow

1. **Upload** — drag/drop or browse; validates format (`.wav/.mp3/.flac`) and size
   (≤50 MB) client-side, then `POST /api/upload`.
2. **Status** — polls `GET /api/status/{jobId}` every 2s, showing progress + ETA
   until the job is `complete` or `failed` (with connection-loss retry).
3. **Results** — fetches `GET /api/results/{jobId}`, renders the chord timeline, a
   first-10 preview table, and a **Download .lab** button (`GET /api/download/{jobId}`).

## Testing

```bash
npm test
```

15 tests covering upload validation, status polling/terminal transitions, results
preview + download filename, and property-based checks for the timeline (segment
positioning and color distinctness) and the chord-color utility.
