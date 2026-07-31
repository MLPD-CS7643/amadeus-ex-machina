# Deploying amadeus-ex-machina

The app splits cleanly into two deployables:

- **Frontend** — a static React/Vite SPA. Host it on a static/CDN service (**AWS Amplify
  Hosting**, or S3 + CloudFront). Pennies/month.
- **Backend** — the FastAPI inference service, shipped as the container built by the
  repo-root `Dockerfile` (CPU PyTorch + the native NNLS Vamp plugin + both model bundles).

> **Read this first — the backend is single-instance.** Job state lives in-process
> (`JobManager`/`JobWorker` in `main.py`) with uploads/results on local disk under `/tmp`.
> A job created on one process can only be polled/downloaded from *that same* process.
> So: run **one container, one uvicorn worker**. Do **not** put multiple replicas behind a
> load balancer (or set `--workers >1`) without first externalizing state — see
> [Scaling out](#scaling-out-later). For a portfolio/hobby deployment, one instance is
> perfectly fine.

---

## Backend container

### Build

```bash
docker build -t amadeus-api .
```

Notes:
- The model binaries (`model.pt`, `*.pkl`) are **gitignored**, so build from a checkout
  that actually has `artifacts/billboard_sevenths/` and `artifacts/guitarset_chroma/`
  populated (this machine does). They're COPYed into the image.
- **CPU-only torch**: the image installs `torch==2.5.1+cpu` from the PyTorch CPU index, so
  there's no ~2.5 GB CUDA payload. Works on both `linux/amd64` and `linux/arm64`.
- **arm64 is cheaper** (Graviton / Lightsail ARM / `t4g`). To build for it explicitly:
  ```bash
  docker buildx build --platform linux/arm64 -t amadeus-api:arm64 .
  ```
- The build compiles the NNLS Chroma Vamp plugin from source and **fails fast** if the
  plugin doesn't load, so a green build means the NNLS path works.

### Run

```bash
docker run -p 8000:8000 \
  -e CORS_ORIGINS=https://your-frontend.example \
  amadeus-api
```

Health check: `curl localhost:8000/api/health` → `{"status":"ok","engineLoaded":true}`.

### Configuration (env vars)

| Var | Default (baked) | Notes |
|---|---|---|
| `CORS_ORIGINS` | `http://localhost:5173` | **Set to your deployed frontend origin(s)**, comma-separated. |
| `PORT` / `HOST` | `8000` / `0.0.0.0` | |
| `MODEL_PATH`, `SCALER_PATH`, `ENCODER_PATH`, `MODEL_CONFIG_PATH` | Billboard bundle | Primary model. |
| `FALLBACK_*` | GuitarSet bundle | Served automatically if the NNLS backend is ever unavailable. |
| `SEPARATION_METHOD` | `hpss` | Ignored for the NNLS model (engine forces it off). |
| `FEATURE_BACKEND` | *(unset)* | Leave unset — the bundle self-selects `nnls`. |
| `MAX_FILE_SIZE`, `RATE_LIMIT_PER_MIN`, `MAX_CONCURRENT_JOBS` | see `config.py` | |

The image runs as a non-root user and exposes a Docker `HEALTHCHECK` on `/api/health`.

---

## Frontend (Amplify)

The SPA calls the API at `VITE_API_BASE_URL` (falling back to same-origin `/api`, which is
how the dev proxy works). For a split deploy, **build it pointed at the backend URL**:

```bash
cd frontend
VITE_API_BASE_URL=https://your-backend.example npm run build   # outputs dist/
```

Then deploy `frontend/dist/` as a static site:

- **Amplify Hosting**: connect the repo (or drag-drop `dist/`). Set the build to
  `npm ci && npm run build`, output dir `dist/`, and add `VITE_API_BASE_URL` as a build
  env var. Amplify serves it over HTTPS on a `*.amplifyapp.com` domain (or your custom
  domain).
- **S3 + CloudFront**: `aws s3 sync dist/ s3://your-bucket`, front it with CloudFront for
  HTTPS + caching. Add an SPA rewrite (404/403 → `/index.html`).

Finally, set the backend's `CORS_ORIGINS` to the frontend's final URL.

> Alternative (no rebuild-per-URL): leave `VITE_API_BASE_URL` unset and configure an
> Amplify/CloudFront **rewrite** of `/api/*` to the backend origin, so the SPA and API
> share an origin. Then CORS isn't even needed.

---

## Where to run the backend (cheap → managed)

All of these run the single container above. Pick by budget/effort:

1. **AWS Lightsail (recommended cheap + simple).** A 2–4 GB instance (~$12–24/mo, data
   transfer included) running Docker, or Lightsail Containers. Predictable flat price, no
   ALB. Best fit for the single-instance design. Put Caddy/nginx in front (or use Lightsail
   Containers' built-in HTTPS) for TLS.
2. **AWS App Runner (managed, no servers).** Point it at the image (ECR) or repo; HTTPS and
   autoscaling are built in, **no ALB to pay for**. Give it ≥ 3 GB. Note App Runner is
   `x86_64` only, and it may run >1 instance under load — pin **max concurrency = 1
   instance** (or accept the single-instance caveat) until state is externalized.
3. **ECS Fargate + ALB.** Works, but the always-on task + mandatory ALB is a ~$40–55/mo
   floor — overkill for one bursty instance. Only worth it once you actually need
   orchestration/autoscaling (and by then, do [Scaling out](#scaling-out-later) first).
4. **GCP Cloud Run** (if you're open to leaving AWS). True scale-to-zero, pay-per-request,
   generous free tier — genuinely the cheapest for this bursty, mostly-idle workload. Set
   min instances 0; accept a few-second cold start while torch imports.

Rough monthly cost, low traffic: frontend ~$1–5, backend ~$12–24 (Lightsail) / ~$5–25
(App Runner, Cloud Run) / ~$40–55 (ECS+ALB).

---

## Scaling out later

To run more than one backend replica, remove the in-process/local-disk state:

- **Uploads + result `.lab`s → S3** (replace the local `UPLOAD_DIR`/`RESULTS_DIR` I/O in
  `files.py`/`worker.py`).
- **Job queue → SQS**, with the inference worker as a separate consumer service/deployment
  (decouples the fat torch worker from the web tier; lets each scale independently).
- **Job status → DynamoDB or Redis** (replace the in-memory `JobManager` dict).

With those three, the web tier becomes stateless and you can run N replicas behind an ALB
(or Fargate service) safely. Until then: one instance, one worker.

---

## Gotchas specific to this app

- **`--workers 1`.** Multiple uvicorn workers = multiple in-memory job stores → "job not
  found" on poll. The Dockerfile pins one worker.
- **Results are ephemeral.** They live under `/tmp` inside the container and vanish on
  restart. Fine for "upload → view/download → leave"; mount a volume or use S3 if you need
  persistence.
- **The Vamp plugin is baked into the image** (built for Linux). No `just nnls-setup`
  needed in the container. If you ever strip it, the app self-degrades to the librosa
  GuitarSet model instead of failing.
- **Image size** is dominated by torch + librosa (~1–1.5 GB even CPU-only). For a leaner
  image, split the Dockerfile into a builder stage (compiler, git, `vamp-plugin-sdk` dev)
  and a slim runtime stage that copies `/opt/venv` + `/usr/local/lib/vamp` and installs
  only the runtime shared libs. Not necessary to ship.
- **No GPU.** Inference is CPU-bound and fast (a 75 s song is a few seconds); don't pay for
  GPU instances.
