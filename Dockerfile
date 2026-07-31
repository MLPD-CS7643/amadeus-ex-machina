# syntax=docker/dockerfile:1
#
# Production image for the amadeus chord-recognition API (FastAPI backend).
#
# What's inside:
#   * CPU-only PyTorch (from the PyTorch CPU wheel index) -- no CUDA, no GPU. Inference of
#     a ~75s song is a few seconds on CPU.
#   * The native Queen Mary NNLS Chroma Vamp plugin, built from source for Linux (only the
#     `bothchroma` feature plugin; the Chordino/boost path is stripped), plus the `vamp`
#     Python host. This is what lets the app serve the Billboard model on true bothchroma.
#   * Both model bundles baked in: artifacts/billboard_sevenths (primary, NNLS, 7th chords)
#     and artifacts/guitarset_chroma (librosa fallback).
#
# The React frontend is a static SPA deployed separately (e.g. AWS Amplify); this image is
# backend-only. See docs/deploy.md for the full deploy story and the single-instance caveat.
#
#   docker build -t amadeus-api .
#   docker run -p 8000:8000 -e CORS_ORIGINS=https://your-frontend.example amadeus-api

FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    VAMP_PATH=/usr/local/lib/vamp \
    OMP_NUM_THREADS=2

# System deps:
#   build-essential/git  -> compile the nnls-chroma plugin + the vamp host C extension
#   vamp-plugin-sdk      -> Vamp SDK headers + libs (build the plugin AND run the host)
#   libsndfile1          -> soundfile/librosa audio I/O (wav/flac/mp3 via libsndfile 1.2)
#   libgomp1             -> OpenMP runtime for numpy/scipy/librosa
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ca-certificates \
        vamp-plugin-sdk libsndfile1 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# --- Build + install the nnls-chroma Vamp plugin (Linux .so) ------------------------
# Ship only the NNLS Chroma feature plugin (bothchroma); Chordino's chord-dictionary path
# pulls in boost, so strip it -- identical to scripts/setup_nnls_backend.sh on macOS. No
# BLAS/LAPACK needed (nnls.c is self-contained; the Vamp SDK supplies the FFT).
RUN <<'BUILD'
set -eux
git clone --depth 1 https://github.com/c4dm/nnls-chroma.git /tmp/nnls
cd /tmp/nnls
python3 - <<'PY'
p = "chromamethods.cpp"
s = open(p).read()
for inc in (
    "#include <boost/tokenizer.hpp>\n",
    "#include <boost/iostreams/device/file.hpp>\n",
    "#include <boost/iostreams/stream.hpp>\n",
    "#include <boost/lexical_cast.hpp>\n",
):
    s = s.replace(inc, "")
s = s.replace("using namespace boost;\n", "")
lines = s.split("\n")
i = next(k for k, l in enumerate(lines) if l.startswith("vector<string> chordDictionary("))
depth, started, end = 0, False, None
for k in range(i, len(lines)):
    depth += lines[k].count("{") - lines[k].count("}")
    started = started or "{" in lines[k]
    if started and depth == 0:
        end = k
        break
assert end is not None
sig = lines[i].split("{")[0].strip()
lines[i:end + 1] = [
    sig + " {",
    "    (void)mchorddict; (void)m_chordnotes; (void)boostN; (void)harte_syntax;",
    "    return std::vector<std::string>();",
    "}",
]
open(p, "w").write("\n".join(lines))

p = "plugins.cpp"
s = open(p).read()
s = s.replace('#include "Chordino.h"\n', "")
s = s.replace("static Vamp::PluginAdapter<Chordino> chordinoAdapter;\n", "")
s = s.replace(
    "    case  0: return chromaAdapter.getDescriptor();\n"
    "    case  1: return chordinoAdapter.getDescriptor();\n"
    "    case  2: return tuningAdapter.getDescriptor();\n",
    "    case  0: return chromaAdapter.getDescriptor();\n"
    "    case  1: return tuningAdapter.getDescriptor();\n",
)
open(p, "w").write(s)
PY
for src in chromamethods NNLSBase NNLSChroma Tuning plugins viterbi; do
    g++ -O3 -ffast-math -fPIC -c "$src.cpp" -o "$src.o"
done
gcc -O3 -ffast-math -fPIC -c nnls.c -o nnls.o
g++ -shared -Wl,-soname=nnls-chroma.so -Wl,--version-script=vamp-plugin.map \
    -o nnls-chroma.so \
    chromamethods.o NNLSBase.o NNLSChroma.o Tuning.o plugins.o viterbi.o nnls.o \
    -lvamp-sdk
mkdir -p /usr/local/lib/vamp
cp nnls-chroma.so /usr/local/lib/vamp/
cp nnls-chroma.cat nnls-chroma.n3 /usr/local/lib/vamp/ 2>/dev/null || true
rm -rf /tmp/nnls
BUILD

# --- Python environment ------------------------------------------------------------
RUN python -m venv "$VIRTUAL_ENV"

# CPU-only torch FIRST (2.5.1+cpu satisfies the pinned torch==2.5.1 without the ~2.5GB CUDA
# build), so the workspace install below reuses it instead of pulling the GPU wheels.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1 torchaudio==2.5.1

# The workspace packages + their remaining deps (single source of truth = their pyproject).
COPY packages/ /app/packages/
RUN pip install /app/packages/amadeus-core /app/packages/amadeus-api setuptools wheel

# The vamp Python host (compiles vampyhost against the Vamp SDK headers from apt; needs
# numpy + setuptools + wheel present, which the steps above provide).
RUN pip install --no-build-isolation "vamp==1.1.0"

# Fail the build now if the plugin didn't build/install correctly (rather than at runtime).
RUN python -c "import vampyhost; ps=vampyhost.list_plugins(); assert 'nnls-chroma:nnls-chroma' in ps, ps; print('NNLS plugin OK:', ps)"

# --- Model artifacts + runtime config ----------------------------------------------
WORKDIR /app
COPY artifacts/billboard_sevenths/ /app/artifacts/billboard_sevenths/
COPY artifacts/guitarset_chroma/   /app/artifacts/guitarset_chroma/

# Explicit artifact paths (so serving doesn't depend on repo-relative path resolution).
# Primary = Billboard NNLS model; fallback = librosa GuitarSet model (used automatically
# if the NNLS backend is ever unavailable).
ENV MODEL_PATH=/app/artifacts/billboard_sevenths/model.pt \
    SCALER_PATH=/app/artifacts/billboard_sevenths/scaler.pkl \
    ENCODER_PATH=/app/artifacts/billboard_sevenths/label_encoder.pkl \
    MODEL_CONFIG_PATH=/app/artifacts/billboard_sevenths/model_config.json \
    FALLBACK_MODEL_PATH=/app/artifacts/guitarset_chroma/model.pt \
    FALLBACK_SCALER_PATH=/app/artifacts/guitarset_chroma/scaler.pkl \
    FALLBACK_ENCODER_PATH=/app/artifacts/guitarset_chroma/label_encoder.pkl \
    FALLBACK_MODEL_CONFIG_PATH=/app/artifacts/guitarset_chroma/model_config.json \
    HOST=0.0.0.0 \
    PORT=8000
# Set to your deployed frontend origin(s), comma-separated, at run time, e.g.
#   -e CORS_ORIGINS=https://main.xxxx.amplifyapp.com
ENV CORS_ORIGINS=http://localhost:5173

# Run as non-root. Uploads/jobs/results default under /tmp (world-writable), so no chown.
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status==200 else 1)"

# One worker only: the job store is in-process + on local disk (see docs/deploy.md).
CMD ["uvicorn", "amadeus_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
