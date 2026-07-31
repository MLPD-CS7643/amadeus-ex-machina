#!/usr/bin/env bash
#
# Set up the optional NNLS feature backend (true McGill "bothchroma" at serve time).
#
# This is intentionally NOT part of `uv sync`: it needs a native Vamp plugin binary and a
# source-built Python Vamp host that are not reliably pip/uv-installable (the `vamp` sdist
# imports numpy at build time and links the Vamp SDK). We keep the default install rock
# solid and make this an explicit opt-in. See docs/handoff-nnls-at-serve.md and
# packages/amadeus-core/src/amadeus_core/features/nnls.py.
#
# What it does (idempotent):
#   1. Build the Queen Mary `nnls-chroma` Vamp plugin from source for the host arch,
#      stripping the Chordino/boost path we don't use, and install the .dylib into the
#      user Vamp plugin directory.
#   2. Install the `vamp` Python host into the uv-managed workspace venv.
#
# Requirements: macOS with Xcode command line tools, Homebrew, and `uv`. Run from the repo
# root (or anywhere; paths are resolved relative to this script).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This script targets macOS. On other platforms, adapt the plugin build (see the" >&2
  echo "nnls-chroma Makefile.linux) and install the vamp host manually." >&2
  exit 1
fi

ARCH="$(uname -m)"
VAMP_PLUGIN_DIR="$HOME/Library/Audio/Plug-Ins/Vamp"

echo "==> [1/2] nnls-chroma Vamp plugin ($ARCH)"

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is required (https://brew.sh)." >&2
  exit 1
fi

# Vamp SDK provides headers + libvamp-sdk.a used to build the plugin and the host.
if ! brew --prefix vamp-plugin-sdk >/dev/null 2>&1; then
  echo "  installing vamp-plugin-sdk via brew..."
  brew install vamp-plugin-sdk
fi
VAMP_SDK_PREFIX="$(brew --prefix vamp-plugin-sdk)"
VAMP_INC="$VAMP_SDK_PREFIX/include"
VAMP_LIB="$VAMP_SDK_PREFIX/lib/libvamp-sdk.a"

BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT
echo "  cloning nnls-chroma source into $BUILD_DIR ..."
git clone --depth 1 https://github.com/c4dm/nnls-chroma.git "$BUILD_DIR/nnls-chroma" >/dev/null 2>&1
cd "$BUILD_DIR/nnls-chroma"

# We only need the NNLS Chroma feature plugin (bothchroma). Chordino's chord-dictionary
# parser pulls in boost; drop Chordino and neutralize that parser so the build has no boost
# dependency. This does not affect bothchroma output in any way.
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
start = next(i for i, l in enumerate(lines) if l.startswith("vector<string> chordDictionary("))
depth = 0
started = False
end = None
for i in range(start, len(lines)):
    depth += lines[i].count("{") - lines[i].count("}")
    if "{" in lines[i]:
        started = True
    if started and depth == 0:
        end = i
        break
assert end is not None
sig = lines[start].split("{")[0].strip()
lines[start:end + 1] = [
    sig + " {",
    "    // Chordino chord-dictionary parsing removed: this build ships only the",
    "    // NNLS-Chroma feature plugin (bothchroma), so this is never called.",
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
print("  patched out boost/Chordino")
PY

echo "  compiling..."
for src in chromamethods NNLSBase NNLSChroma Tuning plugins viterbi; do
  clang++ -arch "$ARCH" -O3 -ffast-math -fPIC -I"$VAMP_INC" -c "$src.cpp" -o "$src.o"
done
clang -arch "$ARCH" -O3 -ffast-math -fPIC -c nnls.c -o nnls.o

echo "  linking..."
clang++ -arch "$ARCH" -dynamiclib -install_name nnls-chroma.dylib \
  -o nnls-chroma.dylib \
  chromamethods.o NNLSBase.o NNLSChroma.o Tuning.o plugins.o nnls.o viterbi.o \
  "$VAMP_LIB" -exported_symbols_list vamp-plugin.list -framework Accelerate

mkdir -p "$VAMP_PLUGIN_DIR"
cp nnls-chroma.dylib "$VAMP_PLUGIN_DIR/"
cp nnls-chroma.cat nnls-chroma.n3 "$VAMP_PLUGIN_DIR/" 2>/dev/null || true
echo "  installed -> $VAMP_PLUGIN_DIR/nnls-chroma.dylib"

echo "==> [2/2] vamp Python host (into the uv workspace venv)"
cd "$REPO_ROOT"
# The vamp sdist builds a C extension (vampyhost) against the Vamp SDK and imports numpy at
# build time, so build without isolation using the venv (which already has numpy) plus
# setuptools/wheel, with the SDK on the include/library path.
export CPLUS_INCLUDE_PATH="$VAMP_INC${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export CPATH="$VAMP_INC${CPATH:+:$CPATH}"
export LIBRARY_PATH="$VAMP_SDK_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
uv pip install setuptools wheel >/dev/null
uv pip install --no-build-isolation "vamp==1.1.0"

echo "==> verifying"
uv run python - <<'PY'
import vampyhost
plugins = vampyhost.list_plugins()
assert "nnls-chroma:nnls-chroma" in plugins, plugins
print("  OK:", plugins)
PY

echo
echo "Done. The NNLS backend is ready. Serve the Billboard model with the FEATURE_BACKEND"
echo "left at its default (the bundle's model_config.json selects 'nnls' automatically)."
