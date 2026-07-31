#!/usr/bin/env python3
"""
Strip the boost/Chordino path from a nnls-chroma source checkout, in place.

We ship only the NNLS Chroma feature plugin (``bothchroma``). Chordino's chord-dictionary
parser is the only thing that pulls in boost, so drop Chordino and neutralize that parser:
the build then needs no boost and ``bothchroma`` output is unchanged. Run with the current
directory at the root of a nnls-chroma checkout (both callers ``cd`` there first):

    python3 patch_nnls_chroma.py

Shared by scripts/setup_nnls_backend.sh (macOS .dylib) and the Dockerfile (Linux .so) so
the patch can't drift between the two plugin builds.
"""

# Drop the boost includes + `using namespace boost`, and replace chordDictionary()'s body
# with a stub -- it is unreachable once Chordino is gone.
p = "chromamethods.cpp"
with open(p) as f:
    s = f.read()
for inc in (
    "#include <boost/tokenizer.hpp>\n",
    "#include <boost/iostreams/device/file.hpp>\n",
    "#include <boost/iostreams/stream.hpp>\n",
    "#include <boost/lexical_cast.hpp>\n",
):
    s = s.replace(inc, "")
s = s.replace("using namespace boost;\n", "")
lines = s.split("\n")
start = next(
    i for i, line in enumerate(lines)
    if line.startswith("vector<string> chordDictionary(")
)
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
lines[start : end + 1] = [
    sig + " {",
    "    // Chordino chord-dictionary parsing removed: this build ships only the",
    "    // NNLS-Chroma feature plugin (bothchroma), so this is never called.",
    "    (void)mchorddict; (void)m_chordnotes; (void)boostN; (void)harte_syntax;",
    "    return std::vector<std::string>();",
    "}",
]
with open(p, "w") as f:
    f.write("\n".join(lines))

# Remove the Chordino plugin from the Vamp adapter table.
p = "plugins.cpp"
with open(p) as f:
    s = f.read()
s = s.replace('#include "Chordino.h"\n', "")
s = s.replace("static Vamp::PluginAdapter<Chordino> chordinoAdapter;\n", "")
s = s.replace(
    "    case  0: return chromaAdapter.getDescriptor();\n"
    "    case  1: return chordinoAdapter.getDescriptor();\n"
    "    case  2: return tuningAdapter.getDescriptor();\n",
    "    case  0: return chromaAdapter.getDescriptor();\n"
    "    case  1: return tuningAdapter.getDescriptor();\n",
)
with open(p, "w") as f:
    f.write(s)
print("patched out boost/Chordino")
