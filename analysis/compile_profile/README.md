# Compile-time profiling

Generates **flame graphs of compilation CPU usage** for the library's
translation units, to find what makes a given file slow to compile.

## Run

```sh
make profile-compile                                    # all examples
make profile-compile FILES=examples/example_lpf.cpp     # one file
python analysis/compile_profile/profile.py --open       # + open worst in Perfetto
```

Per-TU traces are written to `traces/<name>.json`.

## View the flame graph

Open a `traces/*.json` in any Chrome-trace viewer:

- **Perfetto** — <https://ui.perfetto.dev> (drag-and-drop the file)
- **chrome://tracing** — "Load"
- **speedscope** — <https://www.speedscope.app> (import)

## How it works

The build compiler is GCC 14.2 (xPack MinGW), which predates GCC's
`-ftime-trace` (added in GCC 15). The script instead drives **clang**
(already on PATH for `clang-tidy`/`clang-format`) with `-ftime-trace`, which
emits the Chrome-trace JSON. Clang defaults to an MSVC target on Windows, so —
exactly like the `tidy` Makefile target — the script feeds it GCC's target
triple and `-isystem` paths so it parses the same libstdc++ the real build uses.
Compile flags mirror `Tuprules.lua` (`-O3 -std=c++20 -march=native`) so the
optimizer/backend time reflects the real build.

The textual summary splits each TU into **Frontend** (header parsing + template
instantiation) vs **Backend/Optimizer** (codegen), and lists the heaviest
headers and template instantiations — enough to tell at a glance whether a slow
TU is dominated by a third-party header or by the library's own templates.
