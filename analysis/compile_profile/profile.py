#!/usr/bin/env python3
"""Compile-time performance profiler for the control library.

Compiles translation units with clang's ``-ftime-trace`` and emits, per TU, a
Chrome-trace JSON that opens directly as a *flame graph* in any of:

  * https://ui.perfetto.dev      (drag-and-drop the .json)
  * chrome://tracing             ("Load" the .json)
  * https://www.speedscope.app   (import the .json)

It then prints a textual summary that attributes the wall time to frontend
phases (header parsing, template instantiation) vs the optimizer/backend, and
lists the heaviest individual headers and template instantiations. That answers
the practical question -- "is this TU slow because of a heavy third-party header
(e.g. plotly/fmt) or because of our own template-heavy code?".

Why clang and not the build compiler? The project builds with GCC 14.2 (xPack
MinGW), which predates GCC's ``-ftime-trace`` (that landed in GCC 15). Clang 21
is already on the box for clang-tidy/clang-format and emits the trace GCC can't.
Clang defaults to an MSVC target on Windows, so -- exactly like the Makefile
``tidy`` target -- we hand it GCC's target triple and system include paths so it
finds <cmath> et al. from the same libstdc++ the real build uses.

Usage:
    python analysis/compile_profile/profile.py                # all examples
    python analysis/compile_profile/profile.py examples/example_kinematic_maps.cpp
    python analysis/compile_profile/profile.py --summary-only  # skip the JSON note
    python analysis/compile_profile/profile.py --open          # open worst TU in Perfetto

Traces are written to analysis/compile_profile/traces/<name>.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import webbrowser
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TRACE_DIR = Path(__file__).resolve().parent / "traces"

# Include paths mirror examples/Tupfile.lua (the hosted demo profile).
INCLUDES = [
    "-I" + str(REPO / "examples"),
    "-I" + str(REPO / "inc"),
    "-I" + str(REPO / "libs"),
    "-I" + str(REPO / "libs/fmt/include"),
    "-I" + str(REPO / "libs/plotlypp/include"),
    "-I" + str(REPO / "libs/json/single_include"),
]

# Mirror Tuprules.lua CXXFLAGS so backend/optimizer time reflects the real build.
CXXFLAGS = ["-O3", "-std=c++20", "-march=native", "-ffunction-sections", "-fdata-sections"]


def gxx_path() -> str:
    """Read the GCC path/prefix from tup.config (same source the Makefile uses)."""
    cfg = {}
    for line in (REPO / "tup.config").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            cfg[k.strip()] = v.strip()
    path = cfg.get("CONFIG_COMPILER_PATH", "")
    prefix = cfg.get("CONFIG_COMPILER_PREFIX", "")
    return f"{path}/{prefix}g++"


def gcc_toolchain_args(gxx: str) -> list[str]:
    """Target triple + libstdc++ system include dirs so clang matches the build."""
    triple = subprocess.run(
        [gxx, "-dumpmachine"], capture_output=True, text=True, check=True
    ).stdout.strip()
    # `g++ -E -v -` prints the system include search list to stderr.
    proc = subprocess.run(
        [gxx, "-std=c++20", "-E", "-x", "c++", "-v", "-"],
        input="", capture_output=True, text=True,
    )
    isys, collecting = [], False
    for line in proc.stderr.splitlines():
        if "search starts here:" in line:
            collecting = True
            continue
        if "End of search list" in line:
            break
        if collecting and line.startswith(" "):
            isys += ["-isystem", line.strip()]
    return [f"--target={triple}", *isys]


def compile_tu(src: Path, tc_args: list[str]) -> tuple[Path, float]:
    """Compile one TU with -ftime-trace; return (trace_json_path, wall_seconds)."""
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    obj = TRACE_DIR / (src.stem + ".o")
    trace = TRACE_DIR / (src.stem + ".json")  # clang writes <obj-basename>.json beside -o
    cmd = [
        "clang++", *tc_args, *CXXFLAGS, *INCLUDES,
        "-ftime-trace", "-ftime-trace-granularity=200",
        "-c", str(src), "-o", str(obj),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        sys.stderr.write(f"\n[FAILED] {src.name}\n{proc.stderr}\n")
        return trace, wall
    obj.unlink(missing_ok=True)  # keep only the trace; the .o is throwaway
    return trace, wall


def summarize(trace: Path) -> dict:
    """Pull phase totals + heaviest headers/instantiations from a time-trace JSON."""
    events = json.loads(trace.read_text())["traceEvents"]
    totals: dict[str, float] = {}
    headers: dict[str, float] = defaultdict(float)
    insts: dict[str, float] = defaultdict(float)
    # "Source" (header) events come as nested begin/end async pairs without a
    # duration; reconstruct inclusive parse time per header via a stack.
    src_stack: list[tuple[str, int]] = []
    for e in events:
        ph, name = e.get("ph"), e.get("name", "")
        if name == "Source" and ph == "b":
            src_stack.append((e.get("args", {}).get("detail", ""), e.get("ts", 0)))
        elif name == "Source" and ph == "e" and src_stack:
            detail, t0 = src_stack.pop()
            headers[Path(detail).name] += (e.get("ts", 0) - t0) / 1000.0
        elif name.startswith("Total "):
            # clang emits one summary counter per phase (self-time, deduped).
            totals[name[len("Total "):]] = e.get("dur", 0) / 1000.0
        elif name in ("InstantiateClass", "InstantiateFunction"):
            insts[e.get("args", {}).get("detail", "?")] += e.get("dur", 0) / 1000.0
    return {"totals": totals, "headers": headers, "insts": insts}


def bucket(path_name: str) -> str:
    """Classify a header into a coarse owner for the plotly-vs-library question."""
    return path_name  # name only; origin shown via the top-headers list


def fmt_ms(ms: float) -> str:
    return f"{ms/1000:6.2f}s" if ms >= 1000 else f"{ms:6.0f}ms"


def print_report(src: Path, wall: float, summary: dict) -> None:
    t = summary["totals"]
    print(f"\n=== {src.name}  (wall {wall:5.2f}s) ===")
    frontend = t.get("Frontend", 0)
    backend = t.get("Backend", 0)
    optimizer = t.get("Optimizer", 0)
    print(f"  Frontend (parse+instantiate) : {fmt_ms(frontend)}")
    print(f"  Backend (codegen)            : {fmt_ms(backend)}")
    print(f"  Optimizer                    : {fmt_ms(optimizer)}")
    if t.get("Source"):
        print(f"  -- of frontend, header parse : {fmt_ms(t['Source'])}")
    inst_total = t.get("InstantiateClass", 0) + t.get("InstantiateFunction", 0)
    if inst_total:
        print(f"  -- of frontend, template inst: {fmt_ms(inst_total)}")

    top_h = sorted(summary["headers"].items(), key=lambda kv: -kv[1])[:8]
    if top_h:
        print("  Heaviest headers (inclusive parse time):")
        for name, ms in top_h:
            print(f"      {fmt_ms(ms)}  {name}")
    top_i = sorted(summary["insts"].items(), key=lambda kv: -kv[1])[:6]
    if top_i:
        print("  Heaviest template instantiations:")
        for name, ms in top_i:
            print(f"      {fmt_ms(ms)}  {name[:70]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="*", help="source files (default: examples/*.cpp)")
    ap.add_argument("--open", action="store_true",
                    help="open the slowest TU's trace in Perfetto after profiling")
    args = ap.parse_args()

    sources = [Path(s) for s in args.sources] or sorted((REPO / "examples").glob("*.cpp"))
    sources = [s if s.is_absolute() else (REPO / s) for s in sources]

    gxx = gxx_path()
    print(f"Build compiler : {gxx}")
    tc_args = gcc_toolchain_args(gxx)
    print(f"clang target   : {tc_args[0]}")
    print(f"Profiling {len(sources)} translation unit(s)...")

    results = []
    for src in sources:
        trace, wall = compile_tu(src, tc_args)
        if trace.exists():
            results.append((src, wall, trace, summarize(trace)))

    results.sort(key=lambda r: -r[1])
    for src, wall, _trace, summary in results:
        print_report(src, wall, summary)

    print("\n" + "=" * 60)
    print("Flame graphs (open in https://ui.perfetto.dev or chrome://tracing):")
    for src, wall, trace, _ in results:
        print(f"  {wall:5.2f}s  {trace.relative_to(REPO)}")

    if args.open and results:
        worst = results[0][2]
        print(f"\nOpening {worst.name} in Perfetto...")
        webbrowser.open("https://ui.perfetto.dev")
        print(f"(Drag-and-drop {worst} into the Perfetto tab.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
