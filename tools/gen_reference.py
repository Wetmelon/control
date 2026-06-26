#!/usr/bin/env python3
"""Scrape inc/wet headers for public types/functions + @brief into a categorized
markdown reference. Gate on column-0 doc comments so members/converters are skipped."""
import re, pathlib, collections

ROOT = pathlib.Path("inc/wet")

# Pretty names + ordering for top-level categories (dir under inc/wet, or "" for umbrellas).
CATS = [
    ("",            "Core, configuration & backend vocabulary"),
    ("math",        "Scalar math & complex"),
    ("matrix",      "Linear algebra"),
    ("systems",     "LTI models"),
    ("controllers", "Runtime controllers"),
    ("design",      "Design & synthesis"),
    ("estimation",  "Observers & estimators"),
    ("filters",     "Filters & signal conditioning"),
    ("trajectory",  "Trajectory & motion planning"),
    ("kinematics",  "Kinematics"),
    ("power",       "Motor control"),
    ("toolbox",     "Utilities & toolbox"),
    ("analysis",    "Frequency-domain analysis (host)"),
    ("simulation",  "Simulation (host)"),
    ("matlab",      "MATLAB-style aliases (host)"),
]
CAT_ORDER = {k: i for i, (k, _) in enumerate(CATS)}
CAT_NAME = dict(CATS)

# /** ... */ block that starts at column 0, plus the code that follows it.
BLOCK = re.compile(r"^/\*\*(.*?)\*/\n(.*?)(?=\n/\*\*|\n\}|\Z)", re.S | re.M)
BRIEF = re.compile(r"@brief\s+(.*?)(?:\n\s*\*\s*@|\n\s*\*\s*\n|\*/)", re.S)
SKIP_PREFIX = ("template", "[[", "//", "requires", "constexpr", "explicit",
               "inline", "static", "friend")

# Internal, compile-time-selected math backends. Each re-declares the same
# wet:: scalar surface (sin/cos/sqrt/...), so listing their functions is pure
# duplication — they get one file-level section instead. math.hpp (the public
# dispatcher) and complex.hpp stay in the scalar-math tables.
BACKEND_FILES = {"trig.hpp", "math_backend.hpp", "wet_backend.hpp",
                 "series_backend.hpp", "std_fallback.hpp", "constexpr_math.hpp"}


def brief_text(body):
    m = BRIEF.search(body)
    if not m:
        return None
    t = re.sub(r"\s*\*\s*", " ", m.group(1)).strip().rstrip(".")
    t = re.sub(r"@(?:ref|c|p|a)\s+", "", t)        # drop Doxygen inline tags
    return t.replace("|", "\\|")                     # don't break the md table


def decl(code):
    """Return (kind, name) for the declaration following a doc block, or None."""
    lines = [l for l in code.split("\n")]
    # skip blank/attribute/template prefix lines to reach the named declaration
    buf = []
    for l in lines:
        s = l.strip()
        if not s:
            continue
        if s.startswith("#") or s.startswith("namespace") or s.startswith("using"):
            return None  # file-level / namespace brief, not a public entity
        buf.append(s)
        joined = " ".join(buf)
        m = re.search(r"\b(struct|class)\s+(\w+)", joined)
        if m and "{" in joined or (m and ";" not in joined and len(buf) <= 3):
            return ("type", m.group(2))
        m = re.search(r"\benum\s+(?:class\s+)?(\w+)", joined)
        if m:
            return ("enum", m.group(1))
        # function: identifier immediately before '(' once we've cleared prefixes
        if "(" in joined and not s.startswith(SKIP_PREFIX[:1]):
            fm = re.search(r"(\w+)\s*\(", joined)
            if fm and fm.group(1) not in ("if", "for", "while", "return", "as"):
                # require it looks like a free function decl (has a return-type token)
                if re.search(r"\b(struct|class|enum)\b", joined):
                    continue
                return ("fn", fm.group(1))
        if len(buf) > 4:
            break
    return None


def category(path):
    if path.name == "matlab.hpp":
        return "matlab"  # all thin wrappers — keep them out of the core tables
    rel = path.relative_to(ROOT)
    top = rel.parts[0] if len(rel.parts) > 1 else ""
    return top if top in CAT_NAME else "toolbox"


entries = collections.defaultdict(list)  # cat -> list[(kind,name,brief,header)]
seen = collections.defaultdict(set)

for hpp in sorted(ROOT.rglob("*.hpp")):
    if hpp.name in BACKEND_FILES:
        continue  # internal math backends — covered by their own section below
    text = hpp.read_text(encoding="utf-8", errors="replace")
    cat = category(hpp)
    header = str(hpp.relative_to("inc")).replace("\\", "/")
    for m in BLOCK.finditer(text):
        b = brief_text(m.group(1))
        if not b:
            continue
        d = decl(m.group(2))
        if not d:
            continue
        kind, name = d
        if name in seen[cat]:
            continue
        seen[cat].add(name)
        line = text.count("\n", 0, m.start()) + 1  # line of the @brief comment
        entries[cat].append((kind, name, b, f"inc/{header}#L{line}"))

out = ["# API Reference\n",
       "Auto-generated from `@brief` doc comments in `inc/wet/`. "
       "Regenerate with `python tools/gen_reference.py`.\n"]


def table(title, rows):
    if not rows:
        return
    out.append(f"\n**{title}**\n")
    out.append("| Name | Description |")
    out.append("| ---- | ----------- |")
    for _, name, b, link in sorted(rows, key=lambda r: r[1].lower()):
        out.append(f"| [`{name}`]({link}) | {b} |")


for key, _ in CATS:
    cat = entries.get(key)
    if not cat:
        continue
    out.append(f"\n## {CAT_NAME[key]}")
    table("Blocks (structs, classes, enums)", [r for r in cat if r[0] != "fn"])
    table("Functions", [r for r in cat if r[0] == "fn"])

# --- Math backends --------------------------------------------------------
out.append("\n## Math backends")
out.append("\nInternal, compile-time-selected implementations of the `wet::` scalar-math "
           "surface (`sin`/`cos`/`sqrt`/`exp`/…), chosen via `wet/config.hpp`. Every "
           "backend exposes the same functions, so they're listed once here as files "
           "rather than repeated in the tables above. The public dispatcher is "
           "[`wet/math/math.hpp`](inc/wet/math/math.hpp).\n")
out.append("| File | Role |")
out.append("| ---- | ---- |")
for name in ("math_backend.hpp", "std_fallback.hpp", "wet_backend.hpp",
             "series_backend.hpp", "constexpr_math.hpp", "trig.hpp"):
    f = ROOT / "math" / name
    role = brief_text(f.read_text(encoding="utf-8", errors="replace")[:1500])
    if not role:
        role = f"`{name}` math backend"
    out.append(f"| [`wet/math/{name}`](inc/wet/math/{name}) | {role} |")

# --- Examples -------------------------------------------------------------
ex_rows = []
for cpp in sorted(pathlib.Path("examples").rglob("*.cpp")):
    head = cpp.read_text(encoding="utf-8", errors="replace")[:1500]
    m = BRIEF.search(head)
    desc = brief_text(head) if m else None
    if not desc:  # no @brief: humanize the filename
        stem = cpp.stem.removeprefix("example_").replace("_", " ")
        desc = stem[:1].upper() + stem[1:]
    rel = str(cpp).replace("\\", "/")
    ex_rows.append((cpp.name, rel, desc))

out.append("\n## Examples")
out.append(f"\nRunnable programs in `examples/` ({len(ex_rows)} total). "
           "Build with `make` (or `tup --quiet examples`); outputs go to `examples/build/`.\n")
out.append("| Example | Description |")
out.append("| ------- | ----------- |")
for name, rel, desc in sorted(ex_rows, key=lambda r: r[0].lower()):
    out.append(f"| [`{name}`]({rel}) | {desc} |")

# --- Table of contents (GitHub-slug anchors) ------------------------------
def slug(title):
    s = re.sub(r"[^\w\s-]", "", title.lower())  # drop punctuation, keep spaces/hyphens
    return s.replace(" ", "-")                   # per-space so '&' leaves a double hyphen


toc = ["- [API Reference](#api-reference)"]
for line in out:
    s = line.lstrip()
    if s.startswith("## "):
        t = s[3:].strip()
        toc.append(f"  - [{t.replace('&', chr(92) + '&')}](#{slug(t)})")
out[2:2] = ["\n" + "\n".join(toc)]  # splice after the H1 + intro paragraph

pathlib.Path("REFERENCE.md").write_text("\n".join(out) + "\n", encoding="utf-8")
total = sum(len(v) for v in entries.values())
print(f"Wrote REFERENCE.md: {total} entries across {len(entries)} categories")
for key, _ in CATS:
    if entries.get(key):
        print(f"  {CAT_NAME[key]:40s} {len(entries[key])}")
