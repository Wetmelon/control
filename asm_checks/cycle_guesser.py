import re
import sys
from collections import defaultdict

# Approximate Cortex-M7 (Thumb-2 + FPv5-D16 hardware FPU) cycle costs, keyed by the
# *base* mnemonic (suffixes like .f32/.w/.n and condition codes are stripped first).
# These are rough throughput numbers: the M7 is a dual-issue superscalar pipeline, so a
# real count needs the hardware. This is a guess, not a model.
CYCLE_LOOKUP = {
    # Integer ALU / move / shift / bitfield — 1 cycle
    "add": 1, "adc": 1, "sub": 1, "sbc": 1, "rsb": 1, "adr": 1,
    "mov": 1, "movw": 1, "movt": 1, "mvn": 1, "neg": 1,
    "cmp": 1, "cmn": 1, "tst": 1, "teq": 1,
    "and": 1, "orr": 1, "orn": 1, "eor": 1, "bic": 1,
    "lsl": 1, "lsr": 1, "asr": 1, "ror": 1, "rrx": 1,
    "clz": 1, "rev": 1, "rev16": 1, "revsh": 1,
    "uxtb": 1, "uxth": 1, "sxtb": 1, "sxth": 1,
    "bfi": 1, "bfc": 1, "ubfx": 1, "sbfx": 1,
    "nop": 1, "it": 1, "ite": 1, "itt": 1, "itte": 1, "itee": 1, "ittt": 1,

    # Integer multiply / divide
    "mul": 1, "mla": 2, "mls": 2,
    "umull": 1, "smull": 1, "umlal": 1, "smlal": 1,
    "sdiv": 12, "udiv": 12,  # 2-12 cycles, data-dependent; assume worst-ish

    # Branches (taken branch refills the pipeline; M7 has prediction but assume a refill)
    "b": 2, "bl": 3, "bx": 2, "blx": 3, "cbz": 2, "cbnz": 2,

    # Single load / store
    "ldr": 2, "ldrb": 2, "ldrh": 2, "ldrsb": 2, "ldrsh": 2, "ldrd": 3,
    "str": 2, "strb": 2, "strh": 2, "strd": 3,
    # Load/store multiple: handled specially (1 + register count); table value is the base.
    "ldm": 1, "stm": 1, "push": 1, "pop": 1,

    # --- VFP (FPv5 single precision) ---
    "vldr": 2, "vstr": 2, "vldm": 1, "vstm": 1, "vpush": 1, "vpop": 1,
    "vmov": 1, "vmrs": 1, "vmsr": 1,
    "vadd": 1, "vsub": 1, "vmul": 1, "vnmul": 1,
    "vmla": 3, "vmls": 3, "vnmla": 3, "vnmls": 3,
    "vfma": 3, "vfms": 3, "vfnma": 3, "vfnms": 3,  # fused multiply-accumulate, ~3 cyc latency
    "vneg": 1, "vabs": 1, "vcmp": 1, "vcmpe": 1, "vminnm": 1, "vmaxnm": 1,
    "vcvt": 1, "vcvtr": 1, "vrintr": 1, "vrintz": 1, "vrintx": 1, "vsel": 1,
    "vdiv": 14, "vsqrt": 14,  # FPv5 single: ~14 cycles
}

# Addressing-mode spellings that objdump emits for the load/store-multiple family,
# folded onto the base mnemonic (which carries the cost + register-count handling).
LSM_ALIASES = {
    "ldmia": "ldm", "ldmdb": "ldm", "ldmib": "ldm", "ldmda": "ldm",
    "stmia": "stm", "stmdb": "stm", "stmib": "stm", "stmda": "stm",
    "vldmia": "vldm", "vstmia": "vstm", "vldmdb": "vldm", "vstmdb": "vstm",
}

# ARM condition-code suffixes, longest first so we don't strip a 1-letter prefix of a 2-letter code.
CONDITIONS = ["eq", "ne", "cs", "cc", "mi", "pl", "vs", "vc",
              "hi", "ls", "ge", "lt", "gt", "le", "al", "hs", "lo"]

# Lines we don't count: an instruction line is "  <hex addr>:\t<opcode bytes>\t<mnemonic>...".
INSN_RE = re.compile(r"^\s*[0-9a-fA-F]+:\t")
# A bl/blx target like "bl  1234 <sinf>" — capture the callee name for the report
# (greedy to the last '>', since demangled C++ names contain their own '<...>').
CALL_RE = re.compile(r"<(.+)>")


def base_mnemonic(mnemonic):
    """Reduce 'vfma.f32' / 'ldrb.w' / 'addeq' / 'movs' / 'stmia.w' to a lookup key."""
    m = mnemonic.lower().split(".", 1)[0]  # drop .f32 / .w / .n / .i32 ...
    m = LSM_ALIASES.get(m, m)
    if m in CYCLE_LOOKUP:
        return m
    # Strip a trailing condition code, then a flag-setting 's', only if what remains is a
    # known mnemonic (so 'mls' isn't mangled into 'm' via the 'ls' condition, etc.).
    for suffix in CONDITIONS + ["s"]:
        if m.endswith(suffix) and m[: -len(suffix)] in CYCLE_LOOKUP:
            return m[: -len(suffix)]
    return None


def count_reglist(operands):
    """Number of registers in a {...} list, expanding ranges like r4-r11 / d8-d11."""
    m = re.search(r"\{([^}]*)\}", operands)
    if not m:
        return 1
    n = 0
    for part in (p.strip() for p in m.group(1).split(",")):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            ai = re.sub(r"\D", "", a)
            bi = re.sub(r"\D", "", b)
            n += (int(bi) - int(ai) + 1) if (ai and bi) else 1
        else:
            n += 1
    return max(n, 1)


def estimate_cycles(objdump_text):
    total_cycles = 0
    total_instructions = 0
    by_mnemonic = defaultdict(lambda: [0, 0])  # base -> [count, cycles]
    unknown = defaultdict(int)
    calls = defaultdict(int)

    for line in objdump_text.splitlines():
        if not INSN_RE.match(line):
            continue
        # Tab-delimited: ['  addr:', 'opcode bytes', 'mnemonic', 'operands', ...]
        fields = line.split("\t")
        if len(fields) < 3:
            continue
        mnemonic = fields[2].strip()
        if not mnemonic or mnemonic.startswith("."):
            continue  # skip assembler directives / constant-pool data (.word, .short, ...)
        operands = fields[3] if len(fields) > 3 else ""

        total_instructions += 1
        base = base_mnemonic(mnemonic)

        if base is None:
            cycles = 1  # fallback for genuinely unknown mnemonics
            unknown[mnemonic] += 1
        else:
            cycles = CYCLE_LOOKUP[base]
            if base in ("ldm", "stm", "push", "pop", "vldm", "vstm", "vpush", "vpop"):
                cycles = 1 + count_reglist(operands)
            if base in ("bl", "blx"):
                cm = CALL_RE.search(operands)
                if cm:
                    calls[cm.group(1)] += 1

        total_cycles += cycles
        key = base if base is not None else mnemonic + " (?)"
        by_mnemonic[key][0] += 1
        by_mnemonic[key][1] += cycles

    return total_instructions, total_cycles, by_mnemonic, unknown, calls


def report(text):
    count, cycles, by_mnemonic, unknown, calls = estimate_cycles(text)

    print("By mnemonic (sorted by total cycles):")
    for key, (n, cyc) in sorted(by_mnemonic.items(), key=lambda kv: -kv[1][1]):
        print(f"  {key:<10} x{n:<4} -> ~{cyc} cycle(s)")

    if calls:
        print("\nLibrary / external calls (cost NOT included — runs in the callee):")
        for name, n in sorted(calls.items(), key=lambda kv: -kv[1]):
            print(f"  {name} x{n}")

    if unknown:
        print("\nUnknown mnemonics (counted as 1 cycle — add to CYCLE_LOOKUP if they matter):")
        for name, n in sorted(unknown.items(), key=lambda kv: -kv[1]):
            print(f"  {name} x{n}")

    print("-" * 50)
    print(f"Total Instructions: {count}")
    print(f"Rough Estimate:     ~{cycles} clock cycles (excludes bl/blx callees)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            text = f.read()
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        print("Usage: python cycle_guesser.py <objdump.asm>   (or pipe objdump output in)")
        sys.exit(1)

    print(f"Analyzing {sys.argv[1] if len(sys.argv) > 1 else 'stdin'}...\n")
    report(text)
