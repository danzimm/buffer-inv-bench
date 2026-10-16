#!/usr/bin/env python3
"""Build and run the buffer_inv microbenchmarks, then print markdown tables.

Only needs Python 3.6+ and ROCm (hipcc). Works on any wave64 gfx9 GPU with
buffer_inv, e.g. MI300 (gfx942) and MI350 (gfx950).

Usage:
  ./run.py                      # detect arch, build, run, print tables
  ./run.py --arch gfx942        # override the detected arch
  ./run.py --csv out.csv        # also save the raw per-rep CSV
"""

import argparse
import csv
import io
import os
import re
import shutil
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def rocm_path():
    return os.environ.get("ROCM_PATH", "/opt/rocm")


def find_tool(name):
    path = shutil.which(name)
    if path:
        return path
    candidate = os.path.join(rocm_path(), "bin", name)
    return candidate if os.path.exists(candidate) else None


def run(cmd, **kwargs):
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE,
                          universal_newlines=True, **kwargs).stdout


def detect_arch():
    rocminfo = find_tool("rocminfo")
    if rocminfo:
        match = re.search(r"Name:\s+(gfx[0-9a-f]+)", run([rocminfo]))
        if match:
            return match.group(1)
    amdgpu_arch = os.path.join(rocm_path(), "llvm", "bin", "amdgpu-arch")
    if os.path.exists(amdgpu_arch):
        archs = run([amdgpu_arch]).split()
        if archs:
            return archs[0]
    sys.exit("error: could not detect the GPU arch; pass --arch")


def rocm_version():
    try:
        with open(os.path.join(rocm_path(), ".info", "version")) as f:
            return f.read().strip()
    except OSError:
        return "unknown"


def build(arch, hipcc, out):
    cmd = [hipcc, "-O3", "-std=c++17", "--offload-arch=" + arch,
           os.path.join(HERE, "bench.hip"), "-o", out]
    print("building:", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)


def summarize(rows):
    """Median over reps, keyed by (test, variant, aggressor)."""
    groups = {}
    for row in rows:
        key = (row["test"], row["variant"], row["aggressor"])
        groups.setdefault(key, []).append(
            (float(row["cycles_per_iter"]), float(row["ns_per_iter"])))
    return {key: (statistics.median(c for c, _ in vals),
                  statistics.median(n for _, n in vals))
            for key, vals in groups.items()}


def table(title, note, header, lines):
    out = ["### " + title, "", note, "",
           "| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(line) + " |" for line in lines]
    return "\n".join(out) + "\n"


def report(summary, arch, device, rocm):
    out = ["## buffer_inv microbenchmarks: {} ({})".format(device, arch), "",
           "ROCm {}. Medians over reps; each rep is the median wave.".format(
               rocm), ""]

    def get(test, variant, aggressor="none"):
        return summary.get((test, variant, aggressor))

    # For vmcnt-exec0, the "cycles" column holds the fraction of waves whose
    # s_waitcnt vmcnt(1) waited for the earlier load, and "ns" the wave count.
    lines = []
    for variant in ["s_nop 0 (EXEC=0)", "buffer_inv (EXEC=0)",
                    "buffer_inv (EXEC unchanged)",
                    "global_load_dword (EXEC=0)"]:
        val = get("vmcnt-exec0", variant)
        if val:
            lines.append(["`{}`".format(variant), "{:.0f} / {:.0f}".format(
                val[0] * val[1], val[1]),
                "yes" if val[0] == 1 else "no" if val[0] == 0 else "mixed"])
    out.append(table(
        "Does the instruction increment vmcnt?",
        "Sequence: slow `global_load` into a sentinel register; instruction "
        "(with EXEC as listed); `s_waitcnt vmcnt(1)`; copy the register. The "
        "copy holds the loaded value only if the instruction incremented "
        "vmcnt, forcing vmcnt(1) to wait for the load.",
        ["instruction", "waves where vmcnt(1) waited", "increments vmcnt"],
        lines))

    for test, title, note in [
        ("issue", "Issue cost",
         "Back-to-back instructions, one `s_waitcnt vmcnt(0)` at the end. "
         "Cycles per instruction; `none` is loop overhead."),
        ("latency", "Completion latency",
         "Each instruction followed by `s_waitcnt vmcnt(0)`. Cycles per "
         "instruction + wait."),
    ]:
        lines = []
        for variant in ["none", "s_nop 0", "buffer_inv", "buffer_inv sc0"]:
            val = get(test, variant)
            if val:
                lines.append([variant, "{:.1f}".format(val[0]),
                              "{:.1f}".format(val[1])])
        out.append(table(title, note, ["instruction", "cycles", "ns"], lines))

    for mem, desc in [("coarse", "coarse-grained (`hipMalloc`)"),
                      ("fine", "fine-grained (`hipDeviceMallocFinegrained`)")]:
        test = "cache[{}]".format(mem)
        base = get(test, "none")
        lines = []
        rows = [("none", test, "none"),
                ("none, reloads use sc1 (bypass L1; L1-miss reference)",
                 "cache[{}+bypass-l1]".format(mem), "none")]
        rows += [(v, test, v) for v in ["buffer_inv", "buffer_inv sc0",
                                        "buffer_inv sc1"]]
        rows.append(("buffer_inv, reloads use sc1 (L2 check)",
                     "cache[{}+bypass-l1]".format(mem), "buffer_inv"))
        for label, row_test, variant in rows:
            val = get(row_test, variant)
            if val and base:
                lines.append([label, "{:.1f}".format(val[0]),
                              "{:+.1f}%".format(100 * (val[0] / base[0] - 1))])
        out.append(table(
            "L1-resident reload, {} memory".format(mem),
            "Per iteration: the instruction, then 8 reloads of a 2 KiB "
            "footprint in {} memory and `s_waitcnt vmcnt(0)`. If the "
            "instruction invalidated L1, the reloads would cost at least as "
            "much as the sc1 (L1-bypass) reference.".format(desc),
            ["instruction before reload", "cycles/iter", "vs none"], lines))

    base = get("cache+aggressor", "none", "s_nop 0")
    lines = []
    for aggressor in ["s_nop 0", "buffer_inv", "buffer_inv sc0"]:
        val = get("cache+aggressor", "none", aggressor)
        if val and base:
            lines.append([aggressor, "{:.1f}".format(val[0]),
                          "{:+.1f}%".format(100 * (val[0] / base[0] - 1))])
    out.append(table(
        "Effect on other waves",
        "4 waves per block reload their L1-resident footprint while 4 other "
        "waves in the same block issue the aggressor instruction in a loop.",
        ["aggressor instruction", "loader cycles/iter", "vs s_nop 0"], lines))

    base = get("cache+paced-aggressor", "none", "s_nop 0")
    lines = []
    for aggressor in ["s_nop 0", "buffer_inv"]:
        val = get("cache+paced-aggressor", "none", aggressor)
        if val and base:
            lines.append([aggressor, "{:.1f}".format(val[0]),
                          "{:+.1f}%".format(100 * (val[0] / base[0] - 1))])
    out.append(table(
        "Effect on other waves, paced",
        "Same as above, but each aggressor instruction is followed by "
        "`s_sleep 1` (about 64 cycles), i.e. occasional rather than "
        "back-to-back.",
        ["aggressor instruction", "loader cycles/iter", "vs s_nop 0"], lines))
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arch", help="offload arch (default: detect)")
    parser.add_argument("--hipcc", help="hipcc binary (default: from PATH "
                        "or $ROCM_PATH/bin)")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--iters", type=int, default=1024)
    parser.add_argument("--blocks", type=int, default=0,
                        help="blocks per launch (default: number of CUs)")
    parser.add_argument("--csv", help="also write the raw CSV here")
    parser.add_argument("--no-build", action="store_true",
                        help="reuse an existing build")
    args = parser.parse_args()

    arch = args.arch or detect_arch()
    binary = os.path.join(HERE, "build", "bench-" + arch)
    if not args.no_build:
        hipcc = args.hipcc or find_tool("hipcc")
        if not hipcc:
            sys.exit("error: hipcc not found; pass --hipcc or set ROCM_PATH")
        os.makedirs(os.path.dirname(binary), exist_ok=True)
        build(arch, hipcc, binary)

    proc = subprocess.run(
        [binary, "--reps", str(args.reps), "--iters", str(args.iters),
         "--blocks", str(args.blocks)],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True)
    sys.stderr.write(proc.stderr)
    device = re.search(r"device: (.*?) \(", proc.stderr)
    device = device.group(1) if device else "unknown device"
    if args.csv:
        with open(args.csv, "w") as f:
            f.write(proc.stdout)
    rows = list(csv.DictReader(io.StringIO(proc.stdout)))
    print(report(summarize(rows), arch, device, rocm_version()))


if __name__ == "__main__":
    main()
