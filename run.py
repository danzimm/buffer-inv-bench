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


class Results:
    """Medians over reps of the long-format CSV."""

    def __init__(self, rows):
        groups = {}
        for row in rows:
            key = (row["test"], row["instruction"], row["exec"],
                   row["aggressor"], row["metric"])
            groups.setdefault(key, []).append(float(row["value"]))
        self.medians = {k: statistics.median(v) for k, v in groups.items()}
        self.mins = {k: min(v) for k, v in groups.items()}

    def get(self, test, inst, metric, exec_="all", aggressor="none"):
        return self.medians.get((test, inst, exec_, aggressor, metric))

    def min(self, test, inst, metric, exec_="all", aggressor="none"):
        return self.mins.get((test, inst, exec_, aggressor, metric))


def table(title, note, header, lines):
    out = ["### " + title, "", note, "",
           "| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(line) + " |" for line in lines]
    return "\n".join(out) + "\n"


def exec_label(exec_):
    return "EXEC=0" if exec_ == "0" else "all lanes"


def pct(val, base):
    return "{:+.1f}%".format(100 * (val / base - 1))


def report(res, arch, device, rocm):
    out = ["## buffer_inv microbenchmarks: {} ({})".format(device, arch), "",
           "ROCm {}. Medians over reps; each rep is the median wave.".format(
               rocm), ""]

    # vmcnt
    lines = []
    for inst, exec_ in [("s_nop 0", "all"), ("s_nop 0", "0"),
                        ("buffer_inv", "all"), ("buffer_inv", "0"),
                        ("global_load_dword", "all"),
                        ("global_load_dword", "0")]:
        before = res.get("vmcnt", inst, "vmcnt_before", exec_)
        after = res.get("vmcnt", inst, "vmcnt_after", exec_)
        waves = res.get("vmcnt", inst, "waves", exec_)
        # Worst rep, so a single bad rep is not hidden by the median.
        inc = res.min("vmcnt", inst, "waves_incremented", exec_)
        if None in (before, after, waves, inc):
            continue
        verdict = "yes" if inc == waves else "no" if inc == 0 else "mixed"
        lines.append(["`{}`".format(inst), exec_label(exec_),
                      "{:.0f} → {:.0f}".format(before, after),
                      "{:.0f} / {:.0f}".format(inc, waves), verdict])
    out.append(table(
        "Does the instruction increment VM_CNT?",
        "VM_CNT read from `HW_REG_IB_STS` (bits 23:22 and 3:0) right after a "
        "slow `global_load`, and again right after the instruction. The load "
        "keeps VM_CNT from draining in between, since VM_CNT decrements in "
        "order. \"Waves incremented\" is the worst rep.",
        ["instruction", "EXEC", "VM_CNT before → after",
         "waves incremented", "increments VM_CNT"], lines))

    # issue and latency
    for test, title, note, cases in [
        ("issue", "Issue cost",
         "16 back-to-back instructions per iteration inside an EXEC "
         "save/restore, one `s_waitcnt vmcnt(0)` at the end. Cycles per "
         "instruction, including about 0.75 cycles of loop and EXEC "
         "save/restore overhead.",
         [("s_nop 0", "all"), ("buffer_inv", "all"), ("buffer_inv", "0"),
          ("buffer_inv sc0", "all")]),
        ("latency", "Completion latency",
         "Each instruction followed by `s_waitcnt vmcnt(0)`. Cycles per "
         "instruction + wait; `none` is the wait alone.",
         [("none", "all"), ("s_nop 0", "all"), ("buffer_inv", "all"),
          ("buffer_inv", "0"), ("buffer_inv sc0", "all")]),
    ]:
        lines = []
        for inst, exec_ in cases:
            cyc = res.get(test, inst, "cycles", exec_)
            ns = res.get(test, inst, "ns", exec_)
            if cyc is not None:
                lines.append(["`{}`".format(inst), exec_label(exec_),
                              "{:.1f}".format(cyc), "{:.1f}".format(ns)])
        out.append(table(title, note, ["instruction", "EXEC", "cycles", "ns"],
                         lines))

    # cache
    for mem, desc in [("coarse", "coarse-grained (`hipMalloc`)"),
                      ("fine", "fine-grained (`hipDeviceMallocFinegrained`)")]:
        test = "cache[{}]".format(mem)
        bypass = "cache[{}+bypass-l1]".format(mem)
        base = res.get(test, "none", "cycles")
        rows = [("nothing (L1 hits)", test, "none"),
                ("nothing, reloads bypass L1 with sc1 (L1-miss reference)",
                 bypass, "none"),
                ("`buffer_inv`", test, "buffer_inv"),
                ("`buffer_inv sc0`", test, "buffer_inv sc0"),
                ("`buffer_inv sc1`", test, "buffer_inv sc1"),
                ("`buffer_inv`, reloads bypass L1 (L2 check)", bypass,
                 "buffer_inv")]
        lines = []
        for label, row_test, inst in rows:
            val = res.get(row_test, inst, "cycles")
            if val is not None and base:
                lines.append([label, "{:.1f}".format(val), pct(val, base)])
        out.append(table(
            "L1/L2 invalidation, {} memory".format(mem),
            "Per iteration: the instruction, then 8 reloads of a 2 KiB "
            "footprint in {} memory and `s_waitcnt vmcnt(0)`. If the "
            "instruction invalidated L1, its row would cost at least as much "
            "as the L1-miss reference; the L2 check would go to memory if it "
            "invalidated L2.".format(desc),
            ["before the reload", "cycles/iter", "vs nothing"], lines))

    # contention
    for test, title, note, aggressors in [
        ("contention", "Effect on other waves, back-to-back",
         "4 waves per CU reload their L1-resident footprint while 4 other "
         "waves of the same block issue the aggressor instruction in a loop.",
         ["s_nop 0", "buffer_inv", "buffer_inv sc0"]),
        ("contention-paced", "Effect on other waves, paced",
         "Same, but each aggressor instruction is followed by `s_sleep 1` "
         "(about 64 cycles).",
         ["s_nop 0", "buffer_inv"]),
    ]:
        base = res.get(test, "none", "cycles", aggressor="s_nop 0")
        lines = []
        for agg in aggressors:
            val = res.get(test, "none", "cycles", aggressor=agg)
            if val is not None and base:
                lines.append(["`{}`".format(agg), "{:.1f}".format(val),
                              pct(val, base)])
        out.append(table(title, note, ["aggressor instruction",
                                       "loader cycles/iter", "vs s_nop 0"],
                         lines))
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
    print(report(Results(rows), arch, device, rocm_version()))


if __name__ == "__main__":
    main()
