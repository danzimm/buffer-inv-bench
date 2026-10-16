#!/usr/bin/env python3
"""Build and run the buffer_inv microbenchmarks, then print markdown tables.

Only needs Python 3.6+ and ROCm (hipcc, plus rocprofv3 for the cache test).
Works on any wave64 gfx9 GPU with buffer_inv, e.g. MI300 (gfx942) and MI350
(gfx950).

The timed tests run first. Then the cache test runs under rocprofv3, which
collects vector L1 (TCP) and L2 (TCC) hardware counters for each launch.

Usage:
  ./run.py                      # detect arch, build, run, print tables
  ./run.py --arch gfx942        # override the detected arch
  ./run.py --csv out.csv        # also save the raw per-rep CSV
  ./run.py --no-counters        # skip the rocprofv3 cache test
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
import tempfile

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


# Hardware counters for the cache test, and the CSV metric names they are
# reported as, per wave per loop iteration.
COUNTERS = {
    "TCP_TCC_READ_REQ_sum": "l1_to_l2_reads_per_iter",  # L1 misses.
    "TCC_HIT_sum": "l2_hits_per_iter",
    "TCC_MISS_sum": "l2_misses_per_iter",
}

# Each cache test iteration reloads a 2 KiB footprint: 16 128-byte lines.
LINES_PER_ITER = 16


def run_counters(binary, args, workdir):
    """Run the cache test under rocprofv3. Returns CSV rows in the same
    long format as the benchmark's own output."""
    manifest = os.path.join(workdir, "manifest.csv")
    cmd = ["rocprofv3", "--pmc"] + list(COUNTERS) + [
        "--kernel-include-regex", "cacheKernel",
        "-d", workdir, "-o", "counters", "--output-format", "csv",
        "--", binary, "--tests", "cache", "--reps", str(args.reps),
        "--iters", str(args.iters), "--blocks", str(args.blocks),
        "--manifest", manifest]
    print("profiling:", " ".join(cmd), file=sys.stderr)
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                          stderr=subprocess.PIPE, universal_newlines=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        sys.exit("error: rocprofv3 failed; rerun with --no-counters to skip "
                 "the cache test")
    found = [os.path.join(root, f) for root, _, files in os.walk(workdir)
             for f in files if f.endswith("counter_collection.csv")]
    if len(found) != 1:
        sys.exit("error: expected one counter_collection.csv under {}, "
                 "found {}".format(workdir, found))

    # Counter values per dispatch, in dispatch order.
    by_id = {}
    for row in csv.DictReader(open(found[0])):
        by_id.setdefault(int(row["Dispatch_Id"]), {})[row["Counter_Name"]] = \
            float(row["Counter_Value"])
    dispatches = [by_id[d] for d in sorted(by_id)]
    # The manifest says which test each launch belongs to, in launch order.
    launches = [row for row in csv.DictReader(open(manifest))
                if row["kernel"] == "cacheKernel"]
    if len(dispatches) != len(launches):
        sys.exit("error: {} cacheKernel dispatches profiled but {} launched"
                 .format(len(dispatches), len(launches)))

    rows = []
    for counters, launch in zip(dispatches, launches):
        if int(launch["rep"]) < 0:
            continue  # Warm-up launch.
        per = int(launch["waves"]) * int(launch["ops_per_wave"])
        for counter, metric in COUNTERS.items():
            rows.append({"test": launch["test"],
                         "instruction": launch["instruction"],
                         "exec": launch["exec"],
                         "aggressor": launch["aggressor"],
                         "rep": launch["rep"], "metric": metric,
                         "value": "{:.4f}".format(counters[counter] / per)})
    return rows


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
    for mem in ["coarse", "fine"]:
        test = "cache[{}]".format(mem)
        bypass = "cache[{}+bypass-l1]".format(mem)
        rows = [("nothing", test, "none"),
                ("`buffer_inv`", test, "buffer_inv"),
                ("`buffer_inv sc1` (positive control)", test,
                 "buffer_inv sc1"),
                ("`buffer_inv`, then reloads that bypass L1 with sc1", bypass,
                 "buffer_inv")]
        lines = []
        for label, row_test, inst in rows:
            reads = res.get(row_test, inst, "l1_to_l2_reads_per_iter")
            hits = res.get(row_test, inst, "l2_hits_per_iter")
            misses = res.get(row_test, inst, "l2_misses_per_iter")
            if None in (reads, hits, misses):
                continue
            if row_test == bypass:
                l1 = "n/a (reloads bypass L1)"
            else:
                l1 = "yes" if reads > LINES_PER_ITER / 2 else "no"
            # L2 is only observed when the reloads reach it.
            if reads > LINES_PER_ITER / 2:
                l2 = "yes" if misses > LINES_PER_ITER / 2 else "no"
            else:
                l2 = "n/a (reloads hit L1)"
            lines.append([label, "{:.2f}".format(reads), "{:.2f}".format(hits),
                          "{:.2f}".format(misses), l1, l2])
        if lines:
            out.append(table(
                "Cache invalidation, {} memory".format(mem),
                "Hardware counters (rocprofv3) per wave per loop iteration. "
                "Each iteration issues the instruction, then reloads a 2 KiB "
                "footprint ({} 128-byte lines) in {}-grained memory. Plain "
                "reloads hit L1 unless the instruction invalidated it; reloads "
                "that bypass L1 hit L2 unless it invalidated L2. About 0.02 "
                "per iteration is the warm-up spread over the run.".format(
                    LINES_PER_ITER, mem),
                ["before the reload", "L1→L2 reads", "L2 hits", "L2 misses",
                 "L1 invalidated?", "L2 invalidated?"], lines))

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
    parser.add_argument("--no-counters", action="store_true",
                        help="skip the cache test (needs rocprofv3)")
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
        [binary, "--tests", "vmcnt,issue,latency,contention", "--reps",
         str(args.reps), "--iters", str(args.iters), "--blocks",
         str(args.blocks)],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True)
    sys.stderr.write(proc.stderr)
    device = re.search(r"device: (.*?) \(", proc.stderr)
    device = device.group(1) if device else "unknown device"
    rows = list(csv.DictReader(io.StringIO(proc.stdout)))

    if args.no_counters:
        print("skipping the cache test (--no-counters)", file=sys.stderr)
    elif not shutil.which("rocprofv3"):
        print("skipping the cache test: rocprofv3 not found in PATH",
              file=sys.stderr)
    else:
        workdir = tempfile.mkdtemp(prefix="buffer-inv-counters-")
        try:
            rows += run_counters(binary, args, workdir)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    if args.csv:
        fields = ["test", "instruction", "exec", "aggressor", "rep", "metric",
                  "value"]
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(report(Results(rows), arch, device, rocm_version()))


if __name__ == "__main__":
    main()
