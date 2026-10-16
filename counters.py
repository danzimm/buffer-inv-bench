#!/usr/bin/env python3
"""Check cache invalidation directly with hardware counters (rocprofv3).

Runs the benchmark under rocprofv3, collecting vector L1 (TCP) and L2 (TCC)
counters for every cacheKernel launch, and prints, per cache test case, the
counts per wave per loop iteration. This checks what the timing-based cache
tables in run.py infer: whether the reloads hit L1, and whether the
instruction before them invalidates L1 or L2.

Needs ROCm with rocprofv3, plus what run.py needs.

Usage:
  ./counters.py                 # detect arch, build, profile, print tables
  ./counters.py --arch gfx942   # override the detected arch
  ./counters.py --keep DIR      # keep the rocprofv3 output and manifest in DIR
"""

import argparse
import csv
import os
import shutil
import statistics
import subprocess
import sys
import tempfile

import run  # Build helpers shared with run.py.

COUNTERS = [
    "TCP_TOTAL_CACHE_ACCESSES_sum",  # Vector L1 accesses.
    "TCP_TCC_READ_REQ_sum",          # Reads sent from L1 to L2 (L1 misses).
    "TCC_REQ_sum",                   # All L2 requests.
    "TCC_HIT_sum",                   # L2 hits.
    "TCC_MISS_sum",                  # L2 misses.
]

# Each loader wave reloads a 2 KiB footprint per iteration: 8 loads of 256
# bytes, which is 16 128-byte cache lines.
LINES_PER_ITER = 16


def profile(binary, workdir, reps):
    manifest = os.path.join(workdir, "manifest.csv")
    cmd = ["rocprofv3", "--pmc"] + COUNTERS + [
        "--kernel-include-regex", "cacheKernel",
        "-d", workdir, "-o", "counters", "--output-format", "csv",
        "--", binary, "--reps", str(reps), "--manifest", manifest]
    print("profiling:", " ".join(cmd), file=sys.stderr)
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                          stderr=subprocess.PIPE, universal_newlines=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        sys.exit("error: rocprofv3 failed (if it reports that the counters "
                 "don't fit in one pass, collect fewer counters at a time)")
    found = []
    for root, _, files in os.walk(workdir):
        found += [os.path.join(root, f) for f in files
                  if f.endswith("counter_collection.csv")]
    if len(found) != 1:
        sys.exit("error: expected one counter_collection.csv under {}, "
                 "found {}".format(workdir, found))
    return found[0], manifest


def load_dispatches(path):
    """Counter values per dispatch, in dispatch order."""
    by_id = {}
    for row in csv.DictReader(open(path)):
        by_id.setdefault(int(row["Dispatch_Id"]), {})[row["Counter_Name"]] = \
            float(row["Counter_Value"])
    return [by_id[d] for d in sorted(by_id)]


def load_launches(path, kernel):
    return [row for row in csv.DictReader(open(path))
            if row["kernel"] == kernel]


def per_iter(counters, launch):
    """Counts per wave per loop iteration for one launch."""
    waves = int(launch["waves"])
    iters = int(launch["ops_per_wave"])
    return {name: counters[name] / (waves * iters) for name in COUNTERS}


def report(results):
    out = ["## Cache invalidation, measured with hardware counters", "",
           "rocprofv3 counters per `cacheKernel` launch, divided by "
           "(loader waves × iterations): counts per wave per loop iteration. "
           "Each iteration reloads a 2 KiB footprint, which is "
           "{} 128-byte lines. The warm-up loads before the timed loop add "
           "about {} ÷ iterations, which is negligible. Medians over reps."
           .format(LINES_PER_ITER, LINES_PER_ITER), ""]
    for mem in ["coarse", "fine"]:
        test, bypass = "cache[{}]".format(mem), "cache[{}+bypass-l1]".format(mem)
        rows = [("nothing", test, "none"),
                ("nothing, reloads bypass L1 with sc1", bypass, "none"),
                ("`buffer_inv`", test, "buffer_inv"),
                ("`buffer_inv sc0`", test, "buffer_inv sc0"),
                ("`buffer_inv sc1`", test, "buffer_inv sc1"),
                ("`buffer_inv`, reloads bypass L1 (L2 check)", bypass,
                 "buffer_inv")]
        header = ["before the reload", "L1 accesses", "L1→L2 reads",
                  "L2 requests", "L2 hits", "L2 misses", "L1 invalidated?",
                  "L2 invalidated?"]
        lines = []
        for label, row_test, inst in rows:
            c = results.get((row_test, inst))
            if c is None:
                continue
            reads = c["TCP_TCC_READ_REQ_sum"]
            misses = c["TCC_MISS_sum"]
            if row_test == bypass:
                # The reloads skip L1 by design, so L1 can't be judged here.
                l1 = "n/a (bypassed)"
            else:
                l1 = "yes" if reads > LINES_PER_ITER / 2 else "no"
            # If L2 lost the footprint, the reloads that reach L2 would miss.
            l2 = ("yes" if misses > LINES_PER_ITER / 2 else
                  "no" if reads > LINES_PER_ITER / 2 else
                  "n/a (reloads hit L1)")
            lines.append([label] + ["{:.2f}".format(c[n]) for n in COUNTERS]
                         + [l1, l2])
        out += ["### {} memory".format(mem.capitalize()), "",
                "| " + " | ".join(header) + " |",
                "|" + "|".join("---" for _ in header) + "|"]
        out += ["| " + " | ".join(line) + " |" for line in lines]
        out.append("")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arch", help="offload arch (default: detect)")
    parser.add_argument("--hipcc", help="hipcc binary (default: from PATH "
                        "or $ROCM_PATH/bin)")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--no-build", action="store_true",
                        help="reuse an existing build")
    parser.add_argument("--keep", metavar="DIR",
                        help="keep the rocprofv3 output and manifest in DIR")
    args = parser.parse_args()

    if not shutil.which("rocprofv3"):
        sys.exit("error: rocprofv3 not found in PATH")
    arch = args.arch or run.detect_arch()
    binary = os.path.join(run.HERE, "build", "bench-" + arch)
    if not args.no_build:
        hipcc = args.hipcc or run.find_tool("hipcc")
        if not hipcc:
            sys.exit("error: hipcc not found; pass --hipcc or set ROCM_PATH")
        os.makedirs(os.path.dirname(binary), exist_ok=True)
        run.build(arch, hipcc, binary)

    workdir = args.keep or tempfile.mkdtemp(prefix="buffer-inv-counters-")
    os.makedirs(workdir, exist_ok=True)
    try:
        counter_csv, manifest = profile(binary, workdir, args.reps)
        dispatches = load_dispatches(counter_csv)
        launches = load_launches(manifest, "cacheKernel")
        if len(dispatches) != len(launches):
            sys.exit("error: {} cacheKernel dispatches profiled but {} "
                     "launched".format(len(dispatches), len(launches)))
        groups = {}
        for counters, launch in zip(dispatches, launches):
            if int(launch["rep"]) < 0:
                continue  # Warm-up launch.
            key = (launch["test"], launch["instruction"])
            groups.setdefault(key, []).append(per_iter(counters, launch))
        results = {key: {n: statistics.median(r[n] for r in reps)
                         for n in COUNTERS}
                   for key, reps in groups.items()}
        print(report(results))
    finally:
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
