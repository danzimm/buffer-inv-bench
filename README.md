# buffer_inv microbenchmarks

Measures `buffer_inv` with no `sc` bits (`BUFFER_INV 0` in MIR) on gfx942
(MI300) and gfx950 (MI350): whether it increments VM_CNT, what it costs, and
whether it invalidates caches. Where it matters, each test is run with all
lanes active and with EXEC=0.

Requirements: ROCm (`hipcc`, `rocminfo`) and Python 3.6+. Nothing else.

```
./run.py                    # detect arch, build, run, print markdown tables
./run.py --arch gfx942      # override the detected arch
./run.py --csv raw.csv      # also save the raw per-rep CSV
./run.py --reps 10 --iters 4096
```

If ROCm isn't in `/opt/rocm`, set `ROCM_PATH`.

Tests (all instruction sequences are inline asm, so they are exact):

- **vmcnt**: reads VM_CNT from `HW_REG_IB_STS` before and after the
  instruction, with a slow load outstanding so VM_CNT can't drain in between.
  `s_nop` and `global_load_dword` are the references. All lanes and EXEC=0.
- **issue**: 16 back-to-back instructions per iteration. All lanes and EXEC=0.
- **latency**: each instruction followed by `s_waitcnt vmcnt(0)`. All lanes and
  EXEC=0.
- **cache**: whether the instruction invalidates L1 or L2, calibrated against
  reloads that bypass L1 with `sc1` loads. Coarse- and fine-grained memory.
- **contention**: loader waves on the same CU while other waves issue the
  instruction back-to-back, or paced with `s_sleep 1`.

## Checking the cache results with hardware counters

The cache test infers L1/L2 hits from timing. `counters.py` checks it
directly: it runs the benchmark under `rocprofv3`, collects vector L1 (TCP)
and L2 (TCC) counters for each cache test launch, and reports them per wave
per loop iteration with an "invalidated?" verdict for L1 and L2.

```
./counters.py                 # needs rocprofv3 in PATH
./counters.py --arch gfx942
```

`bench.hip --manifest FILE` writes one line per timed launch so the profiler's
per-dispatch records can be matched to tests.

## Checking edits to bench.hip

`bash verify.sh` (or `ARCH=gfx942 bash verify.sh`) compares the GPU machine
code of the working copy against the last commit.
