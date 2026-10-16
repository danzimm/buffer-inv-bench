# buffer_inv microbenchmarks

Measures what `buffer_inv` with no `sc` bits costs on gfx942 (MI300) and
gfx950 (MI350), and whether it increments vmcnt when EXEC is 0.

Requirements: ROCm (`hipcc`, `rocminfo`) and Python 3.6+. Nothing else.

```
./run.py                    # detect arch, build, run, print markdown tables
./run.py --arch gfx942      # override the detected arch
./run.py --csv raw.csv      # also save the raw per-rep CSV
./run.py --reps 10 --iters 4096
```

If ROCm isn't in `/opt/rocm`, set `ROCM_PATH`.

Tests (all instruction sequences are inline asm, so they are exact):

- **vmcnt with EXEC=0**: functional check of whether the instruction
  increments vmcnt, using a sentinel register and `s_waitcnt vmcnt(1)`.
- **Issue cost**: back-to-back instructions.
- **Completion latency**: each instruction followed by `s_waitcnt vmcnt(0)`.
- **L1-resident reload**: whether the instruction invalidates L1 (or L2),
  calibrated against reloads that bypass L1 with `sc1` loads.
- **Effect on other waves**: loader waves on the same CU while other waves
  issue the instruction back-to-back, or paced with `s_sleep 1`.
