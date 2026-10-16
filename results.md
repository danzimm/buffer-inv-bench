# `buffer_inv` microbenchmark results: MI300X and MI350

Measured with `bench.hip` and `run.py` (`./run.py --csv results-<arch>.csv`).
Raw per-rep data: `results-gfx942.csv` and `results-gfx950.csv`.

**Setup:** MI300X (gfx942), 304 CUs, ROCm 6.4.2; MI350 (gfx950), 256 CUs,
ROCm 7.0.2. Each timing is the median over 5 reps; each rep is the median wave
(one wave per CU, or four loader waves per CU for the contention test). Cycles
are from `s_memtime` (shader clock), ns from `s_memrealtime`. Plain
`buffer_inv` means no `sc` bits set (`BUFFER_INV 0` in MIR). All measured
instruction sequences are inline asm, except the MFMA loader, which uses the
compiler builtin so the required wait states are inserted.

## 1. Does the instruction increment VM_CNT?

VM_CNT is read from `HW_REG_IB_STS` (bits 23:22 and 3:0, per the CDNA3/CDNA4
ISA, Table 20) right after a slow `global_load`, and again right after the
instruction. The outstanding load keeps VM_CNT from draining between the two
reads, since VM_CNT decrements in order. "Waves incremented" is the worst of 5
reps.

| instruction | EXEC | VM_CNT before → after (both GPUs) | MI300X: waves incremented | MI350: waves incremented | increments VM_CNT |
|---|---|---|---|---|---|
| `s_nop 0` (reference) | all lanes | 1 → 1 | 0 / 1216 | 0 / 1024 | no |
| `s_nop 0` (reference) | EXEC=0 | 1 → 1 | 0 / 1216 | 0 / 1024 | no |
| **`buffer_inv`** | all lanes | 1 → 2 | 1216 / 1216 | 1024 / 1024 | **yes** |
| **`buffer_inv`** | **EXEC=0** | 1 → 2 | 1216 / 1216 | 1024 / 1024 | **yes** |
| `global_load_dword` (reference) | all lanes | 1 → 2 | 1216 / 1216 | 1024 / 1024 | yes |
| `global_load_dword` (reference) | EXEC=0 | 1 → 2 | 1216 / 1216 | 1024 / 1024 | yes |

## 2. Cost of the instruction itself

Issue: 16 back-to-back instructions per iteration, one `s_waitcnt vmcnt(0)` at
the end. Issue + wait: each instruction followed by `s_waitcnt vmcnt(0)`. Both
include about 0.75 cycles per instruction of loop and EXEC save/restore
overhead, the same for every row.

| instruction | EXEC | MI300X issue | MI350 issue | MI300X issue + `vmcnt(0)` | MI350 issue + `vmcnt(0)` |
|---|---|---|---|---|---|
| nothing (the wait alone) | all lanes | — | — | 6.8 (3.2 ns) | 6.8 (3.1 ns) |
| `s_nop 0` | all lanes | 6.8 (3.2 ns) | 6.8 (3.1 ns) | 10.8 (5.1 ns) | 10.8 (4.9 ns) |
| `buffer_inv` | all lanes | 9.0 (4.3 ns) | 7.8 (3.6 ns) | 95.5 (45.7 ns) | 91.5 (42.1 ns) |
| `buffer_inv` | EXEC=0 | 9.0 (4.3 ns) | 7.8 (3.6 ns) | 99.5 (47.3 ns) | 91.5 (42.1 ns) |
| `buffer_inv sc0` | all lanes | 9.0 (4.3 ns) | 7.8 (3.6 ns) | 95.5 (45.7 ns) | 91.5 (42.1 ns) |

The issue + wait figures move in 4-cycle steps of the timer: on each GPU, all
three `buffer_inv` rows alternate between the same two values across reps
(95.5/99.5 on MI300X, 91.5/95.5 on MI350). The EXEC=0 row's 99.5 on MI300X is
one such step, not a real difference.

## 3. Does it invalidate L1 or L2?

Measured with hardware counters (rocprofv3: `TCP_TCC_READ_REQ_sum`,
`TCC_HIT_sum`, `TCC_MISS_sum`), per wave per loop iteration. Each iteration
issues the instruction, then reloads a 2 KiB footprint (16 128-byte lines).
Plain reloads hit L1 unless the instruction invalidated it; reloads that bypass
L1 (`sc1` loads) hit L2 unless it invalidated L2. About 0.02 per iteration is
the warm-up spread over the run. Coarse-grained (`hipMalloc`) and fine-grained
(`hipDeviceMallocFinegrained`) memory gave identical results, and so did both
GPUs (the counts below are the same to two decimals on MI300X and MI350).

| before the reload | L1→L2 reads | L2 hits | L2 misses | L1 invalidated? | L2 invalidated? |
|---|---|---|---|---|---|
| nothing | 0.02 | 0.01 | 0.02 | no | not observed (reloads hit L1) |
| **`buffer_inv`** | **0.02** | 0.01 | 0.02 | **no** | not observed (reloads hit L1) |
| `buffer_inv sc1` (positive control) | 16.02 | 16.01 | 0.02 | yes | no |
| **`buffer_inv`, then reloads that bypass L1** | 16.02 | **16.01** | **0.02** | n/a (bypassed) | **no** |

This matches the CDNA4 ISA, Table 52 (`BUFFER_INV`): with SC1=0 and SC0=0, the
CU cache and L2 behavior are both "NOP". `buffer_inv sc1` invalidates the CU
cache, and in L2 only non-coherently cached lines, which is why the footprint
here stays in L2.

## 4. Effect on other waves on the same CU

4 loader waves per CU are timed doing the listed work each iteration, while 4
other waves of the same block issue `s_nop 0` (baseline) or `buffer_inv` in a
loop, back-to-back or paced with `s_sleep 1` (about 64 cycles) after each.
Slowdown of the loaders relative to `s_nop` aggressors with the same pacing
(loader cycles per iteration in parentheses: `s_nop` → `buffer_inv`).

| loader work | MI300X back-to-back | MI350 back-to-back | MI300X paced | MI350 paced |
|---|---|---|---|---|
| vector memory loads (8 L1-hit loads, `vmcnt(0)`) | **+167.9%** (192.0 → 514.5) | **+100.1%** (191.9 → 384.0) | **+77.7%** (192.0 → 341.3) | +2.1% (188.1 → 192.1) |
| LDS reads (8 `ds_read_b32`, `lgkmcnt(0)`) | −1.5% (134.1 → 132.1) | −0.2% (135.9 → 135.6) | +0.0% (132.2 → 132.2) | −0.0% (135.0 → 135.0) |
| MFMAs (8 dependent `v_mfma_f32_32x32x8_f16`) | −0.4% (265.1 → 264.1) | −0.4% (265.1 → 264.1) | +0.0% (264.1 → 264.1) | +0.0% (264.1 → 264.1) |

`buffer_inv` from other waves only slows vector memory work: it competes for
the CU's vector memory path. LDS reads and MFMAs are unaffected on both GPUs.
On MI350 the effect is significant only when `buffer_inv` is issued nearly
back-to-back; on MI300X it remains large at one `buffer_inv` per about 64
cycles per aggressor wave (about one per 16 cycles per CU). Both rates are far
higher than branch padding would produce (one `buffer_inv` per padded branch).

## Summary

- `buffer_inv` with no `sc` bits increments VM_CNT, including when EXEC is
  zero, and invalidates neither L1 nor L2, on both MI300X and MI350.
- It costs about one instruction issue (9.0 cycles on MI300X and 7.8 on MI350,
  vs 6.8 for `s_nop`) and completes in about 90–100 cycles.
- Its only measured cost to other waves is contention in the vector memory
  path. On MI300X that contention is noticeable even at moderately high
  `buffer_inv` rates; LDS reads and MFMAs are unaffected on both GPUs.
