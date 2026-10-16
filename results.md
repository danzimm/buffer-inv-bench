# `buffer_inv` microbenchmark results: MI300X and MI350

Measured with `bench.hip` and `run.py` (`./run.py --csv results-<arch>.csv`).
Raw per-rep data: `results-gfx942.csv` and `results-gfx950.csv`.

**Setup:** MI300X (gfx942), 304 CUs, ROCm 6.4.2; MI350 (gfx950), 256 CUs,
ROCm 7.0.2. Each value is the median over 5 reps; each rep is the median wave
(one wave per CU, or four per CU for the vmcnt test). Cycles are from
`s_memtime` (shader clock), ns from `s_memrealtime`. Plain `buffer_inv` means
no `sc` bits set (`BUFFER_INV 0` in MIR). All timed sequences are inline asm,
so the instruction stream is exact.

## 1. Does the instruction increment vmcnt?

Sequence: slow `global_load` into a register holding a sentinel; the
instruction (with EXEC as listed); `s_waitcnt vmcnt(1)`; copy the register. The
copy holds the loaded value only if the instruction incremented vmcnt, forcing
`vmcnt(1)` to wait for the load.

| instruction | MI300X: waves where `vmcnt(1)` waited | MI350: waves where `vmcnt(1)` waited | increments vmcnt |
|---|---|---|---|
| `s_nop 0`, EXEC=0 (control) | 0 / 1216 | 0 / 1024 | no |
| `buffer_inv`, EXEC=0 | 1216 / 1216 | 1024 / 1024 | **yes** |
| `buffer_inv`, EXEC unchanged (control) | 1216 / 1216 | 1024 / 1024 | yes |
| `global_load_dword`, EXEC=0 (reference) | 1216 / 1216 | 1024 / 1024 | yes |

## 2. Cost of the instruction itself

Issue: back-to-back instructions, one `s_waitcnt vmcnt(0)` at the end.
Issue + `vmcnt(0)`: each instruction followed by `s_waitcnt vmcnt(0)`.

| instruction | MI300X issue | MI350 issue | MI300X issue + `vmcnt(0)` | MI350 issue + `vmcnt(0)` |
|---|---|---|---|---|
| nothing (loop overhead) | 0 cycles | 0 cycles | 6.0 cycles | 6.0 cycles |
| `s_nop 0` | 6.0 (2.9 ns) | 6.0 (2.8 ns) | 10.0 (4.8 ns) | 10.0 (4.6 ns) |
| `buffer_inv` | 9.0 (4.3 ns) | 7.0 (3.2 ns) | 98.5 (46.8 ns) | 90.5 (41.5 ns) |
| `buffer_inv sc0` | 9.0 (4.3 ns) | 7.0 (3.2 ns) | 98.5 (46.8 ns) | 94.5 (42.7 ns) |

The timer resolves these in 4-cycle steps; MI350 runs vary between 90.5 and
94.5 cycles for both rows.

## 3. Does it invalidate L1 or L2?

Per iteration: the instruction, then 8 reloads of a 2 KiB footprint that is
already in L1, then `s_waitcnt vmcnt(0)`. Cycles per iteration; percentages are
relative to "nothing". Coarse is `hipMalloc`, fine is
`hipDeviceMallocFinegrained`.

| before the reload | MI300X coarse | MI350 coarse | MI300X fine | MI350 fine |
|---|---|---|---|---|
| nothing (L1 hits) | 176.0 | 172.0 | 176.0 | 172.0 |
| nothing, loads bypass L1 with `sc1` (L1-miss reference) | 272.1 (+54.6%) | 265.4 (+54.3%) | 272.2 (+54.7%) | 264.3 (+53.7%) |
| **`buffer_inv`** | **180.0 (+2.3%)** | **176.0 (+2.3%)** | **180.0 (+2.3%)** | **176.0 (+2.3%)** |
| `buffer_inv sc0` | 180.0 (+2.3%) | 176.0 (+2.3%) | 180.0 (+2.3%) | 176.0 (+2.3%) |
| `buffer_inv sc1` (invalidates, for contrast) | 5384.1 (+2959%) | 4350.4 (+2429%) | 5384.4 (+2959%) | 4350.4 (+2429%) |
| **`buffer_inv`, loads bypass L1 (L2 check)** | **276.0** | **269.2** | **276.9** | **268.9** |
| L2 check vs the L1-miss reference | +1.5% | +1.4% | +1.7% | +1.7% |

If `buffer_inv` invalidated L1, its row would cost at least as much as the
L1-miss reference; if it invalidated L2, the L2-check row would go to memory.
Both rows only add the instruction's own issue cost.

## 4. Effect on other waves on the same CU

Four waves per CU reload their L1-resident footprint while four other waves on
the same CU issue the listed instruction, either back-to-back or paced with one
`s_sleep 1` (about 64 cycles) after each.

| other waves issue… | MI300X loader cycles/iter | MI350 loader cycles/iter |
|---|---|---|
| `s_nop 0` back-to-back (baseline) | 192.0 | 191.9 |
| `buffer_inv` back-to-back | 514.3 (**+167.8%**) | 384.0 (**+100.1%**) |
| `buffer_inv sc0` back-to-back | 514.4 (+167.9%) | 384.0 (+100.1%) |
| `s_nop 0` paced (baseline) | 192.0 | 188.1 |
| `buffer_inv` paced | 341.3 (**+77.7%**) | 192.1 (+2.1%) |

## Summary

- **Same on both:** plain `buffer_inv` increments vmcnt, including when EXEC is
  zero, and invalidates neither L1 nor L2 for coarse- or fine-grained memory. It
  costs one instruction issue (7–9 cycles) and completes in about 90–100 cycles.
- **Different:** contention with other waves on the CU. A back-to-back stream
  slows other waves' loads on both GPUs. With pacing, MI350 recovers (+2%) but
  MI300X still shows +78% at about one `buffer_inv` per 16 cycles per CU (4
  paced waves).
