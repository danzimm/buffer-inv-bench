#!/usr/bin/env bash
# Self-extracting buffer_inv microbenchmark (gfx942/gfx950).
# Unpacks into ${BENCH_DIR:-./buffer-inv-bench}, then builds and runs it.
# Extra arguments are passed to run.py, e.g. --csv raw.csv or --reps 10.
# To check the cache results with hardware counters afterwards, run
#   python3 ${BENCH_DIR:-./buffer-inv-bench}/counters.py
set -euo pipefail
dir="${BENCH_DIR:-./buffer-inv-bench}"
mkdir -p "$dir"
cat > "$dir/bench.hip" <<'BUFFER_INV_BENCH_EOF_BENCH_HIP'
// Microbenchmarks for `buffer_inv` with no scope bits (`BUFFER_INV 0` in MIR)
// on gfx942/gfx950.
//
// Each test measures an instruction under an EXEC condition (all lanes active,
// or EXEC=0), with buffer_inv as the subject and other instructions as
// references:
//
//   vmcnt       Does the instruction increment VM_CNT? Read from HW_REG_IB_STS.
//   issue       Cost of issuing it back-to-back.
//   latency     Cost of issuing it and waiting with `s_waitcnt vmcnt(0)`.
//   cache       Does it invalidate the vector L1 or L2?
//   contention  Does it slow the loads of other waves on the same CU?
//
// All measured sequences are inline asm, so the instruction stream is exact.
// The compiler's waitcnt insertion does not look inside inline asm, so every
// test issues its own s_waitcnt.
//
// Output: long-format CSV on stdout:
//   test,instruction,exec,aggressor,rep,metric,value
// Timing metrics are `cycles` (s_memtime, shader clock) and `ns`
// (s_memrealtime, 100 MHz), per operation or iteration.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t Err = (expr);                                                   \
    if (Err != hipSuccess) {                                                   \
      fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #expr,            \
              hipGetErrorString(Err));                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

//===----------------------------------------------------------------------===//
// Instructions and EXEC conditions
//===----------------------------------------------------------------------===//

enum Inst { NONE, NOP, INV, INV_SC0, INV_SC1, LOAD };

static const char *instName(int I) {
  switch (I) {
  case NONE: return "none";
  case NOP: return "s_nop 0";
  case INV: return "buffer_inv";
  case INV_SC0: return "buffer_inv sc0";
  case INV_SC1: return "buffer_inv sc1";
  case LOAD: return "global_load_dword";
  }
  return "?";
}

enum Exec { EXEC_ALL, EXEC_ZERO };

static const char *execName(int E) { return E == EXEC_ZERO ? "0" : "all"; }

// Dependent false, for the static_assert that ends each if-constexpr chain.
template <int> constexpr bool Unsupported = false;

// Several tests wrap the measured instructions in an EXEC save/restore:
//
//   s_mov_b64 save, exec
//   s_mov_b64 exec, 0         ; for EXEC=0; `s_mov_b64 exec, save` for all lanes
//   <measured instructions>
//   s_mov_b64 exec, save
//
// The all-lanes form writes EXEC back unchanged, so both forms run the same
// number of instructions and their costs are comparable. The whole sequence
// is one asm statement, so the compiler cannot place its own instructions
// while EXEC is 0.

__device__ __forceinline__ void waitVmcnt0() {
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
}

// One instruction, all lanes active.
template <int I> __device__ __forceinline__ void op() {
  if constexpr (I == NONE) {
    // Nothing.
  } else if constexpr (I == NOP) {
    asm volatile("s_nop 0" ::: "memory");
  } else if constexpr (I == INV) {
    asm volatile("buffer_inv" ::: "memory");
  } else if constexpr (I == INV_SC0) {
    asm volatile("buffer_inv sc0" ::: "memory");
  } else if constexpr (I == INV_SC1) {
    asm volatile("buffer_inv sc1" ::: "memory");
  } else {
    static_assert(Unsupported<I>, "unsupported instruction");
  }
}

//===----------------------------------------------------------------------===//
// Timing
//===----------------------------------------------------------------------===//

struct Timer {
  unsigned long long Cycles, Real;
  __device__ __forceinline__ void start() {
    Cycles = __builtin_amdgcn_s_memtime();
    Real = __builtin_amdgcn_s_memrealtime();
  }
  __device__ __forceinline__ void stop(unsigned long long *OutCycles,
                                       unsigned long long *OutReal, int Slot) {
    unsigned long long C = __builtin_amdgcn_s_memtime();
    unsigned long long R = __builtin_amdgcn_s_memrealtime();
    if (threadIdx.x % warpSize == 0) {
      OutCycles[Slot] = C - Cycles;
      OutReal[Slot] = R - Real;
    }
  }
};

//===----------------------------------------------------------------------===//
// Test "vmcnt": does the instruction increment VM_CNT?
//
//   s_waitcnt vmcnt(0)
//   global_load (slow)            ; VM_CNT = 1
//   before = IB_STS
//   <instruction, under EXEC condition>
//   after = IB_STS                ; VM_CNT = 2 if the instruction is counted
//
// The slow load goes first because VM_CNT decrements in order: the
// instruction cannot be retired from the count before the load returns, which
// takes hundreds of cycles, so a fast-completing instruction cannot be
// mistaken for an uncounted one.
//===----------------------------------------------------------------------===//

// CDNA3/CDNA4 ISA, Table 20 (IB_STS): VM_CNT is bits 23:22 (high) and 3:0.
static unsigned vmCnt(unsigned IbSts) {
  return (((IbSts >> 22) & 0x3) << 4) | (IbSts & 0xf);
}

template <int I, int E>
__global__ void vmcntKernel(unsigned *Before, unsigned *After,
                            const unsigned *Buf, size_t StrideDwords) {
  int GlobalWave =
      blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  const unsigned *Addr = Buf + (size_t)GlobalWave * StrideDwords;
  unsigned Dst, Dst2, B, A;
  unsigned long long Save;
  // The whole sequence, with EXEC_SET and OP filled in per case below.
#define VMCNT(EXEC_SET, OP)                                                    \
  asm volatile("s_waitcnt vmcnt(0)\n\t"                                        \
               "global_load_dword %[dst], %[addr], off sc0 sc1\n\t"            \
               "s_getreg_b32 %[before], hwreg(HW_REG_IB_STS)\n\t"              \
               "s_mov_b64 %[save], exec\n\t" EXEC_SET "\n\t" OP "\n\t"         \
               "s_mov_b64 exec, %[save]\n\t"                                   \
               "s_getreg_b32 %[after], hwreg(HW_REG_IB_STS)\n\t"               \
               "s_waitcnt vmcnt(0)"                                            \
               : [dst] "=&v"(Dst), [dst2] "=&v"(Dst2), [before] "=&s"(B),      \
                 [after] "=&s"(A), [save] "=&s"(Save)                          \
               : [addr] "v"(Addr)                                              \
               : "memory")
  if constexpr (I == NOP && E == EXEC_ALL)
    VMCNT("s_mov_b64 exec, %[save]", "s_nop 0");
  else if constexpr (I == NOP && E == EXEC_ZERO)
    VMCNT("s_mov_b64 exec, 0", "s_nop 0");
  else if constexpr (I == INV && E == EXEC_ALL)
    VMCNT("s_mov_b64 exec, %[save]", "buffer_inv");
  else if constexpr (I == INV && E == EXEC_ZERO)
    VMCNT("s_mov_b64 exec, 0", "buffer_inv");
  else if constexpr (I == LOAD && E == EXEC_ALL)
    VMCNT("s_mov_b64 exec, %[save]",
          "global_load_dword %[dst2], %[addr], off offset:64 sc0 sc1");
  else if constexpr (I == LOAD && E == EXEC_ZERO)
    VMCNT("s_mov_b64 exec, 0",
          "global_load_dword %[dst2], %[addr], off offset:64 sc0 sc1");
  else
    static_assert(Unsupported<I>, "unsupported instruction or EXEC mode");
#undef VMCNT
  (void)Dst;
  (void)Dst2;
  if (threadIdx.x % warpSize == 0) {
    Before[GlobalWave] = B;
    After[GlobalWave] = A;
  }
}

template <int I, int E> static void runVmcnt(int Blocks, int Reps) {
  constexpr int WavesPerBlock = 4;
  int Waves = Blocks * WavesPerBlock;
  // Every wave loads its own line, 16 KiB apart, from a freshly written
  // buffer, so the load stays outstanding for hundreds of cycles.
  constexpr size_t StrideDwords = 4096;
  size_t Dwords = (size_t)Waves * StrideDwords;
  unsigned *Buf, *DBefore, *DAfter;
  HIP_CHECK(hipMalloc(&Buf, Dwords * sizeof(unsigned)));
  HIP_CHECK(hipMalloc(&DBefore, Waves * sizeof(unsigned)));
  HIP_CHECK(hipMalloc(&DAfter, Waves * sizeof(unsigned)));
  std::vector<unsigned> HBefore(Waves), HAfter(Waves);
  for (int Rep = 0; Rep < Reps; ++Rep) {
    HIP_CHECK(hipMemset(Buf, 0xab, Dwords * sizeof(unsigned)));
    HIP_CHECK(hipDeviceSynchronize());
    vmcntKernel<I, E><<<Blocks, 64 * WavesPerBlock>>>(DBefore, DAfter, Buf,
                                                      StrideDwords);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(HBefore.data(), DBefore, Waves * sizeof(unsigned),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(HAfter.data(), DAfter, Waves * sizeof(unsigned),
                        hipMemcpyDeviceToHost));
    std::vector<unsigned> VB, VA;
    unsigned Incremented = 0;
    for (int W = 0; W < Waves; ++W) {
      VB.push_back(vmCnt(HBefore[W]));
      VA.push_back(vmCnt(HAfter[W]));
      Incremented += VA.back() == VB.back() + 1;
    }
    std::nth_element(VB.begin(), VB.begin() + Waves / 2, VB.end());
    std::nth_element(VA.begin(), VA.begin() + Waves / 2, VA.end());
    const char *Inst = instName(I), *Ex = execName(E);
    printf("vmcnt,%s,%s,none,%d,vmcnt_before,%u\n", Inst, Ex, Rep,
           VB[Waves / 2]);
    printf("vmcnt,%s,%s,none,%d,vmcnt_after,%u\n", Inst, Ex, Rep,
           VA[Waves / 2]);
    printf("vmcnt,%s,%s,none,%d,waves_incremented,%u\n", Inst, Ex, Rep,
           Incremented);
    printf("vmcnt,%s,%s,none,%d,waves,%d\n", Inst, Ex, Rep, Waves);
  }
  HIP_CHECK(hipFree(Buf));
  HIP_CHECK(hipFree(DBefore));
  HIP_CHECK(hipFree(DAfter));
}

//===----------------------------------------------------------------------===//
// Tests "issue" and "latency"
//
// issue:   16 back-to-back instructions per loop iteration, one
//          s_waitcnt vmcnt(0) at the very end.
// latency: each instruction followed by s_waitcnt vmcnt(0).
//===----------------------------------------------------------------------===//

constexpr int OpsPerIter = 16; // Must match the `.rept 16` below.

template <int I, int E>
__global__ void issueKernel(unsigned long long *Cycles, unsigned long long *Real,
                            int Iters) {
  Timer T;
  T.start();
  for (int It = 0; It < Iters; ++It) {
    unsigned long long Save;
    // 16 copies of OP inside the EXEC save/restore.
#define ISSUE(EXEC_SET, OP)                                                    \
  asm volatile("s_mov_b64 %[save], exec\n\t" EXEC_SET "\n\t"                   \
               ".rept 16\n\t" OP "\n\t"                                        \
               ".endr\n\t"                                                     \
               "s_mov_b64 exec, %[save]"                                       \
               : [save] "=&s"(Save)::"memory")
    if constexpr (I == NOP && E == EXEC_ALL)
      ISSUE("s_mov_b64 exec, %[save]", "s_nop 0");
    else if constexpr (I == INV && E == EXEC_ALL)
      ISSUE("s_mov_b64 exec, %[save]", "buffer_inv");
    else if constexpr (I == INV && E == EXEC_ZERO)
      ISSUE("s_mov_b64 exec, 0", "buffer_inv");
    else if constexpr (I == INV_SC0 && E == EXEC_ALL)
      ISSUE("s_mov_b64 exec, %[save]", "buffer_inv sc0");
    else
      static_assert(Unsupported<I>, "unsupported instruction or EXEC mode");
#undef ISSUE
  }
  waitVmcnt0();
  T.stop(Cycles, Real, blockIdx.x);
}

template <int I, int E>
__global__ void latencyKernel(unsigned long long *Cycles,
                              unsigned long long *Real, int Iters) {
  Timer T;
  T.start();
  for (int It = 0; It < Iters; ++It) {
    unsigned long long Save;
    // 16 copies of OP, each followed by s_waitcnt vmcnt(0), inside the EXEC
    // save/restore. An empty OP measures the wait alone.
#define LATENCY(EXEC_SET, OP)                                                  \
  asm volatile("s_mov_b64 %[save], exec\n\t" EXEC_SET "\n\t"                   \
               ".rept 16\n\t" OP "\n\t"                                        \
               "s_waitcnt vmcnt(0)\n\t"                                        \
               ".endr\n\t"                                                     \
               "s_mov_b64 exec, %[save]"                                       \
               : [save] "=&s"(Save)::"memory")
    if constexpr (I == NONE && E == EXEC_ALL)
      LATENCY("s_mov_b64 exec, %[save]", "");
    else if constexpr (I == NOP && E == EXEC_ALL)
      LATENCY("s_mov_b64 exec, %[save]", "s_nop 0");
    else if constexpr (I == INV && E == EXEC_ALL)
      LATENCY("s_mov_b64 exec, %[save]", "buffer_inv");
    else if constexpr (I == INV && E == EXEC_ZERO)
      LATENCY("s_mov_b64 exec, 0", "buffer_inv");
    else if constexpr (I == INV_SC0 && E == EXEC_ALL)
      LATENCY("s_mov_b64 exec, %[save]", "buffer_inv sc0");
    else
      static_assert(Unsupported<I>, "unsupported instruction or EXEC mode");
#undef LATENCY
  }
  T.stop(Cycles, Real, blockIdx.x);
}

//===----------------------------------------------------------------------===//
// Tests "cache" and "contention"
//===----------------------------------------------------------------------===//

// Each wave owns LoadsPerIter * 64 dwords (LoadsPerIter * 256 bytes), small
// enough to stay resident in the vector L1 cache.
constexpr int LoadsPerIter = 8;
constexpr int DwordsPerWave = LoadsPerIter * 64;

// With BypassL1, the loads use sc1 (agent scope), which bypasses the vector L1
// cache. That gives the cost of the same loop when every load misses L1.
template <bool BypassL1 = false>
__device__ __forceinline__ void loadFootprint(const unsigned *Base) {
  const unsigned *P = Base + (threadIdx.x % warpSize);
  unsigned V[LoadsPerIter];
#pragma unroll
  for (int K = 0; K < LoadsPerIter; ++K) {
    if constexpr (BypassL1)
      asm volatile("global_load_dword %0, %1, off sc1"
                   : "=v"(V[K])
                   : "v"(P + K * 64)
                   : "memory");
    else
      asm volatile("global_load_dword %0, %1, off"
                   : "=v"(V[K])
                   : "v"(P + K * 64)
                   : "memory");
  }
  waitVmcnt0();
  // Keep the loaded registers live until after the wait.
#pragma unroll
  for (int K = 0; K < LoadsPerIter; ++K)
    asm volatile("" ::"v"(V[K]));
}

// cache: reload an L1-resident footprint each iteration, with instruction I
// issued before the reload. If I invalidates L1, the reloads miss and the
// iteration gets slower; the BypassL1 variant (sc1 loads) is the L1-miss
// reference. With BypassL1 and I = buffer_inv, the reloads hit L2, so this
// also checks whether buffer_inv invalidates L2.
//
// contention: the same loader loop with I = NONE, while other waves of the
// same block (and therefore the same CU) issue Aggressor in a loop. Paced
// aggressors follow each instruction with `s_sleep 1` (about 64 cycles).
// Pacing is a template parameter so the unpaced loop is truly back-to-back.
template <int I, int Aggressor, bool Paced, bool BypassL1>
__global__ void cacheKernel(unsigned long long *Cycles, unsigned long long *Real,
                            const unsigned *Buf, int Iters,
                            int LoaderWavesPerBlock) {
  int Wave = threadIdx.x / warpSize;
  if (Wave >= LoaderWavesPerBlock) {
    // Aggressor wave: keep issuing for longer than the loaders run.
    for (int It = 0; It < Iters * 4; ++It) {
#pragma unroll
      for (int J = 0; J < OpsPerIter; ++J) {
        op<Aggressor>();
        if constexpr (Paced)
          asm volatile("s_sleep 1");
      }
    }
    waitVmcnt0();
    return;
  }

  int GlobalWave = blockIdx.x * LoaderWavesPerBlock + Wave;
  const unsigned *Base = Buf + (size_t)GlobalWave * DwordsPerWave;
  loadFootprint(Base); // Warm L1.

  Timer T;
  T.start();
  for (int It = 0; It < Iters; ++It) {
    op<I>();
    loadFootprint<BypassL1>(Base);
  }
  T.stop(Cycles, Real, GlobalWave);
}

//===----------------------------------------------------------------------===//
// Host harness
//===----------------------------------------------------------------------===//

struct Options {
  int Reps = 5;
  int Iters = 1024;
  int Blocks = 0; // 0 = number of CUs.
};

// With --manifest, one line per timed kernel launch, in launch order, so a
// profiler's per-dispatch records (e.g. rocprofv3 hardware counters) can be
// matched to tests. counters.py uses this.
static FILE *Manifest = nullptr;

struct TestId {
  const char *Kernel, *Test;
  int I, E, Aggressor;
};

// Rep -1 is the warm-up launch.
static void logLaunch(const TestId &Id, int Rep, int Waves,
                      double OpsPerWave) {
  if (!Manifest)
    return;
  fprintf(Manifest, "%s,%s,%s,%s,%s,%d,%d,%.0f\n", Id.Kernel, Id.Test,
          instName(Id.I), execName(Id.E), instName(Id.Aggressor), Rep, Waves,
          OpsPerWave);
}

static void emitTiming(const TestId &Id, int Rep,
                       const std::vector<unsigned long long> &Cycles,
                       const std::vector<unsigned long long> &Real,
                       double OpsPerSample) {
  // Report the median wave for this rep.
  std::vector<double> C, R;
  for (size_t S = 0; S < Cycles.size(); ++S) {
    C.push_back(Cycles[S] / OpsPerSample);
    R.push_back(Real[S] * 10.0 / OpsPerSample); // 100 MHz -> 10 ns per tick.
  }
  std::nth_element(C.begin(), C.begin() + C.size() / 2, C.end());
  std::nth_element(R.begin(), R.begin() + R.size() / 2, R.end());
  const char *Agg = instName(Id.Aggressor);
  printf("%s,%s,%s,%s,%d,cycles,%.3f\n", Id.Test, instName(Id.I),
         execName(Id.E), Agg, Rep, C[C.size() / 2]);
  printf("%s,%s,%s,%s,%d,ns,%.3f\n", Id.Test, instName(Id.I), execName(Id.E),
         Agg, Rep, R[R.size() / 2]);
}

template <typename LaunchFn>
static void runTimed(const TestId &Id, int Slots, double OpsPerSample,
                     const Options &O, LaunchFn Launch) {
  unsigned long long *DCycles, *DReal;
  HIP_CHECK(hipMalloc(&DCycles, Slots * sizeof(unsigned long long)));
  HIP_CHECK(hipMalloc(&DReal, Slots * sizeof(unsigned long long)));
  std::vector<unsigned long long> Cycles(Slots), Real(Slots);
  logLaunch(Id, -1, Slots, OpsPerSample);
  Launch(DCycles, DReal); // Warm-up launch, not reported.
  HIP_CHECK(hipDeviceSynchronize());
  for (int Rep = 0; Rep < O.Reps; ++Rep) {
    logLaunch(Id, Rep, Slots, OpsPerSample);
    Launch(DCycles, DReal);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(Cycles.data(), DCycles, Slots * sizeof(Cycles[0]),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(Real.data(), DReal, Slots * sizeof(Real[0]),
                        hipMemcpyDeviceToHost));
    emitTiming(Id, Rep, Cycles, Real, OpsPerSample);
  }
  HIP_CHECK(hipFree(DCycles));
  HIP_CHECK(hipFree(DReal));
}

template <int I, int E> static void runIssue(const Options &O) {
  int B = O.Blocks;
  runTimed({"issueKernel", "issue", I, E, NONE}, B, (double)O.Iters * OpsPerIter, O,
           [&](auto *C, auto *R) {
             issueKernel<I, E><<<B, 64>>>(C, R, O.Iters);
           });
}

template <int I, int E> static void runLatency(const Options &O) {
  int B = O.Blocks;
  runTimed({"latencyKernel", "latency", I, E, NONE}, B, (double)O.Iters * OpsPerIter, O,
           [&](auto *C, auto *R) {
             latencyKernel<I, E><<<B, 64>>>(C, R, O.Iters);
           });
}

template <int I, int Aggressor, bool Paced = false, bool BypassL1 = false>
static void runCache(const char *Test, const Options &O, const unsigned *Buf,
                     int LoaderWaves, int TotalWaves) {
  int B = O.Blocks;
  runTimed({"cacheKernel", Test, I, EXEC_ALL, Aggressor}, B * LoaderWaves, O.Iters, O,
           [&](auto *C, auto *R) {
             cacheKernel<I, Aggressor, Paced, BypassL1>
                 <<<B, 64 * TotalWaves>>>(C, R, Buf, O.Iters, LoaderWaves);
           });
}

int main(int Argc, char **Argv) {
  Options O;
  for (int A = 1; A < Argc; ++A) {
    if (!strcmp(Argv[A], "--reps") && A + 1 < Argc)
      O.Reps = atoi(Argv[++A]);
    else if (!strcmp(Argv[A], "--iters") && A + 1 < Argc)
      O.Iters = atoi(Argv[++A]);
    else if (!strcmp(Argv[A], "--blocks") && A + 1 < Argc)
      O.Blocks = atoi(Argv[++A]);
    else if (!strcmp(Argv[A], "--manifest") && A + 1 < Argc) {
      Manifest = fopen(Argv[++A], "w");
      if (!Manifest) {
        perror(Argv[A]);
        return 1;
      }
      fprintf(Manifest,
              "kernel,test,instruction,exec,aggressor,rep,waves,ops_per_wave\n");
    } else {
      fprintf(stderr,
              "usage: %s [--reps N] [--iters N] [--blocks N] "
              "[--manifest FILE]\n",
              Argv[0]);
      return 2;
    }
  }

  hipDeviceProp_t Prop;
  HIP_CHECK(hipGetDeviceProperties(&Prop, 0));
  if (Prop.warpSize != 64) {
    fprintf(stderr, "expected wave64, got warpSize=%d\n", Prop.warpSize);
    return 1;
  }
  if (!O.Blocks)
    O.Blocks = Prop.multiProcessorCount;
  fprintf(stderr, "device: %s (%s), CUs: %d, blocks: %d, iters: %d, reps: %d\n",
          Prop.name, Prop.gcnArchName, Prop.multiProcessorCount, O.Blocks,
          O.Iters, O.Reps);

  printf("test,instruction,exec,aggressor,rep,metric,value\n");

  // vmcnt: buffer_inv with all lanes and with EXEC=0, against s_nop (never
  // counted) and global_load (always counted).
  runVmcnt<NOP, EXEC_ALL>(O.Blocks, O.Reps);
  runVmcnt<NOP, EXEC_ZERO>(O.Blocks, O.Reps);
  runVmcnt<INV, EXEC_ALL>(O.Blocks, O.Reps);
  runVmcnt<INV, EXEC_ZERO>(O.Blocks, O.Reps);
  runVmcnt<LOAD, EXEC_ALL>(O.Blocks, O.Reps);
  runVmcnt<LOAD, EXEC_ZERO>(O.Blocks, O.Reps);

  // issue and latency: buffer_inv with all lanes and with EXEC=0, against
  // s_nop and buffer_inv sc0. Latency also reports `none` (only the wait).
  runIssue<NOP, EXEC_ALL>(O);
  runIssue<INV, EXEC_ALL>(O);
  runIssue<INV, EXEC_ZERO>(O);
  runIssue<INV_SC0, EXEC_ALL>(O);
  runLatency<NONE, EXEC_ALL>(O);
  runLatency<NOP, EXEC_ALL>(O);
  runLatency<INV, EXEC_ALL>(O);
  runLatency<INV, EXEC_ZERO>(O);
  runLatency<INV_SC0, EXEC_ALL>(O);

  // cache: one loader wave per block. contention: four loader waves plus four
  // aggressor waves per block.
  constexpr int ContentionLoaders = 4, ContentionWaves = 8;
  size_t Dwords = (size_t)O.Blocks * ContentionLoaders * DwordsPerWave;
  unsigned *Coarse, *Fine;
  HIP_CHECK(hipMalloc(&Coarse, Dwords * sizeof(unsigned)));
  HIP_CHECK(hipMemset(Coarse, 0, Dwords * sizeof(unsigned)));
  // The same footprint in fine-grained memory, in case the instruction's
  // effect on L1 differs by memory type.
  HIP_CHECK(hipExtMallocWithFlags((void **)&Fine, Dwords * sizeof(unsigned),
                                  hipDeviceMallocFinegrained));
  HIP_CHECK(hipMemset(Fine, 0, Dwords * sizeof(unsigned)));

  struct Mem {
    const char *Test, *BypassTest;
    const unsigned *Buf;
  };
  for (const Mem &M : {Mem{"cache[coarse]", "cache[coarse+bypass-l1]", Coarse},
                       Mem{"cache[fine]", "cache[fine+bypass-l1]", Fine}}) {
    runCache<NONE, NONE>(M.Test, O, M.Buf, 1, 1);
    runCache<NONE, NONE, false, true>(M.BypassTest, O, M.Buf, 1, 1);
    runCache<INV, NONE>(M.Test, O, M.Buf, 1, 1);
    runCache<INV, NONE, false, true>(M.BypassTest, O, M.Buf, 1, 1);
    runCache<INV_SC0, NONE>(M.Test, O, M.Buf, 1, 1);
    runCache<INV_SC1, NONE>(M.Test, O, M.Buf, 1, 1);
  }

  runCache<NONE, NOP>("contention", O, Coarse, ContentionLoaders,
                      ContentionWaves);
  runCache<NONE, INV>("contention", O, Coarse, ContentionLoaders,
                      ContentionWaves);
  runCache<NONE, INV_SC0>("contention", O, Coarse, ContentionLoaders,
                          ContentionWaves);
  runCache<NONE, NOP, true>("contention-paced", O, Coarse, ContentionLoaders,
                            ContentionWaves);
  runCache<NONE, INV, true>("contention-paced", O, Coarse, ContentionLoaders,
                            ContentionWaves);

  HIP_CHECK(hipFree(Fine));
  HIP_CHECK(hipFree(Coarse));
  if (Manifest)
    fclose(Manifest);
  return 0;
}
BUFFER_INV_BENCH_EOF_BENCH_HIP
cat > "$dir/run.py" <<'BUFFER_INV_BENCH_EOF_RUN_PY'
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
BUFFER_INV_BENCH_EOF_RUN_PY
cat > "$dir/counters.py" <<'BUFFER_INV_BENCH_EOF_COUNTERS_PY'
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
BUFFER_INV_BENCH_EOF_COUNTERS_PY
cat > "$dir/README.md" <<'BUFFER_INV_BENCH_EOF_README_MD'
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
BUFFER_INV_BENCH_EOF_README_MD
chmod +x "$dir/run.py" "$dir/counters.py"
# Run from the caller's directory so relative paths (e.g. --csv) resolve there.
exec python3 "$dir/run.py" "$@"
