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
//   cache       Does it invalidate the vector L1 or L2? Untimed: run.py reads
//               L1 (TCP) and L2 (TCC) hardware counters for these launches
//               with rocprofv3.
//   contention  Does it slow other waves on the same CU? Loader waves doing
//               vector memory loads, LDS reads, or MFMAs are timed while
//               other waves issue the instruction.
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

// cache: the access pattern for checking, with hardware counters, whether
// instruction I invalidates L1 or L2. Not timed. One wave per block: warm the
// wave's 2 KiB footprint, then each iteration issue I and reload the
// footprint. With plain loads the reloads hit L1, unless I invalidated it.
// With BypassL1 (sc1 loads) the reloads skip L1 and hit L2, unless I
// invalidated L2.
template <int I, bool BypassL1>
__global__ void cacheKernel(const unsigned *Buf, int Iters) {
  const unsigned *Base = Buf + (size_t)blockIdx.x * DwordsPerWave;
  loadFootprint(Base); // Warm L1 and L2.
  for (int It = 0; It < Iters; ++It) {
    op<I>();
    loadFootprint<BypassL1>(Base);
  }
}

// What the timed loader waves of the contention test do each iteration.
enum Work { WORK_VMEM, WORK_LDS, WORK_MFMA };

static const char *workName(int W) {
  switch (W) {
  case WORK_VMEM: return "vmem";
  case WORK_LDS: return "lds";
  case WORK_MFMA: return "mfma";
  }
  return "?";
}

constexpr int MaxLoaderWaves = 4;

typedef _Float16 half4 __attribute__((ext_vector_type(4)));
typedef float float16 __attribute__((ext_vector_type(16)));
typedef __attribute__((address_space(3))) const unsigned LdsUint;

// contention: the first LoaderWavesPerBlock waves of each block are timed
// doing Work each iteration:
//   WORK_VMEM: 8 global loads of an L1-resident footprint, s_waitcnt vmcnt(0).
//   WORK_LDS:  8 ds_read_b32 from the wave's own LDS region,
//              s_waitcnt lgkmcnt(0).
//   WORK_MFMA: 8 dependent v_mfma_f32_32x32x8_f16. These use the builtin, not
//              inline asm, so the compiler inserts the wait states that
//              dependent MFMAs need.
// The other waves of the same block (and therefore the same CU) issue
// Aggressor in a loop. Paced aggressors follow each instruction with
// `s_sleep 1` (about 64 cycles). Pacing is a template parameter so the unpaced
// loop is truly back-to-back.
template <int Work, int Aggressor, bool Paced>
__global__ void contentionKernel(unsigned long long *Cycles,
                                 unsigned long long *Real, const unsigned *Buf,
                                 int Iters, int LoaderWavesPerBlock) {
  int Wave = threadIdx.x / warpSize;
  int Lane = threadIdx.x % warpSize;
  if (Wave >= LoaderWavesPerBlock) {
    // Aggressor wave: keep issuing for well over the time any loader workload
    // takes, so the loaders are measured under contention throughout.
    for (int It = 0; It < Iters * 16; ++It) {
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
  Timer T;
  if constexpr (Work == WORK_VMEM) {
    const unsigned *Base = Buf + (size_t)GlobalWave * DwordsPerWave;
    loadFootprint(Base); // Warm L1.
    T.start();
    for (int It = 0; It < Iters; ++It)
      loadFootprint(Base);
  } else if constexpr (Work == WORK_LDS) {
    // The values don't matter, so the LDS is left uninitialized.
    __shared__ unsigned Lds[MaxLoaderWaves * DwordsPerWave];
    LdsUint *P = (LdsUint *)&Lds[Wave * DwordsPerWave] + Lane;
    T.start();
    for (int It = 0; It < Iters; ++It) {
      unsigned V[LoadsPerIter];
#pragma unroll
      for (int K = 0; K < LoadsPerIter; ++K)
        asm volatile("ds_read_b32 %0, %1"
                     : "=v"(V[K])
                     : "v"(P + K * 64)
                     : "memory");
      asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
#pragma unroll
      for (int K = 0; K < LoadsPerIter; ++K)
        asm volatile("" ::"v"(V[K]));
    }
  } else if constexpr (Work == WORK_MFMA) {
    half4 A = {(_Float16)Lane, 1, 2, 3};
    float16 Acc = {};
    T.start();
    // Make A depend on an asm statement after the start timer, so the
    // compiler cannot move the MFMAs before it.
    asm volatile("" : "+v"(A));
    for (int It = 0; It < Iters; ++It) {
#pragma unroll
      for (int K = 0; K < LoadsPerIter; ++K)
        Acc = __builtin_amdgcn_mfma_f32_32x32x8f16(A, A, Acc, 0, 0, 0);
    }
    // Consume the result before the stop timer, so the MFMAs cannot move
    // after it.
    asm volatile("" ::"v"(Acc[0]));
  } else {
    static_assert(Unsupported<Work>, "unsupported loader work");
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
  // Comma-separated tests to run.
  std::string Tests = "vmcnt,issue,latency,cache,contention";
};

static bool runs(const Options &O, const char *Test) {
  std::string List = "," + O.Tests + ",";
  return List.find(std::string(",") + Test + ",") != std::string::npos;
}

// With --manifest, one line per kernel launch, in launch order, so a
// profiler's per-dispatch records (rocprofv3 hardware counters) can be matched
// to tests. run.py uses this for the cache test.
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

template <int I, bool BypassL1>
static void runCache(const char *Test, const Options &O, const unsigned *Buf) {
  TestId Id{"cacheKernel", Test, I, EXEC_ALL, NONE};
  for (int Rep = -1; Rep < O.Reps; ++Rep) { // Rep -1 is the warm-up launch.
    logLaunch(Id, Rep, O.Blocks, O.Iters);
    cacheKernel<I, BypassL1><<<O.Blocks, 64>>>(Buf, O.Iters);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
  }
}

template <int Work, int Aggressor, bool Paced>
static void runContention(const Options &O, const unsigned *Buf,
                          int LoaderWaves, int TotalWaves) {
  // The test name carries the loader work and pacing, e.g. contention[lds].
  static std::string Name;
  Name = std::string(Paced ? "contention-paced[" : "contention[") +
         workName(Work) + "]";
  int B = O.Blocks;
  runTimed({"contentionKernel", Name.c_str(), NONE, EXEC_ALL, Aggressor},
           B * LoaderWaves, O.Iters, O, [&](auto *C, auto *R) {
             contentionKernel<Work, Aggressor, Paced>
                 <<<B, 64 * TotalWaves>>>(C, R, Buf, O.Iters, LoaderWaves);
           });
}

// All contention cases for one loader workload: s_nop and buffer_inv
// aggressors, back-to-back and paced.
template <int Work>
static void runContentionWork(const Options &O, const unsigned *Buf,
                              int LoaderWaves, int TotalWaves) {
  runContention<Work, NOP, false>(O, Buf, LoaderWaves, TotalWaves);
  runContention<Work, INV, false>(O, Buf, LoaderWaves, TotalWaves);
  runContention<Work, NOP, true>(O, Buf, LoaderWaves, TotalWaves);
  runContention<Work, INV, true>(O, Buf, LoaderWaves, TotalWaves);
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
    else if (!strcmp(Argv[A], "--tests") && A + 1 < Argc)
      O.Tests = Argv[++A];
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
              "[--tests vmcnt,issue,latency,cache,contention] "
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
  if (runs(O, "vmcnt")) {
    runVmcnt<NOP, EXEC_ALL>(O.Blocks, O.Reps);
    runVmcnt<NOP, EXEC_ZERO>(O.Blocks, O.Reps);
    runVmcnt<INV, EXEC_ALL>(O.Blocks, O.Reps);
    runVmcnt<INV, EXEC_ZERO>(O.Blocks, O.Reps);
    runVmcnt<LOAD, EXEC_ALL>(O.Blocks, O.Reps);
    runVmcnt<LOAD, EXEC_ZERO>(O.Blocks, O.Reps);
  }

  // issue and latency: buffer_inv with all lanes and with EXEC=0, against
  // s_nop and buffer_inv sc0. Latency also reports `none` (only the wait).
  if (runs(O, "issue")) {
    runIssue<NOP, EXEC_ALL>(O);
    runIssue<INV, EXEC_ALL>(O);
    runIssue<INV, EXEC_ZERO>(O);
    runIssue<INV_SC0, EXEC_ALL>(O);
  }
  if (runs(O, "latency")) {
    runLatency<NONE, EXEC_ALL>(O);
    runLatency<NOP, EXEC_ALL>(O);
    runLatency<INV, EXEC_ALL>(O);
    runLatency<INV, EXEC_ZERO>(O);
    runLatency<INV_SC0, EXEC_ALL>(O);
  }

  // cache: one wave per block. contention: four loader waves plus four
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

  // cache: plain reloads alone (they should hit L1), after buffer_inv (does it
  // invalidate L1?), after buffer_inv sc1 (positive control: it does), and
  // L1-bypassing reloads after buffer_inv (does it invalidate L2?).
  if (runs(O, "cache")) {
    runCache<NONE, false>("cache[coarse]", O, Coarse);
    runCache<INV, false>("cache[coarse]", O, Coarse);
    runCache<INV_SC1, false>("cache[coarse]", O, Coarse);
    runCache<INV, true>("cache[coarse+bypass-l1]", O, Coarse);
    runCache<NONE, false>("cache[fine]", O, Fine);
    runCache<INV, false>("cache[fine]", O, Fine);
    runCache<INV_SC1, false>("cache[fine]", O, Fine);
    runCache<INV, true>("cache[fine+bypass-l1]", O, Fine);
  }

  if (runs(O, "contention")) {
    static_assert(ContentionLoaders <= MaxLoaderWaves, "LDS sized for 4");
    runContentionWork<WORK_VMEM>(O, Coarse, ContentionLoaders,
                                 ContentionWaves);
    runContentionWork<WORK_LDS>(O, Coarse, ContentionLoaders,
                                ContentionWaves);
    runContentionWork<WORK_MFMA>(O, Coarse, ContentionLoaders,
                                 ContentionWaves);
  }

  HIP_CHECK(hipFree(Fine));
  HIP_CHECK(hipFree(Coarse));
  if (Manifest)
    fclose(Manifest);
  return 0;
}
