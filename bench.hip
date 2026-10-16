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
  if constexpr (I == NOP && E == EXEC_ALL)
    asm volatile("s_waitcnt vmcnt(0)\n\t"
                 "global_load_dword %[dst], %[addr], off sc0 sc1\n\t"
                 "s_getreg_b32 %[before], hwreg(HW_REG_IB_STS)\n\t"
                 "s_mov_b64 %[save], exec\n\t"
                 "s_mov_b64 exec, %[save]\n\t"
                 "s_nop 0\n\t"
                 "s_mov_b64 exec, %[save]\n\t"
                 "s_getreg_b32 %[after], hwreg(HW_REG_IB_STS)\n\t"
                 "s_waitcnt vmcnt(0)"
                 : [dst] "=&v"(Dst), [dst2] "=&v"(Dst2), [before] "=&s"(B),
                   [after] "=&s"(A), [save] "=&s"(Save)
                 : [addr] "v"(Addr)
                 : "memory");
  else if constexpr (I == NOP && E == EXEC_ZERO)
    asm volatile("s_waitcnt vmcnt(0)\n\t"
                 "global_load_dword %[dst], %[addr], off sc0 sc1\n\t"
                 "s_getreg_b32 %[before], hwreg(HW_REG_IB_STS)\n\t"
                 "s_mov_b64 %[save], exec\n\t"
                 "s_mov_b64 exec, 0\n\t"
                 "s_nop 0\n\t"
                 "s_mov_b64 exec, %[save]\n\t"
                 "s_getreg_b32 %[after], hwreg(HW_REG_IB_STS)\n\t"
                 "s_waitcnt vmcnt(0)"
                 : [dst] "=&v"(Dst), [dst2] "=&v"(Dst2), [before] "=&s"(B),
                   [after] "=&s"(A), [save] "=&s"(Save)
                 : [addr] "v"(Addr)
                 : "memory");
  else if constexpr (I == INV && E == EXEC_ALL)
    asm volatile("s_waitcnt vmcnt(0)\n\t"
                 "global_load_dword %[dst], %[addr], off sc0 sc1\n\t"
                 "s_getreg_b32 %[before], hwreg(HW_REG_IB_STS)\n\t"
                 "s_mov_b64 %[save], exec\n\t"
                 "s_mov_b64 exec, %[save]\n\t"
                 "buffer_inv\n\t"
                 "s_mov_b64 exec, %[save]\n\t"
                 "s_getreg_b32 %[after], hwreg(HW_REG_IB_STS)\n\t"
                 "s_waitcnt vmcnt(0)"
                 : [dst] "=&v"(Dst), [dst2] "=&v"(Dst2), [before] "=&s"(B),
                   [after] "=&s"(A), [save] "=&s"(Save)
                 : [addr] "v"(Addr)
                 : "memory");
  else if constexpr (I == INV && E == EXEC_ZERO)
    asm volatile("s_waitcnt vmcnt(0)\n\t"
                 "global_load_dword %[dst], %[addr], off sc0 sc1\n\t"
                 "s_getreg_b32 %[before], hwreg(HW_REG_IB_STS)\n\t"
                 "s_mov_b64 %[save], exec\n\t"
                 "s_mov_b64 exec, 0\n\t"
                 "buffer_inv\n\t"
                 "s_mov_b64 exec, %[save]\n\t"
                 "s_getreg_b32 %[after], hwreg(HW_REG_IB_STS)\n\t"
                 "s_waitcnt vmcnt(0)"
                 : [dst] "=&v"(Dst), [dst2] "=&v"(Dst2), [before] "=&s"(B),
                   [after] "=&s"(A), [save] "=&s"(Save)
                 : [addr] "v"(Addr)
                 : "memory");
  else if constexpr (I == LOAD && E == EXEC_ALL)
    asm volatile("s_waitcnt vmcnt(0)\n\t"
                 "global_load_dword %[dst], %[addr], off sc0 sc1\n\t"
                 "s_getreg_b32 %[before], hwreg(HW_REG_IB_STS)\n\t"
                 "s_mov_b64 %[save], exec\n\t"
                 "s_mov_b64 exec, %[save]\n\t"
                 "global_load_dword %[dst2], %[addr], off offset:64 sc0 sc1\n\t"
                 "s_mov_b64 exec, %[save]\n\t"
                 "s_getreg_b32 %[after], hwreg(HW_REG_IB_STS)\n\t"
                 "s_waitcnt vmcnt(0)"
                 : [dst] "=&v"(Dst), [dst2] "=&v"(Dst2), [before] "=&s"(B),
                   [after] "=&s"(A), [save] "=&s"(Save)
                 : [addr] "v"(Addr)
                 : "memory");
  else if constexpr (I == LOAD && E == EXEC_ZERO)
    asm volatile("s_waitcnt vmcnt(0)\n\t"
                 "global_load_dword %[dst], %[addr], off sc0 sc1\n\t"
                 "s_getreg_b32 %[before], hwreg(HW_REG_IB_STS)\n\t"
                 "s_mov_b64 %[save], exec\n\t"
                 "s_mov_b64 exec, 0\n\t"
                 "global_load_dword %[dst2], %[addr], off offset:64 sc0 sc1\n\t"
                 "s_mov_b64 exec, %[save]\n\t"
                 "s_getreg_b32 %[after], hwreg(HW_REG_IB_STS)\n\t"
                 "s_waitcnt vmcnt(0)"
                 : [dst] "=&v"(Dst), [dst2] "=&v"(Dst2), [before] "=&s"(B),
                   [after] "=&s"(A), [save] "=&s"(Save)
                 : [addr] "v"(Addr)
                 : "memory");
  else
    static_assert(Unsupported<I>, "unsupported instruction or EXEC mode");
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
    if constexpr (I == NOP && E == EXEC_ALL)
      asm volatile("s_mov_b64 %[save], exec\n\t"
                   "s_mov_b64 exec, %[save]\n\t"
                   ".rept 16\n\t"
                   "s_nop 0\n\t"
                   ".endr\n\t"
                   "s_mov_b64 exec, %[save]"
                   : [save] "=&s"(Save)::"memory");
    else if constexpr (I == INV && E == EXEC_ALL)
      asm volatile("s_mov_b64 %[save], exec\n\t"
                   "s_mov_b64 exec, %[save]\n\t"
                   ".rept 16\n\t"
                   "buffer_inv\n\t"
                   ".endr\n\t"
                   "s_mov_b64 exec, %[save]"
                   : [save] "=&s"(Save)::"memory");
    else if constexpr (I == INV && E == EXEC_ZERO)
      asm volatile("s_mov_b64 %[save], exec\n\t"
                   "s_mov_b64 exec, 0\n\t"
                   ".rept 16\n\t"
                   "buffer_inv\n\t"
                   ".endr\n\t"
                   "s_mov_b64 exec, %[save]"
                   : [save] "=&s"(Save)::"memory");
    else if constexpr (I == INV_SC0 && E == EXEC_ALL)
      asm volatile("s_mov_b64 %[save], exec\n\t"
                   "s_mov_b64 exec, %[save]\n\t"
                   ".rept 16\n\t"
                   "buffer_inv sc0\n\t"
                   ".endr\n\t"
                   "s_mov_b64 exec, %[save]"
                   : [save] "=&s"(Save)::"memory");
    else
      static_assert(Unsupported<I>, "unsupported instruction or EXEC mode");
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
    if constexpr (I == NONE && E == EXEC_ALL)
      asm volatile("s_mov_b64 %[save], exec\n\t"
                   "s_mov_b64 exec, %[save]\n\t"
                   ".rept 16\n\t"
                   "s_waitcnt vmcnt(0)\n\t"
                   ".endr\n\t"
                   "s_mov_b64 exec, %[save]"
                   : [save] "=&s"(Save)::"memory");
    else if constexpr (I == NOP && E == EXEC_ALL)
      asm volatile("s_mov_b64 %[save], exec\n\t"
                   "s_mov_b64 exec, %[save]\n\t"
                   ".rept 16\n\t"
                   "s_nop 0\n\t"
                   "s_waitcnt vmcnt(0)\n\t"
                   ".endr\n\t"
                   "s_mov_b64 exec, %[save]"
                   : [save] "=&s"(Save)::"memory");
    else if constexpr (I == INV && E == EXEC_ALL)
      asm volatile("s_mov_b64 %[save], exec\n\t"
                   "s_mov_b64 exec, %[save]\n\t"
                   ".rept 16\n\t"
                   "buffer_inv\n\t"
                   "s_waitcnt vmcnt(0)\n\t"
                   ".endr\n\t"
                   "s_mov_b64 exec, %[save]"
                   : [save] "=&s"(Save)::"memory");
    else if constexpr (I == INV && E == EXEC_ZERO)
      asm volatile("s_mov_b64 %[save], exec\n\t"
                   "s_mov_b64 exec, 0\n\t"
                   ".rept 16\n\t"
                   "buffer_inv\n\t"
                   "s_waitcnt vmcnt(0)\n\t"
                   ".endr\n\t"
                   "s_mov_b64 exec, %[save]"
                   : [save] "=&s"(Save)::"memory");
    else if constexpr (I == INV_SC0 && E == EXEC_ALL)
      asm volatile("s_mov_b64 %[save], exec\n\t"
                   "s_mov_b64 exec, %[save]\n\t"
                   ".rept 16\n\t"
                   "buffer_inv sc0\n\t"
                   "s_waitcnt vmcnt(0)\n\t"
                   ".endr\n\t"
                   "s_mov_b64 exec, %[save]"
                   : [save] "=&s"(Save)::"memory");
    else
      static_assert(Unsupported<I>, "unsupported instruction or EXEC mode");
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

struct TestId {
  const char *Test;
  int I, E, Aggressor;
};

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
  Launch(DCycles, DReal); // Warm-up launch, not reported.
  HIP_CHECK(hipDeviceSynchronize());
  for (int Rep = 0; Rep < O.Reps; ++Rep) {
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
  runTimed({"issue", I, E, NONE}, B, (double)O.Iters * OpsPerIter, O,
           [&](auto *C, auto *R) {
             issueKernel<I, E><<<B, 64>>>(C, R, O.Iters);
           });
}

template <int I, int E> static void runLatency(const Options &O) {
  int B = O.Blocks;
  runTimed({"latency", I, E, NONE}, B, (double)O.Iters * OpsPerIter, O,
           [&](auto *C, auto *R) {
             latencyKernel<I, E><<<B, 64>>>(C, R, O.Iters);
           });
}

template <int I, int Aggressor, bool Paced = false, bool BypassL1 = false>
static void runCache(const char *Test, const Options &O, const unsigned *Buf,
                     int LoaderWaves, int TotalWaves) {
  int B = O.Blocks;
  runTimed({Test, I, EXEC_ALL, Aggressor}, B * LoaderWaves, O.Iters, O,
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
    else {
      fprintf(stderr, "usage: %s [--reps N] [--iters N] [--blocks N]\n",
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
  return 0;
}
