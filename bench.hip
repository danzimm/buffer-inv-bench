// Microbenchmarks for the cost of `buffer_inv` (no sc bits) on gfx942/gfx950.
//
// All timed sequences are written as inline asm so the measured instruction
// stream is exact. The compiler's waitcnt insertion does not look inside inline
// asm, so every test issues its own `s_waitcnt`.
//
// Output: CSV on stdout, one row per (test, variant, aggressor, rep):
//   test,variant,aggressor,rep,cycles_per_iter,ns_per_iter
// cycles come from s_memtime (shader clock), ns from s_memrealtime (100 MHz).

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

enum Variant { NONE, NOP, INV, INV_SC0, INV_SC1 };

static const char *variantName(int V) {
  switch (V) {
  case NONE: return "none";
  case NOP: return "s_nop 0";
  case INV: return "buffer_inv";
  case INV_SC0: return "buffer_inv sc0";
  case INV_SC1: return "buffer_inv sc1";
  }
  return "?";
}

template <int V> __device__ __forceinline__ void op() {
  if constexpr (V == NOP)
    asm volatile("s_nop 0" ::: "memory");
  else if constexpr (V == INV)
    asm volatile("buffer_inv" ::: "memory");
  else if constexpr (V == INV_SC0)
    asm volatile("buffer_inv sc0" ::: "memory");
  else if constexpr (V == INV_SC1)
    asm volatile("buffer_inv sc1" ::: "memory");
}

__device__ __forceinline__ void waitVmcnt0() {
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
}

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

constexpr int Unroll = 16;

// Test "issue": back-to-back ops with a single wait at the end. Measures the
// issue cost of the instruction.
template <int V>
__global__ void issueKernel(unsigned long long *Cycles, unsigned long long *Real,
                            int Iters) {
  Timer T;
  T.start();
  for (int I = 0; I < Iters; ++I) {
#pragma unroll
    for (int J = 0; J < Unroll; ++J)
      op<V>();
  }
  waitVmcnt0();
  T.stop(Cycles, Real, blockIdx.x);
}

// Test "latency": each op is followed by `s_waitcnt vmcnt(0)`. Measures the
// round trip until the op is reported complete.
template <int V>
__global__ void latencyKernel(unsigned long long *Cycles,
                              unsigned long long *Real, int Iters) {
  Timer T;
  T.start();
  for (int I = 0; I < Iters; ++I) {
#pragma unroll
    for (int J = 0; J < Unroll; ++J) {
      op<V>();
      waitVmcnt0();
    }
  }
  T.stop(Cycles, Real, blockIdx.x);
}

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
#pragma unroll
  for (int K = 0; K < LoadsPerIter; ++K)
    asm volatile("" ::"v"(V[K]));
}

// Test "cache": reload an L1-resident footprint each iteration, with the op
// issued before the reload. If the op invalidates L1, the reloads miss and
// the iteration gets slower. The BypassL1 variant (sc1 loads) gives the cost
// of the same loop when every load misses L1, as the reference.
//
// With Aggressor != NONE, the block also runs aggressor waves that only issue
// the aggressor op in a loop, to measure the effect on other waves of the CU.
// Paced aggressors follow each op with `s_sleep 1` (about 64 cycles) to model
// an occasional op rather than a flood. Pacing is a template parameter so the
// unpaced loop is truly back-to-back.
template <int V, int Aggressor, bool Paced, bool BypassL1>
__global__ void cacheKernel(unsigned long long *Cycles, unsigned long long *Real,
                            const unsigned *Buf, int Iters,
                            int LoaderWavesPerBlock) {
  int Wave = threadIdx.x / warpSize;
  if (Wave >= LoaderWavesPerBlock) {
    // Aggressor wave: keep issuing the op for longer than the loaders run.
    for (int I = 0; I < Iters * 4; ++I) {
#pragma unroll
      for (int J = 0; J < Unroll; ++J) {
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
  for (int I = 0; I < Iters; ++I) {
    op<V>();
    loadFootprint<BypassL1>(Base);
  }
  T.stop(Cycles, Real, GlobalWave);
}

// Test "vmcnt-exec0": does the op increment vmcnt when EXEC is 0?
//
//   v = sentinel
//   global_load v, slow address        ; vmcnt = 1
//   exec = 0 ; <op> ; exec = saved      ; vmcnt = 2 if the op is counted
//   s_waitcnt vmcnt(1)
//   snapshot = v
//
// If the op incremented vmcnt, vmcnt(1) must wait for the earlier load, so the
// snapshot holds the loaded value. If not, the wait passes immediately and the
// snapshot still holds the sentinel. Writes 1 per wave if the load had
// completed, 0 otherwise.
enum Exec0Variant {
  E0_NOP,         // s_nop with EXEC=0: never counted (control).
  E0_INV,         // buffer_inv with EXEC=0: the question.
  E0_INV_EXEC_ON, // buffer_inv with EXEC unchanged: counted (control).
  E0_LOAD,        // global_load with EXEC=0: reference.
};

static const char *exec0Name(int V) {
  switch (V) {
  case E0_NOP: return "s_nop 0 (EXEC=0)";
  case E0_INV: return "buffer_inv (EXEC=0)";
  case E0_INV_EXEC_ON: return "buffer_inv (EXEC unchanged)";
  case E0_LOAD: return "global_load_dword (EXEC=0)";
  }
  return "?";
}

constexpr unsigned Exec0Magic = 0xabababab; // hipMemset byte 0xab.
constexpr unsigned Exec0Sentinel = 0;

template <int V>
__global__ void exec0Kernel(unsigned *Out, const unsigned *Buf,
                            size_t StrideDwords) {
  int GlobalWave =
      blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  const unsigned *Addr = Buf + (size_t)GlobalWave * StrideDwords;
  unsigned Dst, Snap, Scratch;
  unsigned long long Save;
#define EXEC0_ASM(OP, EXEC_OFF)                                                \
  asm volatile("v_mov_b32 %0, %5\n\t"                                          \
               "global_load_dword %0, %4, off sc0 sc1\n\t"                     \
               "s_mov_b64 %1, exec\n\t" EXEC_OFF OP "\n\t"                     \
               "s_mov_b64 exec, %1\n\t"                                        \
               "s_waitcnt vmcnt(1)\n\t"                                        \
               "v_mov_b32 %2, %0\n\t"                                          \
               "s_waitcnt vmcnt(0)"                                            \
               : "=&v"(Dst), "=&s"(Save), "=&v"(Snap), "=&v"(Scratch)          \
               : "v"(Addr), "v"(Exec0Sentinel)                                 \
               : "memory")
  if constexpr (V == E0_NOP)
    EXEC0_ASM("s_nop 0", "s_mov_b64 exec, 0\n\t");
  else if constexpr (V == E0_INV)
    EXEC0_ASM("buffer_inv", "s_mov_b64 exec, 0\n\t");
  else if constexpr (V == E0_INV_EXEC_ON)
    EXEC0_ASM("buffer_inv", "");
  else if constexpr (V == E0_LOAD)
    EXEC0_ASM("global_load_dword %3, %4, off offset:64",
              "s_mov_b64 exec, 0\n\t");
#undef EXEC0_ASM
  (void)Dst;
  (void)Scratch;
  if (threadIdx.x % warpSize == 0)
    Out[GlobalWave] = Snap == Exec0Magic;
}

template <int V> static void runExec0(int Blocks, int Reps) {
  constexpr int WavesPerBlock = 4;
  int Waves = Blocks * WavesPerBlock;
  // Every wave gets its own line, 16 KiB apart, in a freshly written buffer,
  // so the load stays outstanding for hundreds of cycles.
  constexpr size_t StrideDwords = 4096;
  size_t Dwords = (size_t)Waves * StrideDwords;
  unsigned *Buf, *DOut;
  HIP_CHECK(hipMalloc(&Buf, Dwords * sizeof(unsigned)));
  HIP_CHECK(hipMalloc(&DOut, Waves * sizeof(unsigned)));
  std::vector<unsigned> Out(Waves);
  for (int Rep = 0; Rep < Reps; ++Rep) {
    HIP_CHECK(hipMemset(Buf, 0xab, Dwords * sizeof(unsigned)));
    HIP_CHECK(hipDeviceSynchronize());
    exec0Kernel<V><<<Blocks, 64 * WavesPerBlock>>>(DOut, Buf, StrideDwords);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(Out.data(), DOut, Waves * sizeof(unsigned),
                        hipMemcpyDeviceToHost));
    unsigned Completed = 0;
    for (unsigned X : Out)
      Completed += X;
    // Reuses the CSV schema: cycles_per_iter holds the fraction of waves whose
    // vmcnt(1) waited for the earlier load; ns_per_iter holds the wave count.
    printf("vmcnt-exec0,%s,none,%d,%.4f,%d\n", exec0Name(V), Rep,
           (double)Completed / Waves, Waves);
  }
  HIP_CHECK(hipFree(Buf));
  HIP_CHECK(hipFree(DOut));
}

struct Options {
  int Reps = 5;
  int Iters = 1024;
  int Blocks = 0; // 0 = number of CUs.
};

static void emit(const char *Test, int V, int Aggressor, int Rep,
                 const std::vector<unsigned long long> &Cycles,
                 const std::vector<unsigned long long> &Real,
                 double OpsPerSample) {
  // Report the median sample for this rep.
  std::vector<double> C, R;
  for (size_t I = 0; I < Cycles.size(); ++I) {
    C.push_back(Cycles[I] / OpsPerSample);
    R.push_back(Real[I] * 10.0 / OpsPerSample); // 100 MHz -> 10 ns per tick.
  }
  std::nth_element(C.begin(), C.begin() + C.size() / 2, C.end());
  std::nth_element(R.begin(), R.begin() + R.size() / 2, R.end());
  printf("%s,%s,%s,%d,%.3f,%.3f\n", Test, variantName(V),
         variantName(Aggressor), Rep, C[C.size() / 2], R[R.size() / 2]);
}

template <typename LaunchFn>
static void runTest(const char *Test, int V, int Aggressor, int Slots,
                    double OpsPerSample, const Options &O, LaunchFn Launch) {
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
    emit(Test, V, Aggressor, Rep, Cycles, Real, OpsPerSample);
  }
  HIP_CHECK(hipFree(DCycles));
  HIP_CHECK(hipFree(DReal));
}

template <int V> static void runIssueAndLatency(const Options &O) {
  int B = O.Blocks;
  double Ops = (double)O.Iters * Unroll;
  runTest("issue", V, NONE, B, Ops, O, [&](auto *C, auto *R) {
    issueKernel<V><<<B, 64>>>(C, R, O.Iters);
  });
  runTest("latency", V, NONE, B, Ops, O, [&](auto *C, auto *R) {
    latencyKernel<V><<<B, 64>>>(C, R, O.Iters);
  });
}

template <int V, int Aggressor, bool Paced = false, bool BypassL1 = false>
static void runCache(const char *Test, const Options &O, const unsigned *Buf,
                     int LoaderWaves, int TotalWaves) {
  int B = O.Blocks;
  runTest(Test, V, Aggressor, B * LoaderWaves, O.Iters, O,
          [&](auto *C, auto *R) {
            cacheKernel<V, Aggressor, Paced, BypassL1>
                <<<B, 64 * TotalWaves>>>(C, R, Buf, O.Iters, LoaderWaves);
          });
}

int main(int Argc, char **Argv) {
  Options O;
  for (int I = 1; I < Argc; ++I) {
    if (!strcmp(Argv[I], "--reps") && I + 1 < Argc)
      O.Reps = atoi(Argv[++I]);
    else if (!strcmp(Argv[I], "--iters") && I + 1 < Argc)
      O.Iters = atoi(Argv[++I]);
    else if (!strcmp(Argv[I], "--blocks") && I + 1 < Argc)
      O.Blocks = atoi(Argv[++I]);
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

  printf("test,variant,aggressor,rep,cycles_per_iter,ns_per_iter\n");

  runExec0<E0_NOP>(O.Blocks, O.Reps);
  runExec0<E0_INV>(O.Blocks, O.Reps);
  runExec0<E0_INV_EXEC_ON>(O.Blocks, O.Reps);
  runExec0<E0_LOAD>(O.Blocks, O.Reps);

  runIssueAndLatency<NONE>(O);
  runIssueAndLatency<NOP>(O);
  runIssueAndLatency<INV>(O);
  runIssueAndLatency<INV_SC0>(O);

  // Cache tests: one loader wave per block without aggressors; four loader
  // waves plus four aggressor waves per block with aggressors.
  constexpr int AggLoaders = 4, AggTotal = 8;
  size_t Dwords = (size_t)O.Blocks * AggLoaders * DwordsPerWave;
  unsigned *Buf;
  HIP_CHECK(hipMalloc(&Buf, Dwords * sizeof(unsigned)));
  HIP_CHECK(hipMemset(Buf, 0, Dwords * sizeof(unsigned)));

  // The same footprint in fine-grained memory, in case the instruction's
  // effect on L1 differs by memory type.
  unsigned *FineBuf;
  HIP_CHECK(hipExtMallocWithFlags((void **)&FineBuf, Dwords * sizeof(unsigned),
                                  hipDeviceMallocFinegrained));
  HIP_CHECK(hipMemset(FineBuf, 0, Dwords * sizeof(unsigned)));

  for (auto [Name, B] : {std::pair<const char *, const unsigned *>{
                             "cache[coarse]", Buf},
                         {"cache[fine]", FineBuf}}) {
    runCache<NONE, NONE>(Name, O, B, 1, 1);
    // Calibration: the same reloads with sc1 loads, which bypass L1.
    runCache<NONE, NONE, false, true>(
        Name == std::string("cache[coarse]") ? "cache[coarse+bypass-l1]"
                                             : "cache[fine+bypass-l1]",
        O, B, 1, 1);
    runCache<INV, NONE>(Name, O, B, 1, 1);
    // L2 check: the L1-bypassing reloads hit L2; if buffer_inv invalidated
    // L2, they would go to memory instead.
    runCache<INV, NONE, false, true>(
        Name == std::string("cache[coarse]") ? "cache[coarse+bypass-l1]"
                                             : "cache[fine+bypass-l1]",
        O, B, 1, 1);
    runCache<INV_SC0, NONE>(Name, O, B, 1, 1);
    runCache<INV_SC1, NONE>(Name, O, B, 1, 1);
  }

  runCache<NONE, NOP>("cache+aggressor", O, Buf, AggLoaders, AggTotal);
  runCache<NONE, INV>("cache+aggressor", O, Buf, AggLoaders, AggTotal);
  runCache<NONE, INV_SC0>("cache+aggressor", O, Buf, AggLoaders, AggTotal);

  runCache<NONE, NOP, true>("cache+paced-aggressor", O, Buf, AggLoaders,
                            AggTotal);
  runCache<NONE, INV, true>("cache+paced-aggressor", O, Buf, AggLoaders,
                            AggTotal);

  HIP_CHECK(hipFree(FineBuf));

  HIP_CHECK(hipFree(Buf));
  return 0;
}
