#include "backend.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>

namespace solid {
namespace {

void checked(cudaError_t error, const char* operation) {
  if (error != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

size_t product(size_t a, size_t b) {
  if (b && a > std::numeric_limits<size_t>::max() / b)
    throw std::length_error("CUDA buffer size overflow");
  return a * b;
}

template<class T> struct Buffer {
  T* data = nullptr;
  size_t capacity = 0;
  ~Buffer() { if (data) cudaFree(data); }
  void reserve(size_t count) {
    if (count <= capacity) return;
    T* next = nullptr;
    checked(cudaMalloc(reinterpret_cast<void**>(&next), product(count, sizeof(T))),
            "cudaMalloc");
    if (data) {
      const cudaError_t error = cudaFree(data);
      if (error != cudaSuccess) {
        cudaFree(next);
        checked(error, "cudaFree while growing buffer");
      }
    }
    data = next;
    capacity = count;
  }
};

// Each physical uint32 word holds two 16-bit cells: type2, m3, biased S6.
// Every mixed old/new S is in [-30,18], so (S+32) is in [2,50].
// Valid cells stay in [64,1626]: full-word adds cannot cross a half boundary.
__device__ __forceinline__ int cellType(uint32_t cell) { return int(cell & 3u); }
__device__ __forceinline__ int cellM(uint32_t cell) { return int((cell >> 2) & 7u); }
__device__ __forceinline__ int cellS(uint32_t cell) { return int(cell >> 5) - 32; }
// Ordinary cache loads are confined to CTA-joined evaluation/export phases.
__device__ __forceinline__ uint32_t readCell(const uint32_t* cells, int site) {
  return (cells[site >> 1] >> ((site & 1) * 16)) & 65535u;
}
__device__ __forceinline__ uint32_t AtomicReadCell(uint32_t* cells, int site) {
  const uint32_t word = atomicAdd(cells + (site >> 1), 0u);
  return (word >> ((site & 1) * 16)) & 65535u;
}
__device__ __forceinline__ void AtomicAddCell(uint32_t* cells, int site,
                                           uint32_t delta) {
  atomicAdd(cells + (site >> 1), delta << ((site & 1) * 16));
}
__device__ __forceinline__ int target(int type) { return type == 1 ? 3 : 5; }
__device__ __forceinline__ int q(int type, int m) { return type ? m - target(type) : 0; }
__device__ __forceinline__ int siteEnergy(int type, int m) {
  const int v = q(type, m);
  return v * v;
}
__device__ __forceinline__ uint64_t mixWord(uint64_t x) {
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

struct DeviceStream {
  uint64_t key, cache = 0;
  uint32_t cursor = 0;
  bool have = false;
  __device__ explicit DeviceStream(uint64_t k) : key(k) {}
  __device__ __forceinline__ uint32_t next() {
    const uint32_t pair = cursor >> 1;
    ++cursor;
    if (have) {
      have = false;
      return uint32_t(cache);
    }
    cache = mixWord(key + uint64_t(pair) * 0x9e3779b97f4a7c15ULL);
    have = true;
    return uint32_t(cache >> 32);
  }
  __device__ __forceinline__ uint32_t index(uint32_t n, uint32_t t) {
    for (;;) {
      const uint64_t value = uint64_t(next()) * n;
      if (uint32_t(value) >= t) return uint32_t(value >> 32);
    }
  }
};

// C=0 is the runtime-size fallback; L8 and L21 have constant divisions.
template<int C> struct Geometry {
  int side;
  __device__ explicit Geometry(int l) : side(C ? C : l) {}
  __device__ __forceinline__ int nb(int s, int d) const {
    const int l = C ? C : side;
    const int x = s % l, y = (s / l) % l, z = s / (l * l);
    if (d == 0) return s + (x == l - 1 ? 1 - l : 1);
    if (d == 1) return s + (x == 0 ? l - 1 : -1);
    if (d == 2) return s + (y == l - 1 ? l * (1 - l) : l);
    if (d == 3) return s + (y == 0 ? l * (l - 1) : -l);
    if (d == 4) return s + (z == l - 1 ? l * l * (1 - l) : l * l);
    return s + (z == 0 ? l * l * (l - 1) : -l * l);
  }
  __device__ __forceinline__ uint32_t endpoint(int s) const {
    const int l = C ? C : side;
    const uint32_t coords = uint32_t(s % l) | (uint32_t((s / l) % l) << 5) |
                            (uint32_t(s / (l * l)) << 10);
    return uint32_t(s) | (coords << 16);
  }
  __device__ __forceinline__ int distance(uint32_t a, uint32_t b) const {
    const int l = C ? C : side;
    a >>= 16;
    b >>= 16;
    int dx = abs(int(a & 31u) - int(b & 31u));
    int dy = abs(int((a >> 5) & 31u) - int((b >> 5) & 31u));
    int dz = abs(int((a >> 10) & 31u) - int((b >> 10) & 31u));
    dx = min(dx, l - dx);
    dy = min(dy, l - dy);
    dz = min(dz, l - dz);
    return dx + dy + dz;
  }
};

// Trivial shared-memory object: no implicit dynamic shared initialization.
struct DeviceStats {
  bool measured;
  uint64_t rounds, evaluated, finalized, replay_rounds;
  uint64_t cycles[6];
  uint32_t smid;
};

struct Round {
  uint32_t a[256], b[256], flags[256];
  int delta[256];
  uint32_t accepted[8];
  int affected[14];
  int cut, energy;
};

__device__ __forceinline__ void blockSync() {
  if (blockDim.x == 32) __syncwarp(0xffffffffu);
  else __syncthreads();
}

__device__ __forceinline__ int nextAccepted(const Round& r, int begin, int cut) {
  if (begin >= cut) return cut;
  int word = begin >> 5;
  uint32_t bits = r.accepted[word] & (0xffffffffu << (begin & 31));
  while (!bits && ++word * 32 < cut) bits = r.accepted[word];
  return bits ? min(cut, word * 32 + __ffs(bits) - 1) : cut;
}

__device__ __forceinline__ int warpMin(int value) {
  for (int offset = 16; offset; offset >>= 1)
    value = min(value, __shfl_down_sync(0xffffffffu, value, offset));
  return value;
}

// Endpoint-only exact delta. Local particle/hole moves exclude the departing
// particle from the arrival count and S; unlike occupied types keep all counts.
template<bool Local>
__device__ __forceinline__ int deltaEnergy(uint32_t ca, uint32_t cb) {
  const int ta = cellType(ca), tb = cellType(cb);
  if (ta && tb)
    return 2 * (target(tb) - target(ta)) * (cellM(cb) - cellM(ca));
  const uint32_t source = ta ? ca : cb, dest = ta ? cb : ca;
  const int m = cellM(source), n = cellM(dest), ell = target(cellType(source));
  const int before = m - ell;
  const int after = n - (Local ? 1 : 0) - ell;
  int delta = after * after - before * before - 2 * cellS(source) + m;
  if constexpr (Local)
    delta += 2 * (cellS(dest) - m + ell) + n - 1;
  else
    delta += 2 * cellS(dest) + n;
  return delta;
}

// All block threads participate, including the 31 spare lanes for W=1.
// Each proposal is regenerated from its frozen key after a cut, retaining its
// exact rejection stream and lazy acceptance low word on every replay.
template<int C, bool Local, bool Profile>
__device__ void sweep(uint32_t* cells, Round& r, const Geometry<C>& geo,
                      int n, int window, uint32_t lemN, uint64_t seed,
                      uint64_t chain, uint64_t indexBase,
                      const Threshold* thresholds, int thresholdCount,
                      CudaTrace* trace, uint32_t traceCount, uint64_t traceBase,
                      uint32_t& acceptedCount, DeviceStats& stats) {
  const int tid = int(threadIdx.x), warp = tid >> 5, lane = tid & 31;
  int done = 0;
  while (done < n) {
    const int active = min(window, n - done);
    const bool live = tid < active;
    uint64_t roundStart = 0, stageStart = 0;
    if constexpr (Profile) { blockSync(); roundStart = stageStart = clock64(); }
    int a = 0, b = 0, delta = 0, partner = 0;
    DeviceStream rng(0);
    uint32_t flags = 0, pa = 0, pb = 0;
    if (live) {
      const uint64_t index = indexBase + uint64_t(done + tid);
      rng.key = seed ^ (chain * 0x9e3779b97f4a7c15ULL) ^
                (index * 0xd1342543de82ef95ULL) ^
                (Local ? 0x1234ULL : 0x5678ULL);
      a = int(rng.index(uint32_t(n), lemN));
      if constexpr (Local) partner = int(rng.index(6u, 4u));
      else partner = int(rng.index(uint32_t(n), lemN));
    }
    if constexpr (Profile) {
      blockSync();
      if (tid == 0) stats.cycles[int(CudaStage::Rng)] += clock64() - stageStart;
      stageStart = clock64();
    }
    if (live) {
      if constexpr (Local) b = geo.nb(a, partner);
      else b = partner;
      pa = geo.endpoint(a);
      pb = geo.endpoint(b);
    }
    if constexpr (Profile) {
      blockSync();
      if (tid == 0) stats.cycles[int(CudaStage::Address)] += clock64() - stageStart;
      stageStart = clock64();
    }
    if (live) {
      const bool valid = Local || a != b;
      if (valid) {
        const uint32_t hi = rng.next();
        const uint32_t ca = readCell(cells, a), cb = readCell(cells, b);
        const int ta = cellType(ca), tb = cellType(cb);
        flags = 1u | (uint32_t(ta) << 3) | (uint32_t(tb) << 5);
        if (ta == tb) flags |= 2u;
        else {
          if constexpr (Local) {
            delta = deltaEnergy<true>(ca, cb);
          } else if (ta && tb) {
            delta = deltaEnergy<false>(ca, cb);
          } else {
            const int distance = geo.distance(pa, pb);
            delta = distance == 1 ? deltaEnergy<true>(ca, cb)
                                  : deltaEnergy<false>(ca, cb);
            if (distance == 2) {
              // For L>=4 each endpoint's six neighbors are distinct. A common
              // occupied neighbor keeps its count: remove the far formula's +2.
              for (int d = 0; d < 6; ++d) {
                const int neighbor = geo.nb(a, d);
                if (!cellType(readCell(cells, neighbor))) continue;
                for (int e = 0; e < 6; ++e) {
                  if (neighbor == geo.nb(b, e)) {
                    delta -= 2;
                    break;
                  }
                }
              }
            }
          }
          bool accept = delta <= 0;
          if (!accept && delta < thresholdCount) {
            const Threshold t = thresholds[delta];
            accept = hi < t.hi || (hi == t.hi && rng.next() < t.lo);
          }
          if (accept) flags |= 4u;
        }
      }
    }
    r.a[tid] = pa;
    r.b[tid] = pb;
    r.flags[tid] = flags;
    r.delta[tid] = delta;
    const uint32_t mask = __ballot_sync(0xffffffffu, (flags & 4u) != 0);
    if (lane == 0) r.accepted[warp] = mask;
    if (tid == 0) r.cut = active;
    blockSync();

    if constexpr (Profile) {
      if (tid == 0) stats.cycles[int(CudaStage::LoadDelta)] += clock64() - stageStart;
      stageStart = clock64();
    }
    // Rejected and no-op endpoints are readers too: otherwise an accepted
    // predecessor could resurrect a rejection or turn a no-op into a move.
    const int radius = (flags & 2u) ? 0 :
      (((flags >> 3) & 3u) && ((flags >> 5) & 3u) ? 1 : 2);
    bool conflict = false;
    if (flags & 1u) {
      for (int w = 0; w <= warp && !conflict; ++w) {
        uint32_t bits = r.accepted[w];
        if (w == warp) bits &= (uint32_t(1) << lane) - 1u;
        while (bits) {
          const int j = w * 32 + __ffs(bits) - 1;
          bits &= bits - 1;
          if (geo.distance(pa, r.a[j]) <= radius || geo.distance(pa, r.b[j]) <= radius ||
              geo.distance(pb, r.a[j]) <= radius || geo.distance(pb, r.b[j]) <= radius) {
            conflict = true;
            break;
          }
        }
      }
    }
    const int candidate = warpMin(conflict ? tid : active);
    if (lane == 0) atomicMin(&r.cut, candidate);
    blockSync();
    const int cut = r.cut; // Always >=1: the first proposal has no predecessors.
    if (live && tid < cut && traceBase + uint64_t(done + tid) < traceCount) {
      CudaTrace entry;
      entry.proposal = indexBase + uint64_t(done + tid);
      entry.a = uint32_t(a);
      entry.b = uint32_t(b);
      entry.delta = delta;
      entry.flags = flags & 7u;
      trace[traceBase + uint64_t(done + tid)] = entry;
    }

    if constexpr (Profile) {
      blockSync();
      if (tid == 0) stats.cycles[int(CudaStage::Conflict)] += clock64() - stageStart;
      stageStart = clock64();
    }
    // Serial accepted order, with endpoint-only maintenance for occupied swaps.
    // All particle/hole moves retain the packed affected-site ownership.
    for (int j = nextAccepted(r, 0, cut); j < cut; j = nextAccepted(r, j + 1, cut)) {
      const uint32_t jf = r.flags[j];
      const int aa = int(r.a[j] & 65535u), bb = int(r.b[j] & 65535u);
      const int ta = int((jf >> 3) & 3u), tb = int((jf >> 5) & 3u);
      if (!(ta && tb)) {
        if (tid == 0) r.affected[0] = aa;
        else if (tid == 1) r.affected[1] = bb;
        else if (tid < 8) r.affected[tid] = geo.nb(aa, tid - 2);
        else if (tid < 14) r.affected[tid] = geo.nb(bb, tid - 8);
      }
      blockSync();
      if (ta && tb) {
        // Occupancy and every m stay fixed. Endpoint and S updates can alias
        // either physical half, so all updates use full-word packed atomics.
        if (tid < 2) {
          const int site = tid == 0 ? aa : bb;
          const int oldType = tid == 0 ? ta : tb;
          const int newType = tid == 0 ? tb : ta;
          AtomicAddCell(cells, site, uint32_t(newType) - uint32_t(oldType));
          const int dq = target(oldType) - target(newType);
          for (int d = 0; d < 6; ++d)
            AtomicAddCell(cells, geo.nb(site, d), uint32_t(dq) * 32u);
        }
      } else if (tid < 14) {
        const int site = r.affected[tid];
        bool first = true;
        for (int i = 0; i < tid; ++i)
          if (r.affected[i] == site) first = false;
        if (first) {
          const uint32_t old = AtomicReadCell(cells, site) & 31u;
          const int oldType = cellType(old), oldM = cellM(old);
          const int occupancyDelta = int(tb != 0) - int(ta != 0);
          int dm = 0;
          for (int i = 2; i < 8; ++i)
            if (r.affected[i] == site) dm += occupancyDelta;
          for (int i = 8; i < 14; ++i)
            if (r.affected[i] == site) dm -= occupancyDelta;
          const int nt = site == aa ? tb : (site == bb ? ta : oldType);
          const int nm = oldM + dm;
          const uint32_t next = uint32_t(nt | (nm << 2));
          const int dq = q(nt, nm) - q(oldType, oldM);
          // Logical owners may share a physical word, including S destinations.
          // Only full-word atomics may touch either half during this phase.
          AtomicAddCell(cells, site, next - old);
          if (dq)
            for (int d = 0; d < 6; ++d)
              AtomicAddCell(cells, geo.nb(site, d), uint32_t(dq) * 32u);
        }
      }
      if (tid == 0) {
        r.energy += r.delta[j];
        ++acceptedCount;
      }
      blockSync();
    }
    if constexpr (Profile) {
      blockSync();
      if (tid == 0) {
        stats.cycles[int(CudaStage::Commit)] += clock64() - stageStart;
        ++stats.rounds;
        stats.evaluated += active;
        stats.finalized += cut;
        stats.replay_rounds += cut < active;
        stats.cycles[int(CudaStage::Round)] += clock64() - roundStart;
      }
    }
    done += cut;
    // A round with no accepted proposals still needs a barrier before its
    // shared proposal arrays can be reused by the next round.
    blockSync();
  }
}

template<int C, int Mode, bool Profile>
__global__ void runKernel(uint8_t* state, uint32_t* energies,
                          uint32_t* localCounts, uint32_t* nonlocalCounts,
                          int l, int n, int window, uint32_t lemN,
                          uint64_t seed, uint64_t chainOffset, uint64_t start,
                          uint32_t sweeps, const Threshold* thresholds,
                          int thresholdCount, CudaTrace* traces,
                          uint32_t traceCount, CudaStats* outputStats) {
  extern __shared__ uint32_t cells[];
  __shared__ Round round;
  __shared__ DeviceStats stats;
  const int tid = int(threadIdx.x);
  const size_t chainIndex = size_t(blockIdx.x);
  uint8_t* const bytes = state + chainIndex * size_t(n);
  const Geometry<C> geo(l);
  if (tid == 0) {
    round.energy = 0;
    stats = DeviceStats{};
    stats.measured = Profile;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(stats.smid));
  }
  blockSync();
  int energy = 0;
  for (int word = tid; word < (n + 1) / 2; word += int(blockDim.x)) {
    uint32_t packed = 0;
    // One physical-word owner builds both halves from immutable input bytes.
    // The odd-N phantom high half stays zero and is never read from input.
    for (int half = 0; half < 2; ++half) {
      const int s = 2 * word + half;
      if (s < n) {
        const uint32_t byte = bytes[s];
        const int type = int(byte & 3u), m = __popc(byte >> 2);
        int sum = 0;
        for (int d = 0; d < 6; ++d) {
          const uint32_t neighbor = bytes[geo.nb(s, d)];
          sum += q(int(neighbor & 3u), __popc(neighbor >> 2));
        }
        const uint32_t cell = uint32_t(type | (m << 2)) |
                              (uint32_t(sum + 32) << 5);
        packed |= cell << (half * 16);
        energy += siteEnergy(type, m);
      }
    }
    cells[word] = packed;
  }
  for (int offset = 16; offset; offset >>= 1)
    energy += __shfl_down_sync(0xffffffffu, energy, offset);
  if ((tid & 31) == 0) atomicAdd(&round.energy, energy);
  blockSync();
  CudaTrace* const trace = traceCount ? traces + chainIndex * traceCount : nullptr;
  for (uint32_t s = 0; s < sweeps; ++s) {
    const uint64_t sweepIndex = start + uint64_t(s);
    const uint64_t traceBase = uint64_t(s) * uint64_t(n) * (Mode == 0 ? 2u : 1u);
    uint32_t al = 0, an = 0;
    if constexpr (Mode != 2)
      sweep<C, false, Profile>(cells, round, geo, n, window, lemN, seed,
                       chainOffset + uint64_t(blockIdx.x), (2 * sweepIndex) * uint64_t(n),
                       thresholds, thresholdCount, trace, traceCount, traceBase, an, stats);
    if constexpr (Mode != 1)
      sweep<C, true, Profile>(cells, round, geo, n, window, lemN, seed,
                      chainOffset + uint64_t(blockIdx.x), (2 * sweepIndex + 1) * uint64_t(n),
                      thresholds, thresholdCount, trace, traceCount,
                      traceBase + (Mode == 0 ? uint64_t(n) : 0u), al, stats);
    if (tid == 0) {
      const size_t h = chainIndex * sweeps + s;
      energies[h] = uint32_t(round.energy);
      localCounts[h] = al;
      nonlocalCounts[h] = an;
    }
  }
  if (tid == 0) {
    CudaStats& out = outputStats[chainIndex];
    out.measured = stats.measured;
    out.rounds = stats.rounds;
    out.evaluated = stats.evaluated;
    out.finalized = stats.finalized;
    out.replay_rounds = stats.replay_rounds;
    for (int i = 0; i < 6; ++i) out.cycles[i] = stats.cycles[i];
    out.smid = stats.smid;
  }
  // Write CPU's directional occupancy mask, not the internal cached count.
  for (int s = tid; s < n; s += int(blockDim.x)) {
    uint32_t byte = readCell(cells, s) & 3u;
    for (int d = 0; d < 6; ++d)
      byte |= uint32_t(cellType(readCell(cells, geo.nb(s, d))) != 0) << (d + 2);
    bytes[s] = uint8_t(byte);
  }
}

} // namespace

struct CudaBackend::Impl {
  int L, N, chains, thresholdCount, device;
  CudaLaunch config;
  uint64_t seed, chainOffset;
  Buffer<uint8_t> state;
  Buffer<Threshold> thresholds;
  Buffer<uint32_t> histories;
  Buffer<CudaTrace> traces;
  Buffer<CudaStats> stats;
  cudaStream_t stream = nullptr;
  cudaEvent_t begin = nullptr, end = nullptr;

  Impl(int l, int c, uint64_t s, CudaLaunch cfg, uint64_t offset, int dev)
      : L(l), N(l*l*l), chains(c), thresholdCount(0), device(dev), config(cfg),
        seed(s), chainOffset(offset) {}
  ~Impl() {
    cudaSetDevice(device);
    if (stream) cudaStreamSynchronize(stream);
    if (begin) cudaEventDestroy(begin);
    if (end) cudaEventDestroy(end);
    if (stream) cudaStreamDestroy(stream);
  }
  template<int C, int Mode, bool Profile>
  void launch(uint64_t start, uint32_t sweeps, size_t samples, uint32_t traceCount) {
    runKernel<C, Mode, Profile><<<chains, config.threads,
      size_t((N + 1) / 2)*sizeof(uint32_t), stream>>>(
      state.data, histories.data, histories.data + samples,
      histories.data + 2*samples, L, N, config.window,
      (0u-uint32_t(N))%uint32_t(N), seed, chainOffset, start, sweeps,
      thresholds.data, thresholdCount, traces.data, traceCount, stats.data);
  }
  template<int C, bool Profile>
  void dispatch(uint64_t start, uint32_t sweeps, int mode, size_t samples,
                uint32_t traceCount) {
    if (mode == 0) launch<C, 0, Profile>(start, sweeps, samples, traceCount);
    else if (mode == 1) launch<C, 1, Profile>(start, sweeps, samples, traceCount);
    else launch<C, 2, Profile>(start, sweeps, samples, traceCount);
  }
  template<bool Profile>
  void dispatchSize(uint64_t start, uint32_t sweeps, int mode, size_t samples,
                    uint32_t traceCount) {
    if (L == 8) dispatch<8, Profile>(start, sweeps, mode, samples, traceCount);
    else if (L == 21) dispatch<21, Profile>(start, sweeps, mode, samples, traceCount);
    else dispatch<0, Profile>(start, sweeps, mode, samples, traceCount);
  }
};

CudaDeviceInfo cudaDeviceInfo() {
  int device = 0;
  checked(cudaGetDevice(&device), "cudaGetDevice");
  cudaDeviceProp prop;
  checked(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
  return {prop.name, prop.multiProcessorCount, prop.clockRate};
}

CudaBackend::CudaBackend(int L, int chains, uint64_t seed,
                         const std::vector<uint8_t>& initial,
                         const std::vector<Threshold>& thresholds, CudaLaunch config,
                         uint64_t chain_offset) : impl_(nullptr) {
  if (L < 4 || L > 21 || chains <= 0)
    throw std::invalid_argument("CUDA requires 4 <= L <= 21 and positive chains");
  if (config.window != 1 && config.window != 32 && config.window != 64 &&
      config.window != 128 && config.window != 256)
    throw std::invalid_argument("baseline CUDA window must be 1,32,64,128,256");
  if ((config.threads != 32 && config.threads != 64 && config.threads != 128 &&
       config.threads != 256) || config.window > config.threads ||
      config.proposals_per_thread != 1)
    throw std::invalid_argument("baseline requires T32/64/128/256, W<=T, ppt1");
  if (uint64_t(chains-1) > UINT64_MAX-chain_offset)
    throw std::invalid_argument("CUDA chain range overflow");
  const size_t bytes = product(size_t(chains), size_t(L*L*L));
  if (initial.size() != bytes)
    throw std::invalid_argument("CUDA initial state size does not match chains*L^3");
  if (thresholds.empty() || thresholds.size() > size_t(std::numeric_limits<int>::max()))
    throw std::invalid_argument("CUDA threshold table size is invalid");
  int device = 0;
  checked(cudaGetDevice(&device), "cudaGetDevice");
  cudaDeviceProp prop;
  checked(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
  if (size_t((L*L*L + 1)/2)*sizeof(uint32_t)+sizeof(Round)+sizeof(DeviceStats) >
        prop.sharedMemPerBlock ||
      config.threads > prop.maxThreadsPerBlock || chains > prop.maxGridSize[0])
    throw std::runtime_error("CUDA device cannot support requested shared-memory launch");
  std::unique_ptr<Impl> p(new Impl(L, chains, seed, config, chain_offset, device));
  checked(cudaStreamCreateWithFlags(&p->stream, cudaStreamNonBlocking), "create CUDA stream");
  p->state.reserve(bytes);
  p->thresholds.reserve(thresholds.size());
  p->stats.reserve(size_t(chains));
  p->thresholdCount = int(thresholds.size());
  checked(cudaMemcpyAsync(p->state.data, initial.data(), bytes, cudaMemcpyHostToDevice, p->stream),
          "upload CUDA state");
  checked(cudaMemcpyAsync(p->thresholds.data, thresholds.data(),
          product(thresholds.size(), sizeof(Threshold)), cudaMemcpyHostToDevice, p->stream),
          "upload CUDA thresholds");
  checked(cudaEventCreate(&p->begin), "create CUDA start event");
  checked(cudaEventCreate(&p->end), "create CUDA stop event");
  checked(cudaStreamSynchronize(p->stream), "complete CUDA initialization");
  impl_ = p.release();
}
CudaBackend::~CudaBackend() { delete impl_; }

void CudaBackend::reserve(uint32_t max_sweeps, uint32_t max_trace_per_chain) {
  Impl& p = *impl_;
  checked(cudaSetDevice(p.device), "select CUDA backend device");
  p.histories.reserve(product(product(size_t(p.chains), size_t(max_sweeps)), 3));
  p.traces.reserve(product(size_t(p.chains), size_t(max_trace_per_chain)));
}

void CudaBackend::run(uint64_t start, uint32_t sweeps, int mode,
                       CudaResult& result, uint32_t trace_per_chain) {
  if (mode < 0 || mode > 2) throw std::invalid_argument("CUDA mode must be 0,1,2");
  Impl& p = *impl_;
  checked(cudaSetDevice(p.device), "select CUDA backend device");
  const size_t samples = product(size_t(p.chains), size_t(sweeps));
  const uint64_t attempts = uint64_t(sweeps)*uint64_t(p.N)*(mode == 0 ? 2u : 1u);
  const uint32_t traceCount = uint32_t(std::min(uint64_t(trace_per_chain), attempts));
  reserve(sweeps, traceCount);
  result.state.resize(product(size_t(p.chains), size_t(p.N)));
  result.energy.resize(samples);
  result.acceptLocal.resize(samples);
  result.acceptNonlocal.resize(samples);
  result.trace.resize(product(size_t(p.chains), size_t(traceCount)));
  result.stats.assign(sweeps ? size_t(p.chains) : 0, CudaStats{});
  result.milliseconds = 0;
  if (sweeps) {
    checked(cudaEventRecord(p.begin, p.stream), "record CUDA start event");
    if (p.config.profile) p.dispatchSize<true>(start, sweeps, mode, samples, traceCount);
    else p.dispatchSize<false>(start, sweeps, mode, samples, traceCount);
    checked(cudaGetLastError(), "launch CUDA chain kernel");
    checked(cudaEventRecord(p.end, p.stream), "record CUDA stop event");
    const size_t historyBytes = product(samples, sizeof(uint32_t));
    checked(cudaMemcpyAsync(result.energy.data(), p.histories.data, historyBytes,
            cudaMemcpyDeviceToHost, p.stream), "download CUDA energies");
    checked(cudaMemcpyAsync(result.acceptLocal.data(), p.histories.data+samples, historyBytes,
            cudaMemcpyDeviceToHost, p.stream), "download CUDA local counts");
    checked(cudaMemcpyAsync(result.acceptNonlocal.data(), p.histories.data+2*samples, historyBytes,
            cudaMemcpyDeviceToHost, p.stream), "download CUDA nonlocal counts");
    if (traceCount)
      checked(cudaMemcpyAsync(result.trace.data(), p.traces.data,
              product(result.trace.size(), sizeof(CudaTrace)), cudaMemcpyDeviceToHost, p.stream),
              "download CUDA trace");
    checked(cudaMemcpyAsync(result.stats.data(), p.stats.data,
            product(result.stats.size(), sizeof(CudaStats)), cudaMemcpyDeviceToHost, p.stream),
            "download CUDA stats");
  }
  checked(cudaMemcpyAsync(result.state.data(), p.state.data, result.state.size(),
          cudaMemcpyDeviceToHost, p.stream), "download CUDA state");
  checked(cudaStreamSynchronize(p.stream), "complete CUDA backend run");
  if (sweeps)
    checked(cudaEventElapsedTime(&result.milliseconds, p.begin, p.end), "time CUDA kernel");
}
} // namespace solid