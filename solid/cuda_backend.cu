#include "cuda_backend.hpp"
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

// Low five bits hold type and m. The signed high half holds S. Adding
// dq * 65536 atomically changes only S; adding a low-field difference changes
// only type/m, because the resulting low field is always nonnegative and <32.
__device__ __forceinline__ int cellType(uint32_t cell) { return int(cell & 3u); }
__device__ __forceinline__ int cellM(uint32_t cell) { return int((cell >> 2) & 7u); }
__device__ __forceinline__ int cellS(uint32_t cell) { return int32_t(cell) >> 16; }
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
template<int C, bool Local>
__device__ void sweep(uint32_t* cells, Round& r, const Geometry<C>& geo,
                      int n, int window, uint32_t lemN, uint64_t seed,
                      uint64_t chain, uint64_t indexBase,
                      const Threshold* thresholds, int thresholdCount,
                      CudaTrace* trace, uint32_t traceCount, uint64_t traceBase,
                      uint32_t& acceptedCount) {
  const int tid = int(threadIdx.x), warp = tid >> 5, lane = tid & 31;
  int done = 0;
  while (done < n) {
    const int active = min(window, n - done);
    const bool live = tid < active;
    int a = 0, b = 0, delta = 0;
    uint32_t flags = 0, pa = 0, pb = 0;
    if (live) {
      const uint64_t index = indexBase + uint64_t(done + tid);
      DeviceStream rng(seed ^ (chain * 0x9e3779b97f4a7c15ULL) ^
                       (index * 0xd1342543de82ef95ULL) ^
                       (Local ? 0x1234ULL : 0x5678ULL));
      a = int(rng.index(uint32_t(n), lemN));
      if constexpr (Local)
        b = geo.nb(a, int(rng.index(6u, 4u)));
      else
        b = int(rng.index(uint32_t(n), lemN));
      pa = geo.endpoint(a);
      pb = geo.endpoint(b);
      const bool valid = Local || a != b;
      if (valid) {
        const uint32_t hi = rng.next();
        const uint32_t ca = cells[a], cb = cells[b];
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
                if (!cellType(cells[neighbor])) continue;
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

    // Serial accepted order, cooperative exact cache maintenance. There are
    // at most 14 distinct changed-q sites. S scatters commute via shared atomics.
    for (int j = nextAccepted(r, 0, cut); j < cut; j = nextAccepted(r, j + 1, cut)) {
      const uint32_t jf = r.flags[j];
      const int aa = int(r.a[j] & 65535u), bb = int(r.b[j] & 65535u);
      const int ta = int((jf >> 3) & 3u), tb = int((jf >> 5) & 3u);
      if (tid == 0) r.affected[0] = aa;
      else if (tid == 1) r.affected[1] = bb;
      else if (tid < 8) r.affected[tid] = geo.nb(aa, tid - 2);
      else if (tid < 14) r.affected[tid] = geo.nb(bb, tid - 8);
      blockSync();
      if (tid < 14) {
        const int site = r.affected[tid];
        bool first = true;
        for (int i = 0; i < tid; ++i)
          if (r.affected[i] == site) first = false;
        if (first) {
          const uint32_t old = atomicAdd(cells + site, 0u) & 31u;
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
          // Atomic full-word accesses also avoid races with neighbors' S adds.
          atomicAdd(cells + site, next - old);
          if (dq)
            for (int d = 0; d < 6; ++d)
              atomicAdd(cells + geo.nb(site, d), uint32_t(dq) * 65536u);
        }
      }
      if (tid == 0) {
        r.energy += r.delta[j];
        ++acceptedCount;
      }
      blockSync();
    }
    done += cut;
    // A round with no accepted proposals still needs a barrier before its
    // shared proposal arrays can be reused by the next round.
    blockSync();
  }
}

template<int C, int Mode>
__global__ void runKernel(uint8_t* state, uint32_t* energies,
                          uint32_t* localCounts, uint32_t* nonlocalCounts,
                          int l, int n, int window, uint32_t lemN,
                          uint64_t seed, uint64_t chainOffset, uint64_t start,
                          uint32_t sweeps, const Threshold* thresholds,
                          int thresholdCount, CudaTrace* traces,
                          uint32_t traceCount) {
  extern __shared__ uint32_t cells[];
  __shared__ Round round;
  const int tid = int(threadIdx.x);
  const size_t chainIndex = size_t(blockIdx.x);
  uint8_t* const bytes = state + chainIndex * size_t(n);
  const Geometry<C> geo(l);
  if (tid == 0) round.energy = 0;
  blockSync();
  int energy = 0;
  for (int s = tid; s < n; s += int(blockDim.x)) {
    const uint32_t byte = bytes[s];
    const int type = int(byte & 3u), m = __popc(byte >> 2);
    int sum = 0;
    // Read immutable global input here, so concurrent shared initialization
    // cannot race a neighboring thread's packed-cell store.
    for (int d = 0; d < 6; ++d) {
      const uint32_t neighbor = bytes[geo.nb(s, d)];
      sum += q(int(neighbor & 3u), __popc(neighbor >> 2));
    }
    cells[s] = uint32_t(type | (m << 2)) | (uint32_t(sum) << 16);
    energy += siteEnergy(type, m);
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
      sweep<C, false>(cells, round, geo, n, window, lemN, seed,
                       chainOffset + uint64_t(blockIdx.x), (2 * sweepIndex) * uint64_t(n),
                       thresholds, thresholdCount, trace, traceCount, traceBase, an);
    if constexpr (Mode != 1)
      sweep<C, true>(cells, round, geo, n, window, lemN, seed,
                      chainOffset + uint64_t(blockIdx.x), (2 * sweepIndex + 1) * uint64_t(n),
                      thresholds, thresholdCount, trace, traceCount,
                      traceBase + (Mode == 0 ? uint64_t(n) : 0u), al);
    if (tid == 0) {
      const size_t h = chainIndex * sweeps + s;
      energies[h] = uint32_t(round.energy);
      localCounts[h] = al;
      nonlocalCounts[h] = an;
    }
  }
  // Write CPU's directional occupancy mask, not the internal cached count.
  for (int s = tid; s < n; s += int(blockDim.x)) {
    uint32_t byte = cells[s] & 3u;
    for (int d = 0; d < 6; ++d)
      byte |= uint32_t(cellType(cells[geo.nb(s, d)]) != 0) << (d + 2);
    bytes[s] = uint8_t(byte);
  }
}

} // namespace

struct CudaBackend::Impl {
  int L, N, chains, window, thresholdCount;
  uint64_t seed, chainOffset;
  Buffer<uint8_t> state;
  Buffer<Threshold> thresholds;
  Buffer<uint32_t> histories;
  Buffer<CudaTrace> traces;
  cudaEvent_t begin = nullptr, end = nullptr;

  Impl(int l, int c, uint64_t s, int w, uint64_t offset)
      : L(l), N(l * l * l), chains(c), window(w), thresholdCount(0),
        seed(s), chainOffset(offset) {}
  ~Impl() {
    if (begin) cudaEventDestroy(begin);
    if (end) cudaEventDestroy(end);
  }

  template<int C, int Mode>
  void launch(uint64_t start, uint32_t sweeps, size_t samples, uint32_t traceCount) {
    runKernel<C, Mode><<<chains, std::max(32, window), size_t(N) * sizeof(uint32_t)>>>(
      state.data, histories.data, histories.data + samples,
      histories.data + 2 * samples, L, N, window,
      (0u - uint32_t(N)) % uint32_t(N), seed, chainOffset, start, sweeps,
      thresholds.data, thresholdCount, traces.data, traceCount);
  }
  template<int C>
  void dispatch(uint64_t start, uint32_t sweeps, int mode, size_t samples,
                uint32_t traceCount) {
    if (mode == 0) launch<C, 0>(start, sweeps, samples, traceCount);
    else if (mode == 1) launch<C, 1>(start, sweeps, samples, traceCount);
    else launch<C, 2>(start, sweeps, samples, traceCount);
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
                         const std::vector<Threshold>& thresholds, int window,
                         uint64_t chain_offset) : impl_(nullptr) {
  if (L < 4 || L > 21 || chains <= 0)
    throw std::invalid_argument("CUDA requires 4 <= L <= 21 and positive chains");
  if (window != 1 && window != 32 && window != 64 && window != 128 && window != 256)
    throw std::invalid_argument("CUDA window must be 1, 32, 64, 128, or 256");
  const size_t bytes = product(size_t(chains), size_t(L * L * L));
  if (initial.size() != bytes)
    throw std::invalid_argument("CUDA initial state size does not match chains * L^3");
  if (thresholds.size() > size_t(std::numeric_limits<int>::max()))
    throw std::length_error("CUDA threshold table is too large");
  int device = 0;
  checked(cudaGetDevice(&device), "cudaGetDevice");
  cudaDeviceProp prop;
  checked(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
  if (size_t(L * L * L) * sizeof(uint32_t) + sizeof(Round) > prop.sharedMemPerBlock ||
      std::max(32, window) > prop.maxThreadsPerBlock || chains > prop.maxGridSize[0])
    throw std::runtime_error("CUDA device cannot support requested shared-memory launch");
  std::unique_ptr<Impl> p(new Impl(L, chains, seed, window, chain_offset));
  p->state.reserve(bytes);
  p->thresholds.reserve(thresholds.size());
  p->thresholdCount = int(thresholds.size());
  checked(cudaMemcpy(p->state.data, initial.data(), bytes, cudaMemcpyHostToDevice),
          "upload CUDA state");
  if (!thresholds.empty())
    checked(cudaMemcpy(p->thresholds.data, thresholds.data(),
                       product(thresholds.size(), sizeof(Threshold)), cudaMemcpyHostToDevice),
            "upload CUDA thresholds");
  checked(cudaEventCreate(&p->begin), "create CUDA start event");
  checked(cudaEventCreate(&p->end), "create CUDA stop event");
  impl_ = p.release();
}

CudaBackend::~CudaBackend() { delete impl_; }

void CudaBackend::run(uint64_t start, uint32_t sweeps, int mode,
                       CudaResult& result, uint32_t trace_per_chain) {
  if (mode < 0 || mode > 2) throw std::invalid_argument("CUDA mode must be 0, 1, or 2");
  Impl& p = *impl_;
  const size_t samples = product(size_t(p.chains), size_t(sweeps));
  const uint64_t attempts = uint64_t(sweeps) * uint64_t(p.N) * (mode == 0 ? 2u : 1u);
  const uint32_t traceCount = uint32_t(std::min(uint64_t(trace_per_chain), attempts));
  result.state.resize(product(size_t(p.chains), size_t(p.N)));
  result.energy.resize(samples);
  result.acceptLocal.resize(samples);
  result.acceptNonlocal.resize(samples);
  result.trace.resize(product(size_t(p.chains), size_t(traceCount)));
  result.milliseconds = 0;
  if (sweeps) {
    p.histories.reserve(product(samples, 3));
    p.traces.reserve(result.trace.size());
    checked(cudaEventRecord(p.begin), "record CUDA start event");
    if (p.L == 8) p.dispatch<8>(start, sweeps, mode, samples, traceCount);
    else if (p.L == 21) p.dispatch<21>(start, sweeps, mode, samples, traceCount);
    else p.dispatch<0>(start, sweeps, mode, samples, traceCount);
    checked(cudaGetLastError(), "launch CUDA chain kernel");
    checked(cudaEventRecord(p.end), "record CUDA stop event");
    checked(cudaEventSynchronize(p.end), "synchronize CUDA chain kernel");
    checked(cudaEventElapsedTime(&result.milliseconds, p.begin, p.end), "time CUDA kernel");
    const size_t historyBytes = product(samples, sizeof(uint32_t));
    checked(cudaMemcpy(result.energy.data(), p.histories.data, historyBytes,
                       cudaMemcpyDeviceToHost), "download CUDA energies");
    checked(cudaMemcpy(result.acceptLocal.data(), p.histories.data + samples, historyBytes,
                       cudaMemcpyDeviceToHost), "download CUDA local counts");
    checked(cudaMemcpy(result.acceptNonlocal.data(), p.histories.data + 2 * samples, historyBytes,
                       cudaMemcpyDeviceToHost), "download CUDA nonlocal counts");
    if (traceCount)
      checked(cudaMemcpy(result.trace.data(), p.traces.data,
                         product(result.trace.size(), sizeof(CudaTrace)), cudaMemcpyDeviceToHost),
              "download CUDA trace");
  }
  checked(cudaMemcpy(result.state.data(), p.state.data, result.state.size(),
                     cudaMemcpyDeviceToHost), "download CUDA state");
}

} // namespace solid
