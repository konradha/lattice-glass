#include "lattice.hpp"
#include "sweeps_cpu.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <cerrno>

#ifndef SOLID_HAVE_METAL
#define SOLID_HAVE_METAL 0
#endif

#if SOLID_HAVE_METAL
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "Metal/Metal.hpp"
#endif

using namespace solid;

// ---------------------------------------------------------------------------
// Minimal stored-ZIP .npz writer.
// ---------------------------------------------------------------------------
static uint32_t crc32_tab[256];
static void init_crc() {
  for (uint32_t i = 0; i < 256; ++i) {
    uint32_t c = i;
    for (int k = 0; k < 8; ++k) c = (c & 1) ? (0xedb88320u ^ (c >> 1)) : (c >> 1);
    crc32_tab[i] = c;
  }
}
static uint32_t crc32(const uint8_t* p, size_t n) {
  uint32_t c = 0xffffffffu;
  for (size_t i = 0; i < n; ++i) c = crc32_tab[(c ^ p[i]) & 255] ^ (c >> 8);
  return c ^ 0xffffffffu;
}
static void put16(std::vector<uint8_t>& b, uint16_t v) {
  b.push_back(uint8_t(v)); b.push_back(uint8_t(v >> 8));
}
static void put32(std::vector<uint8_t>& b, uint32_t v) {
  b.push_back(uint8_t(v)); b.push_back(uint8_t(v >> 8));
  b.push_back(uint8_t(v >> 16)); b.push_back(uint8_t(v >> 24));
}
struct ZipEntry { std::string name; uint32_t crc, size, offset; };
static std::vector<uint8_t> npyHeader(const std::string& descr,
                                      const std::vector<size_t>& shape) {
  std::string s = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': (";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) s += ", ";
    s += std::to_string(shape[i]);
  }
  if (shape.size() == 1) s += ",";
  s += "), }";
  const size_t base = 10;
  size_t pad = (16 - ((base + s.size() + 1) % 16)) % 16;
  s.append(pad, ' ');
  s += '\n';
  std::vector<uint8_t> h = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
  put16(h, uint16_t(s.size()));
  h.insert(h.end(), s.begin(), s.end());
  return h;
}
static void zipAdd(std::ofstream& f, std::vector<ZipEntry>& entries,
                   const std::string& name, const std::string& descr,
                   const std::vector<size_t>& shape,
                   const std::vector<uint8_t>& payload) {
  std::vector<uint8_t> data = npyHeader(descr, shape);
  data.insert(data.end(), payload.begin(), payload.end());
  ZipEntry e; e.name = name; e.crc = crc32(data.data(), data.size());
  e.size = uint32_t(data.size()); e.offset = uint32_t(f.tellp());
  // Write local header separately before data.
  std::vector<uint8_t> lh;
  put32(lh, 0x04034b50u); put16(lh, 20); put16(lh, 0); put16(lh, 0);
  put16(lh, 0); put16(lh, 0); put32(lh, e.crc); put32(lh, e.size); put32(lh, e.size);
  put16(lh, uint16_t(name.size())); put16(lh, 0);
  lh.insert(lh.end(), name.begin(), name.end());
  f.write((const char*)lh.data(), lh.size());
  f.write((const char*)data.data(), data.size());
  entries.push_back(e);
}
static void zipFinish(std::ofstream& f, const std::vector<ZipEntry>& entries) {
  uint32_t start = uint32_t(f.tellp());
  for (auto& e : entries) {
    std::vector<uint8_t> c;
    put32(c, 0x02014b50u); put16(c, 20); put16(c, 20); put16(c, 0); put16(c, 0);
    put16(c, 0); put16(c, 0); put32(c, e.crc); put32(c, e.size); put32(c, e.size);
    put16(c, uint16_t(e.name.size())); put16(c, 0); put16(c, 0);
    put16(c, 0); put16(c, 0); put32(c, 0); put32(c, e.offset);
    c.insert(c.end(), e.name.begin(), e.name.end());
    f.write((const char*)c.data(), c.size());
  }
  uint32_t end = uint32_t(f.tellp());
  std::vector<uint8_t> e;
  put32(e, 0x06054b50u); put16(e, 0); put16(e, 0);
  put16(e, uint16_t(entries.size())); put16(e, uint16_t(entries.size()));
  put32(e, end - start); put32(e, start); put16(e, 0);
  f.write((const char*)e.data(), e.size());
}
static std::vector<uint8_t> asBytes(const void* p, size_t n) {
  const uint8_t* q = (const uint8_t*)p; return std::vector<uint8_t>(q, q + n);
}

struct Hist {
  std::vector<float> energy, accLocal, accNonlocal;
  uint64_t every = 1, sweeps = 0;
};

static void cpuSegment(std::vector<Lattice>& lats, int chains, uint64_t seed,
                       uint64_t start, uint64_t count, uint64_t histStart,
                       const std::vector<Threshold>& th,
                       Hist& hist, uint64_t total, int nocc, int mode) {
  std::atomic<int> next{0};
  int nthreads = std::min<int>(chains, 64);
  std::vector<std::thread> ts;
  for (int w = 0; w < nthreads; ++w) ts.emplace_back([&, w]() {
    for (;;) {
      int c = next.fetch_add(1);
      if (c >= chains) break;
      Lattice& lat = lats[c];
      int e = lat.energy();
      for (uint64_t k = 0; k < count; ++k) {
        uint64_t sw = start + k;
        int an = (mode != 2) ? nonlocalSweep(lat, seed, c, sw, th, e) : 0;
        int al = (mode != 1) ? localSweep(lat, seed, c, sw, th, e) : 0;
        e = lat.energy();
        uint64_t elapsed = sw - histStart + 1;
        if (total && (elapsed % hist.every == 0 || elapsed == hist.sweeps)) {
          uint64_t idx = uint64_t(c) * total + (elapsed - 1) / hist.every;
          hist.energy[idx] = float(e);
          hist.accNonlocal[idx] = float(an) / float(lat.N);
          hist.accLocal[idx] = float(al) / float(lat.N);
        }
      }
    }
  });
  for (auto& t : ts) t.join();
}

#if SOLID_HAVE_METAL
struct GPU {
  MTL::Device* dev = nullptr;
  MTL::CommandQueue* q = nullptr;
  MTL::ComputePipelineState* pso = nullptr;
  explicit GPU(const char* path) {
    dev = MTL::CreateSystemDefaultDevice();
    if (!dev) { fprintf(stderr, "no Metal device\n"); exit(1); }
    q = dev->newCommandQueue();
    std::ifstream mf(path); if (!mf) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    std::string src((std::istreambuf_iterator<char>(mf)), std::istreambuf_iterator<char>());
    NS::Error* err = nullptr;
    NS::String* ns = NS::String::string(src.c_str(), NS::UTF8StringEncoding);
    MTL::Library* lib = dev->newLibrary(ns, nullptr, &err);
    if (!lib) { fprintf(stderr, "MSL compile: %s\n", err ? err->localizedDescription()->utf8String() : "?"); exit(1); }
    NS::String* fn = NS::String::string("run_persistent", NS::UTF8StringEncoding);
    MTL::Function* f = lib->newFunction(fn);
    if (!f) { fprintf(stderr, "missing kernel run_persistent in %s\n", path); exit(1); }
    pso = dev->newComputePipelineState(f, &err);
    if (!pso) { fprintf(stderr, "pipeline: %s\n", err ? err->localizedDescription()->utf8String() : "?"); exit(1); }
  }
};


// One persistent-kernel segment.  Mirrors struct Seg in kernels.metal.
struct GPUSeg {
  uint64_t start;
  uint64_t count;
  uint32_t snapSlot;
  uint32_t hist;
};
static constexpr uint32_t GPUSNAP_NONE = 0xffffffffu;

// Dispatch a list of segments in one command buffer.  The kernel keeps each
// chain's state in threadgroup memory across segments, writes sampled history
// into the full-length device history arrays, and copies the post-segment state
// into `snaps` only for segments with a snapshot slot.  The state is written
// back to `state` at the end of the dispatch, so successive dispatches (the
// wall-clock checkpoint chunks) continue from it.
static void gpuRun(GPU& gpu, MTL::Buffer* state, MTL::Buffer* thbuf,
                   MTL::Buffer* pcbuf, MTL::Buffer* nnbuf,
                   MTL::Buffer* eh, MTL::Buffer* al, MTL::Buffer* an,
                   MTL::Buffer* segbuf,
                   int chains, uint64_t seed, uint32_t nsegs,
                   uint64_t histStart, uint64_t totalSweeps, uint64_t histEvery,
                   uint32_t samples, int L, int N, int mode,
                   MTL::Buffer* snaps) {
  if (nsegs == 0) return;
  MTL::CommandBuffer* cb = gpu.q->commandBuffer();
  MTL::ComputeCommandEncoder* en = cb->computeCommandEncoder();
  en->setComputePipelineState(gpu.pso);
  en->setBuffer(state, 0, 0);
  en->setBuffer(eh, 0, 1);
  en->setBuffer(al, 0, 2);
  en->setBuffer(an, 0, 3);
  uint32_t l = uint32_t(L), n = uint32_t(N), c = uint32_t(chains);
  en->setBytes(&l, 4, 4); en->setBytes(&n, 4, 5); en->setBytes(&c, 4, 6);
  en->setBuffer(segbuf, 0, 7);
  uint32_t ns = nsegs; en->setBytes(&ns, 4, 8);
  uint64_t sd = seed; en->setBytes(&sd, 8, 9);
  en->setBuffer(thbuf, 0, 10);
  uint32_t md = uint32_t(mode); en->setBytes(&md, 4, 11);
  en->setBuffer(pcbuf, 0, 12); en->setBuffer(nnbuf, 0, 13);
  uint32_t lemN = (0u - uint32_t(N)) % uint32_t(N);
  uint32_t lem6 = (0u - 6u) % 6u;
  en->setBytes(&lemN, 4, 14); en->setBytes(&lem6, 4, 15);
  uint64_t he = histEvery; en->setBytes(&he, 8, 16);
  uint64_t hs = histStart; en->setBytes(&hs, 8, 17);
  uint32_t smp = samples; en->setBytes(&smp, 4, 18);
  en->setBuffer(snaps, 0, 19);
  uint64_t ts = totalSweeps; en->setBytes(&ts, 8, 20);
  en->dispatchThreadgroups(MTL::Size::Make(chains, 1, 1), MTL::Size::Make(128, 1, 1));
  en->endEncoding(); cb->commit(); cb->waitUntilCompleted();
  if (cb->error()) { fprintf(stderr, "GPU error\n"); exit(1); }
}
#endif

int main(int argc, char** argv) {
  if (argc < 9) {
    fprintf(stderr, "usage: %s L beta rho rho1 rho2 NUM_SWEEPS NUM_CORES GPU nonlocal|local-colored EQUIL_SWEEPS [STATE_RAW] [--seed N] [--start-sweep N] [--history-every N] [--max-history-samples N]\n", argv[0]);
    return 2;
  }
  int L = atoi(argv[1]);
  const std::string betaText = argv[2];
  double beta = atof(argv[2]);
  double rho = atof(argv[3]), rho1 = atof(argv[4]), rho2 = atof(argv[5]);
  uint64_t sweeps = strtoull(argv[6], nullptr, 10);
  int cores = atoi(argv[7]);
  bool useGPU = std::string(argv[8]) == "GPU" || std::string(argv[8]) == "gpu" || std::string(argv[8]) == "1";
  uint64_t seed = 1, startSweep = 0, historyEvery = 1, maxHistorySamples = 100000;
  std::vector<std::string> positional;
  for (int i = 9; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.compare(0, 2, "--") != 0) {
      positional.push_back(arg);
      continue;
    }
    if (arg != "--seed" && arg != "--start-sweep" && arg != "--history-every" && arg != "--max-history-samples") {
      fprintf(stderr, "unknown option: %s\n", arg.c_str()); return 2;
    }
    if (++i == argc) { fprintf(stderr, "missing value for %s\n", arg.c_str()); return 2; }
    char* end = nullptr;
    errno = 0;
    uint64_t value = strtoull(argv[i], &end, 10);
    if (argv[i][0] == '-' || end == argv[i] || *end || errno == ERANGE ||
        (arg != "--seed" && arg != "--start-sweep" && value == 0)) {
      fprintf(stderr, "invalid value for %s: %s\n", arg.c_str(), argv[i]); return 2;
    }
    if (arg == "--seed") seed = value;
    else if (arg == "--start-sweep") startSweep = value;
    else if (arg == "--history-every") historyEvery = value;
    else maxHistorySamples = value;
  }
  if (positional.size() > 3) { fprintf(stderr, "too many positional arguments\n"); return 2; }
  std::string modeStr = positional.empty() ? "nonlocal" : positional[0];
  if (modeStr != "nonlocal" && modeStr != "local-colored") {
    fprintf(stderr, "persistent engine requires nonlocal or explicit local-colored; use driver_physical for physical local/both\n");
    return 2;
  }
  int mode = modeStr == "nonlocal" ? 1 : 2;
  uint64_t equil = positional.size() >= 2 ? strtoull(positional[1].c_str(), nullptr, 10) : 0;
  std::string statePath = positional.size() >= 3 ? positional[2] : "";
#if !SOLID_HAVE_METAL
  if (useGPU) {
    fprintf(stderr, "Metal backend not built\n");
    return 2;
  }
#endif
  if (!useGPU || L < 4 || L > 21 || cores < 1) {
    fprintf(stderr, "Metal requires GPU, 4<=L<=21 and positive chains\n"); return 2;
  }
  if (mode == 2 && L != 21) {
    fprintf(stderr, "local-colored requires L=21\n"); return 2;
  }
  if (fabs((rho1 + rho2) - rho) > 1e-8)
    fprintf(stderr, "WARNING: rho1+rho2=%.12g != rho=%.12g; using rho1/rho2 for composition\n", rho1 + rho2, rho);
  Lattice proto(L);
  int N = proto.N;
  // Bound both the absolute sweep and every proposal index before dispatch.
  const uint64_t sweepLimit = UINT64_MAX / (uint64_t(N) * (mode == 2 ? 3 : 2));
  if (startSweep > sweepLimit || equil > sweepLimit - startSweep ||
      sweeps > sweepLimit - startSweep - equil) {
    fprintf(stderr, "absolute sweep range exceeds persistent proposal-counter limit %llu\n",
            (unsigned long long)sweepLimit); return 2;
  }
  const uint64_t historyStart = startSweep + equil;
  const uint64_t nextSweep = historyStart + sweeps;
  int n1 = int(std::llround(rho1 * N));
  int n2 = int(std::llround(rho2 * N));
  if (n1 + n2 > N) {
    fprintf(stderr, "WARNING: n1+n2 > N; clamping n2\n");
    n2 = std::max(0, N - n1);
  }
  int nocc = n1 + n2;
  if (nocc <= 0) { fprintf(stderr, "empty occupied set\n"); return 2; }

  std::vector<Lattice> lats(cores, Lattice(L));
  for (int c = 0; c < cores; ++c) lats[c].init(seed + uint64_t(c) * 0x9e3779b97f4a7c15ULL, n1, n2);
  if (!statePath.empty()) {
    std::ifstream sf(statePath, std::ios::binary);
    if (!sf) { fprintf(stderr, "cannot open seed state %s\n", statePath.c_str()); return 2; }
    std::vector<uint8_t> raw((size_t)cores * N);
    sf.read((char*)raw.data(), raw.size());
    if (!sf) { fprintf(stderr, "short seed state %s\n", statePath.c_str()); return 2; }
    for (int c = 0; c < cores; ++c)
      std::copy(raw.begin() + size_t(c)*N, raw.begin() + size_t(c+1)*N, lats[c].st.begin());
  }
  std::vector<uint8_t> initial(size_t(cores) * N);
  for (int c = 0; c < cores; ++c)
    std::copy(lats[c].st.begin(), lats[c].st.end(), initial.begin() + size_t(c) * N);
  std::vector<uint8_t> equilibrated;

  std::vector<Threshold> th = makeThresholds(betaText);
  Hist hist;
  hist.every = std::max(historyEvery, sweeps / maxHistorySamples + uint64_t(sweeps % maxHistorySamples != 0));
  hist.sweeps = sweeps;
  uint64_t samples = sweeps / hist.every + uint64_t(sweeps % hist.every != 0);
  if (samples > UINT32_MAX) { fprintf(stderr, "too many Metal history samples\n"); return 2; }
  hist.energy.assign(size_t(cores) * samples, 0.0f);
  hist.accLocal.assign(size_t(cores) * samples, 0.0f);
  hist.accNonlocal.assign(size_t(cores) * samples, 0.0f);
  std::vector<uint64_t> historySweeps(samples);
  for (uint64_t i = 0; i < samples; ++i)
    historySweeps[i] = i + 1 == samples ? sweeps : (i + 1) * hist.every;

  std::vector<std::vector<uint8_t>> snapshots;
  std::vector<uint64_t> snapSweeps;
  auto snapshot = [&](uint64_t sw) {
    snapshots.emplace_back(size_t(cores) * N);
    for (int c = 0; c < cores; ++c)
      std::copy(lats[c].st.begin(), lats[c].st.end(), snapshots.back().begin() + size_t(c) * N);
    snapSweeps.push_back(sw);
  };

  std::vector<uint64_t> planned;
  for (uint64_t p = 1; p <= std::min<uint64_t>(sweeps, 1000000ULL); p <<= 1) planned.push_back(p);
  if (sweeps >= 1000000ULL) planned.push_back(1000000ULL);
  std::sort(planned.begin(), planned.end());
  planned.erase(std::unique(planned.begin(), planned.end()), planned.end());

#if SOLID_HAVE_METAL
  GPU* gpu = nullptr;
  MTL::Buffer* gstate = nullptr;
  MTL::Buffer* gth = nullptr;
  MTL::Buffer* gpc = nullptr;
  MTL::Buffer* gnn = nullptr;
  MTL::Buffer* ghistE = nullptr;
  MTL::Buffer* ghistL = nullptr;
  MTL::Buffer* ghistN = nullptr;
  if (useGPU) {
    gpu = new GPU(SOLID_MSL_PATH);
    gstate = gpu->dev->newBuffer(initial.data(), initial.size(), MTL::ResourceStorageModeShared);
    gth = gpu->dev->newBuffer(th.data(), th.size() * sizeof(Threshold), MTL::ResourceStorageModeShared);
    std::vector<uint32_t> pcv(static_cast<size_t>(N));
    std::vector<uint16_t> nnv(static_cast<size_t>(N) * 6);
    auto ix = [&](int x, int y, int z) { return x + L * y + L * L * z; };
    const int dirs[6][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
    for (int z = 0; z < L; ++z) for (int y = 0; y < L; ++y) for (int x = 0; x < L; ++x) {
      int i = ix(x, y, z);
      pcv[size_t(i)] = uint32_t(x | (y << 5) | (z << 10));
      for (int j = 0; j < 6; ++j)
        nnv[size_t(i) * 6 + j] = uint16_t(ix((x + dirs[j][0] + L) % L,
                                             (y + dirs[j][1] + L) % L,
                                             (z + dirs[j][2] + L) % L));
    }
    gpc = gpu->dev->newBuffer(pcv.data(), pcv.size() * 4, MTL::ResourceStorageModeShared);
    gnn = gpu->dev->newBuffer(nnv.data(), nnv.size() * 2, MTL::ResourceStorageModeShared);
    size_t histBytes = std::max<size_t>(1, size_t(cores) * samples * 4);
    ghistE = gpu->dev->newBuffer(histBytes, MTL::ResourceStorageModeShared);
    ghistL = gpu->dev->newBuffer(histBytes, MTL::ResourceStorageModeShared);
    ghistN = gpu->dev->newBuffer(histBytes, MTL::ResourceStorageModeShared);
  }
#endif

  std::vector<uint8_t> finalState(size_t(cores) * N);
  auto wall0 = std::chrono::steady_clock::now();
  bool ran = false;

#if SOLID_HAVE_METAL
  if (useGPU) {
    // ---- equilibration: one persistent dispatch, history discarded ----
    equilibrated.assign(size_t(cores) * N, 0);
    MTL::Buffer* gsegs = gpu->dev->newBuffer(sizeof(GPUSeg), MTL::ResourceStorageModeShared);
    MTL::Buffer* gsnaps = gpu->dev->newBuffer(std::max<size_t>(1, size_t(cores) * N),
                                              MTL::ResourceStorageModeShared);
    if (equil > 0) {
      GPUSeg eq{startSweep, equil, GPUSNAP_NONE, 0};
      std::memcpy(gsegs->contents(), &eq, sizeof(eq));
      gpuRun(*gpu, gstate, gth, gpc, gnn, ghistE, ghistL, ghistN, gsegs,
             cores, seed, 1, startSweep, 0, hist.every, uint32_t(samples), L, N, mode, gsnaps);
      const uint8_t* p = (const uint8_t*)gstate->contents();
      std::copy(p, p + size_t(cores) * N, equilibrated.begin());
    } else {
      for (int c = 0; c < cores; ++c)
        std::copy(initial.begin() + size_t(c) * N, initial.begin() + size_t(c + 1) * N,
                  equilibrated.begin() + size_t(c) * N);
    }

    wall0 = std::chrono::steady_clock::now();
    auto lastTimed = wall0;

    // ---- every planned snapshot interval plus the tail in one dispatch ----
    // Explicit local-colored uses the L21 schedule, not physical random edges.
    // Its recorded stationarity gate does not establish kinetic equivalence.
    std::vector<GPUSeg> segs;
    std::vector<uint64_t> plannedSnapSweeps;
    uint64_t cur = 0;
    for (uint64_t b : planned) {
      if (b <= cur || b > sweeps) continue;
      segs.push_back(GPUSeg{historyStart + cur, b - cur, uint32_t(plannedSnapSweeps.size()), 1});
      plannedSnapSweeps.push_back(b);
      cur = b;
    }
    if (sweeps <= 1000000ULL && cur < sweeps) {
      segs.push_back(GPUSeg{historyStart + cur, sweeps - cur, GPUSNAP_NONE, 1});
      cur = sweeps;
    }
    gsegs->release();
    gsegs = gpu->dev->newBuffer(std::max<size_t>(1, segs.size() * sizeof(GPUSeg)),
                                MTL::ResourceStorageModeShared);
    if (!segs.empty())
      std::memcpy(gsegs->contents(), segs.data(), segs.size() * sizeof(GPUSeg));
    size_t slotCount = std::max<size_t>(1, plannedSnapSweeps.size());
    gsnaps->release();
    gsnaps = gpu->dev->newBuffer(slotCount * size_t(cores) * N,
                                 MTL::ResourceStorageModeShared);
    gpuRun(*gpu, gstate, gth, gpc, gnn, ghistE, ghistL, ghistN, gsegs,
           cores, seed, uint32_t(segs.size()), historyStart, sweeps, hist.every,
           uint32_t(samples), L, N, mode, gsnaps);

    snapshots.clear();
    snapSweeps.clear();
    {
      const uint8_t* p = (const uint8_t*)gsnaps->contents();
      for (size_t i = 0; i < plannedSnapSweeps.size(); ++i) {
        snapshots.emplace_back(size_t(cores) * N);
        std::copy(p + i * size_t(cores) * N, p + (i + 1) * size_t(cores) * N,
                  snapshots.back().begin());
      }
      snapSweeps = plannedSnapSweeps;
    }

    // ---- after 1M sweeps: wall-clock checkpoints, one chunk per dispatch ----
    if (sweeps > 1000000ULL) {
      const uint64_t chunk = 1024;
      MTL::Buffer* gone = gpu->dev->newBuffer(sizeof(GPUSeg), MTL::ResourceStorageModeShared);
      while (cur < sweeps) {
        uint64_t n = std::min<uint64_t>(chunk, sweeps - cur);
        GPUSeg one{historyStart + cur, n, GPUSNAP_NONE, 1};
        std::memcpy(gone->contents(), &one, sizeof(one));
        gpuRun(*gpu, gstate, gth, gpc, gnn, ghistE, ghistL, ghistN, gone,
               cores, seed, 1, historyStart, sweeps, hist.every, uint32_t(samples),
               L, N, mode, gsnaps);
        cur += n;
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - lastTimed).count() >= 600.0) {
          snapshots.emplace_back(size_t(cores) * N);
          const uint8_t* p = (const uint8_t*)gstate->contents();
          std::copy(p, p + size_t(cores) * N, snapshots.back().begin());
          snapSweeps.push_back(cur);
          lastTimed = now;
        }
      }
    }

    if (samples > 0) {
      const uint32_t* ep = (const uint32_t*)ghistE->contents();
      const uint32_t* lp = (const uint32_t*)ghistL->contents();
      const uint32_t* np = (const uint32_t*)ghistN->contents();
      size_t totalv = size_t(cores) * samples;
      for (size_t i = 0; i < totalv; ++i) {
        hist.energy[i] = float(ep[i]);
        hist.accLocal[i] = float(lp[i]) / float(N);
        hist.accNonlocal[i] = float(np[i]) / float(N);
      }
    }
    {
      const uint8_t* p = (const uint8_t*)gstate->contents();
      for (int c = 0; c < cores; ++c)
        std::copy(p + size_t(c) * N, p + size_t(c + 1) * N, lats[c].st.begin());
    }
    ran = true;
  }
#endif
  if (!ran) {
    auto runCpuSeg = [&](uint64_t start, uint64_t count, uint64_t histStart,
                         Hist& target, uint64_t total) {
      while (count) {
        uint64_t chunk = std::min<uint64_t>(count, 4096);
        cpuSegment(lats, cores, seed, start, chunk, histStart, th, target, total, nocc, mode);
        start += chunk;
        count -= chunk;
      }
    };
    if (equil > 0) {
      Hist eqh;
      runCpuSeg(startSweep, equil, startSweep, eqh, 0);
    }
    equilibrated.assign(size_t(cores) * N, 0);
    for (int c = 0; c < cores; ++c)
      std::copy(lats[c].st.begin(), lats[c].st.end(), equilibrated.begin() + size_t(c) * N);

    wall0 = std::chrono::steady_clock::now();
    auto lastTimed = wall0;
    uint64_t cur = 0;
    for (uint64_t b : planned) {
      if (b <= cur || b > sweeps) continue;
      runCpuSeg(historyStart + cur, b - cur, historyStart, hist, samples);
      cur = b;
      snapshot(cur);
    }
    // After 1M sweeps: snapshot at least every 10 minutes.
    if (sweeps > 1000000ULL) {
      const uint64_t chunk = 1024;
      while (cur < sweeps) {
        uint64_t n = std::min<uint64_t>(chunk, sweeps - cur);
        runCpuSeg(historyStart + cur, n, historyStart, hist, samples);
        cur += n;
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - lastTimed).count() >= 600.0) {
          snapshot(cur);
          lastTimed = now;
        }
      }
    } else if (cur < sweeps) {
      runCpuSeg(historyStart + cur, sweeps - cur, historyStart, hist, samples);
      cur = sweeps;
    }
  }
  for (int c = 0; c < cores; ++c)
    std::copy(lats[c].st.begin(), lats[c].st.end(), finalState.begin() + size_t(c) * N);

  // Aggregate into .npz
  init_crc();
  std::ofstream z("run.npz", std::ios::binary);
  std::vector<ZipEntry> entries;
  zipAdd(z, entries, "initial.npy", "<u1", {size_t(cores), size_t(N)}, initial);
  zipAdd(z, entries, "equilibrated.npy", "<u1", {size_t(cores), size_t(N)}, equilibrated);
  zipAdd(z, entries, "final.npy", "<u1", {size_t(cores), size_t(N)}, finalState);
  zipAdd(z, entries, "snapshot_sweeps.npy", "<u8", {snapSweeps.size()},
         asBytes(snapSweeps.data(), snapSweeps.size() * 8));
  {
    std::vector<uint8_t> all; all.reserve(snapshots.size() * size_t(cores) * N);
    for (auto& s : snapshots) all.insert(all.end(), s.begin(), s.end());
    zipAdd(z, entries, "snapshots.npy", "<u1",
           {snapshots.size(), size_t(cores), size_t(N)}, all);
  }
  zipAdd(z, entries, "energy.npy", "<f4", {size_t(cores), size_t(samples)},
         asBytes(hist.energy.data(), hist.energy.size() * 4));
  zipAdd(z, entries, "accept_local.npy", "<f4", {size_t(cores), size_t(samples)},
         asBytes(hist.accLocal.data(), hist.accLocal.size() * 4));
  zipAdd(z, entries, "accept_nonlocal.npy", "<f4", {size_t(cores), size_t(samples)},
         asBytes(hist.accNonlocal.data(), hist.accNonlocal.size() * 4));
  zipAdd(z, entries, "history_sweeps.npy", "<u8", {historySweeps.size()},
         asBytes(historySweeps.data(), historySweeps.size() * 8));
  zipAdd(z, entries, "history_every.npy", "<u8", {1}, asBytes(&hist.every, sizeof(hist.every)));
  zipAdd(z, entries, "seed.npy", "<u8", {1}, asBytes(&seed, sizeof(seed)));
  zipAdd(z, entries, "start_sweep.npy", "<u8", {1}, asBytes(&startSweep, sizeof(startSweep)));
  zipAdd(z, entries, "next_sweep.npy", "<u8", {1}, asBytes(&nextSweep, sizeof(nextSweep)));
  double scal[4] = {beta, rho, rho1, rho2};
  zipAdd(z, entries, "scalars.npy", "<f8", {4}, asBytes(scal, sizeof(scal)));
  uint64_t meta[5] = {(uint64_t)L, (uint64_t)cores, sweeps, (uint64_t)mode, equil};
  zipAdd(z, entries, "meta.npy", "<u8", {5}, asBytes(meta, sizeof(meta)));
  zipFinish(z, entries);
  z.close();
  double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
  printf("done L=%d chains=%d sweeps=%llu beta=%.3f wall=%.3fs snapshots=%zu npz=run.npz\n",
         L, cores, (unsigned long long)sweeps, beta, wall, snapshots.size());
  return 0;
}
