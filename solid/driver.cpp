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

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "Metal/Metal.hpp"

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
    NS::String* fn = NS::String::string("run_segment", NS::UTF8StringEncoding);
    MTL::Function* f = lib->newFunction(fn);
    pso = dev->newComputePipelineState(f, &err);
    if (!pso) { fprintf(stderr, "pipeline: %s\n", err ? err->localizedDescription()->utf8String() : "?"); exit(1); }
  }
};

struct Hist { std::vector<float> energy, accLocal, accNonlocal; };

static void cpuSegment(std::vector<Lattice>& lats, int chains, uint64_t seed,
                       uint64_t start, uint64_t count,
                       const std::vector<uint32_t>& th,
                       Hist& hist, uint64_t total, int nocc) {
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
        int an = nonlocalSweep(lat, seed, c, sw, th, e);
        int al = localSweep(lat, seed, c, sw, th, e);
        uint64_t idx = uint64_t(c) * total + sw;
        e = lat.energy();
        hist.energy[idx] = float(e) / float(nocc);
        hist.accNonlocal[idx] = float(an) / float(lat.N);
        hist.accLocal[idx] = float(al) / float(lat.N);
      }
    }
  });
  for (auto& t : ts) t.join();
}

static void gpuSegment(GPU& gpu, MTL::Buffer* state, MTL::Buffer* thbuf,
                       int chains, uint64_t seed, uint64_t start, uint64_t count,
                       Hist& hist, uint64_t total, int L, int N, int nocc) {
  size_t hs = size_t(chains) * count;
  MTL::Buffer* eh = gpu.dev->newBuffer(hs * 4, MTL::ResourceStorageModeShared);
  MTL::Buffer* al = gpu.dev->newBuffer(hs * 4, MTL::ResourceStorageModeShared);
  MTL::Buffer* an = gpu.dev->newBuffer(hs * 4, MTL::ResourceStorageModeShared);
  MTL::CommandBuffer* cb = gpu.q->commandBuffer();
  MTL::ComputeCommandEncoder* en = cb->computeCommandEncoder();
  en->setComputePipelineState(gpu.pso);
  en->setBuffer(state, 0, 0); en->setBuffer(eh, 0, 1);
  en->setBuffer(al, 0, 2); en->setBuffer(an, 0, 3);
  uint32_t l = L, n = N, c = chains, s = uint32_t(start), k = uint32_t(count);
  uint64_t sd = seed;
  en->setBytes(&l, 4, 4); en->setBytes(&n, 4, 5); en->setBytes(&c, 4, 6);
  en->setBytes(&s, 4, 7); en->setBytes(&k, 4, 8); en->setBytes(&sd, 8, 9);
  en->setBuffer(thbuf, 0, 10);
  en->dispatchThreadgroups(MTL::Size::Make(chains, 1, 1), MTL::Size::Make(1, 1, 1));
  en->endEncoding(); cb->commit(); cb->waitUntilCompleted();
  if (cb->error()) { fprintf(stderr, "GPU error\n"); exit(1); }
  uint32_t* ep = (uint32_t*)eh->contents();
  uint32_t* ap = (uint32_t*)al->contents();
  uint32_t* np = (uint32_t*)an->contents();
  for (int ch = 0; ch < chains; ++ch) for (uint64_t j = 0; j < count; ++j) {
    uint64_t idx = uint64_t(ch) * total + start + j;
    hist.energy[idx] = float(ep[ch * count + j]) / float(nocc);
    hist.accLocal[idx] = float(ap[ch * count + j]) / float(N);
    hist.accNonlocal[idx] = float(np[ch * count + j]) / float(N);
  }
  eh->release(); al->release(); an->release();
}

int main(int argc, char** argv) {
  if (argc != 9) {
    fprintf(stderr, "usage: %s L beta rho rho1 rho2 NUM_SWEEPS NUM_CORES CPU|GPU\n", argv[0]);
    return 2;
  }
  int L = atoi(argv[1]);
  double beta = atof(argv[2]);
  double rho = atof(argv[3]), rho1 = atof(argv[4]), rho2 = atof(argv[5]);
  uint64_t sweeps = strtoull(argv[6], nullptr, 10);
  int cores = atoi(argv[7]);
  bool useGPU = std::string(argv[8]) == "GPU" || std::string(argv[8]) == "gpu" || std::string(argv[8]) == "1";
  if (L < 4 || cores < 1) { fprintf(stderr, "bad L/cores\n"); return 2; }
  if (fabs((rho1 + rho2) - rho) > 1e-8)
    fprintf(stderr, "WARNING: rho1+rho2=%.12g != rho=%.12g; using rho1/rho2 for composition\n", rho1 + rho2, rho);
  Lattice proto(L);
  int N = proto.N;
  int n1 = int(std::llround(rho1 * N));
  int n2 = int(std::llround(rho2 * N));
  if (n1 + n2 > N) {
    fprintf(stderr, "WARNING: n1+n2 > N; clamping n2\n");
    n2 = std::max(0, N - n1);
  }
  int nocc = n1 + n2;
  if (nocc <= 0) { fprintf(stderr, "empty occupied set\n"); return 2; }

  const uint64_t seed = 1;
  std::vector<Lattice> lats(cores, Lattice(L));
  for (int c = 0; c < cores; ++c) lats[c].init(seed + uint64_t(c) * 0x9e3779b97f4a7c15ULL, n1, n2);
  std::vector<uint8_t> initial(size_t(cores) * N);
  for (int c = 0; c < cores; ++c)
    std::copy(lats[c].st.begin(), lats[c].st.end(), initial.begin() + size_t(c) * N);

  std::vector<uint32_t> th = makeThresholds(beta);
  Hist hist; hist.energy.assign(size_t(cores) * sweeps, 0.0f);
  hist.accLocal.assign(size_t(cores) * sweeps, 0.0f);
  hist.accNonlocal.assign(size_t(cores) * sweeps, 0.0f);

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

  GPU* gpu = nullptr;
  MTL::Buffer* gstate = nullptr;
  MTL::Buffer* gth = nullptr;
  if (useGPU) {
#if defined(SOLID_MSL_PATH)
    gpu = new GPU(SOLID_MSL_PATH);
#else
    gpu = new GPU("solid/kernels.metal");
#endif
    gstate = gpu->dev->newBuffer(initial.data(), initial.size(), MTL::ResourceStorageModeShared);
    gth = gpu->dev->newBuffer(th.data(), th.size() * 4, MTL::ResourceStorageModeShared);
  }

  auto runSeg = [&](uint64_t start, uint64_t count) {
    if (!count) return;
    if (useGPU) {
      for (uint64_t k = 0; k < count; ++k) {
        gpuSegment(*gpu, gstate, gth, cores, seed, start + k, 1, hist, sweeps, L, N, nocc);
        uint8_t* p = (uint8_t*)gstate->contents();
        for (int c = 0; c < cores; ++c) {
          std::copy(p + size_t(c) * N, p + size_t(c + 1) * N, lats[c].st.begin());
          hist.energy[size_t(c) * sweeps + start + k] = float(lats[c].energy()) / float(nocc);
        }
      }
    } else {
      cpuSegment(lats, cores, seed, start, count, th, hist, sweeps, nocc);
    }
  };

  uint64_t cur = 0;
  auto wall0 = std::chrono::steady_clock::now();
  auto lastTimed = wall0;
  for (uint64_t b : planned) {
    if (b <= cur || b > sweeps) continue;
    runSeg(cur, b - cur);
    cur = b;
    snapshot(cur);
  }
  // After 1M sweeps: snapshot at least every 10 minutes.
  if (sweeps > 1000000ULL) {
    const uint64_t chunk = 1024;
    while (cur < sweeps) {
      uint64_t n = std::min<uint64_t>(chunk, sweeps - cur);
      runSeg(cur, n);
      cur += n;
      auto now = std::chrono::steady_clock::now();
      if (std::chrono::duration<double>(now - lastTimed).count() >= 600.0) {
        snapshot(cur);
        lastTimed = now;
      }
    }
  } else if (cur < sweeps) {
    runSeg(cur, sweeps - cur);
    cur = sweeps;
  }
  std::vector<uint8_t> finalState(size_t(cores) * N);
  for (int c = 0; c < cores; ++c)
    std::copy(lats[c].st.begin(), lats[c].st.end(), finalState.begin() + size_t(c) * N);

  // Aggregate into .npz
  init_crc();
  std::ofstream z("run.npz", std::ios::binary);
  std::vector<ZipEntry> entries;
  zipAdd(z, entries, "initial.npy", "<u1", {size_t(cores), size_t(N)}, initial);
  zipAdd(z, entries, "final.npy", "<u1", {size_t(cores), size_t(N)}, finalState);
  zipAdd(z, entries, "snapshot_sweeps.npy", "<u8", {snapSweeps.size()},
         asBytes(snapSweeps.data(), snapSweeps.size() * 8));
  {
    std::vector<uint8_t> all; all.reserve(snapshots.size() * size_t(cores) * N);
    for (auto& s : snapshots) all.insert(all.end(), s.begin(), s.end());
    zipAdd(z, entries, "snapshots.npy", "<u1",
           {snapshots.size(), size_t(cores), size_t(N)}, all);
  }
  zipAdd(z, entries, "energy.npy", "<f4", {size_t(cores), size_t(sweeps)},
         asBytes(hist.energy.data(), hist.energy.size() * 4));
  zipAdd(z, entries, "accept_local.npy", "<f4", {size_t(cores), size_t(sweeps)},
         asBytes(hist.accLocal.data(), hist.accLocal.size() * 4));
  zipAdd(z, entries, "accept_nonlocal.npy", "<f4", {size_t(cores), size_t(sweeps)},
         asBytes(hist.accNonlocal.data(), hist.accNonlocal.size() * 4));
  double scal[4] = {beta, rho, rho1, rho2};
  zipAdd(z, entries, "scalars.npy", "<f8", {4}, asBytes(scal, sizeof(scal)));
  uint64_t meta[3] = {(uint64_t)L, (uint64_t)cores, sweeps};
  zipAdd(z, entries, "meta.npy", "<u8", {3}, asBytes(meta, sizeof(meta)));
  zipFinish(z, entries);
  z.close();
  double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
  printf("done L=%d chains=%d sweeps=%llu beta=%.3f wall=%.3fs snapshots=%zu npz=run.npz\n",
         L, cores, (unsigned long long)sweeps, beta, wall, snapshots.size());
  return 0;
}
