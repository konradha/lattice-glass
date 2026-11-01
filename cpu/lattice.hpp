#pragma once
// CPU lattice-glass engine, fixed-beta chains, history, and checkpoints.
// L=25, 1-byte state (2-bit type + 6-bit neighbor mask), fixed composition.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <signal.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace lg {

constexpr int L = 25;
constexpr int N = L * L * L;
constexpr int N1 = 4688;
constexpr int N2 = 7031;

// ---------------------------------------------------------------------------
// RNG: xoshiro256++ (explicit state, checkpointable)
// ---------------------------------------------------------------------------
struct Rng {
  uint64_t s[4];
  static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
  static uint64_t splitmix(uint64_t& x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
  explicit Rng(uint64_t seed = 1) { uint64_t x = seed; for (auto& v : s) v = splitmix(x); }
  uint64_t next() {
    uint64_t r = rotl(s[0] + s[3], 23) + s[0];
    uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return r;
  }
  uint32_t bounded(uint32_t range) {
    uint64_t x = next();
    __uint128_t m = static_cast<__uint128_t>(x) * range;
    uint64_t l = static_cast<uint64_t>(m);
    if (l < range) {
      uint64_t t = (0 - static_cast<uint64_t>(range)) % range;
      while (l < t) {
        x = next();
        m = static_cast<__uint128_t>(x) * range;
        l = static_cast<uint64_t>(m);
      }
    }
    return static_cast<uint32_t>(m >> 64);
  }
  double uniform() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }
};

// ---------------------------------------------------------------------------
// Geometry and energy
// ---------------------------------------------------------------------------
inline int nb(int s, int j) {
  int k = s % L, t = s / L, jj = t % L, ii = t / L;
  const int di = (j == 1) - (j == 0);
  const int dj = (j == 3) - (j == 2);
  const int dk = (j == 5) - (j == 4);
  ii = (ii + di + L) % L;
  jj = (jj + dj + L) % L;
  k = (k + dk + L) % L;
  return k + L * (jj + ii * L);
}
inline int manhattan(int a, int b) {
  int ia = a / (L*L), ja = (a / L) % L, ka = a % L;
  int ib = b / (L*L), jb = (b / L) % L, kb = b % L;
  int dx = ia - ib; if (dx < 0) dx = -dx; if (dx > L - dx) dx = L - dx;
  int dy = ja - jb; if (dy < 0) dy = -dy; if (dy > L - dy) dy = L - dy;
  int dz = ka - kb; if (dz < 0) dz = -dz; if (dz > L - dz) dz = L - dz;
  return dx + dy + dz;
}
inline int local_energy(int t, int m) {
  if (t == 0) return 0;
  const int conn = (t == 1) ? 3 : 5;
  const int v = __builtin_popcount(static_cast<unsigned>(m)) - conn;
  return v * v;
}

struct State {
  std::vector<uint8_t> st;
  State() : st(N, 0) {}
  explicit State(uint64_t seed) : st(N, 0) { init(seed); }
  void init(uint64_t seed) {
    std::vector<int> sites(N);
    for (int i=0;i<N;i++) sites[i]=i;
    std::mt19937 gen(static_cast<uint32_t>(seed));
    std::shuffle(sites.begin(), sites.end(), gen);
    for (int i=0;i<N1;i++) st[sites[i]] = 1;
    for (int i=0;i<N2;i++) st[sites[N1+i]] = 2;
    for (int s=0;s<N;s++) {
      int m = 0;
      for (int j=0;j<6;j++) if ((st[nb(s,j)] & 3) != 0) m |= (1<<j);
      st[s] = static_cast<uint8_t>((st[s] & 3) | (m << 2));
    }
  }
  int full_energy() const {
    int e = 0;
    for (int s=0;s<N;s++) e += local_energy(st[s] & 3, st[s] >> 2);
    return e;
  }
};

struct Aff { int s; uint8_t ba, bb, ns; };

// Fast path: 14 distinct sites, Manhattan distance >= 3.
inline int delta_far(State& S, int a, int b, Aff* aff) {
  const int sites[14] = {a, b,
    nb(a,0), nb(a,1), nb(a,2), nb(a,3), nb(a,4), nb(a,5),
    nb(b,0), nb(b,1), nb(b,2), nb(b,3), nb(b,4), nb(b,5)};
  const uint8_t ba[14] = {0,0,2,1,4,3,6,5,0,0,0,0,0,0};
  const uint8_t bb[14] = {0,0,0,0,0,0,0,0,2,1,4,3,6,5};
  const int ta = S.st[a] & 3, tb = S.st[b] & 3;
  const int oa = (tb != 0), ob = (ta != 0);
  int old_e = 0, new_e = 0;
  for (int k=0;k<14;k++) {
    const int s = sites[k]; const uint8_t v = S.st[s];
    const int ts = v & 3, ms = v >> 2;
    old_e += local_energy(ts, ms);
    int nm = ms;
    if (ba[k]) { const int bit = ba[k]-1; nm = (nm & ~(1<<bit)) | (oa<<bit); }
    if (bb[k]) { const int bit = bb[k]-1; nm = (nm & ~(1<<bit)) | (ob<<bit); }
    const int nt = (s == a) ? tb : ((s == b) ? ta : ts);
    aff[k].s = s; aff[k].ba = ba[k]; aff[k].bb = bb[k];
    aff[k].ns = static_cast<uint8_t>(nt | (nm << 2));
    new_e += local_energy(nt, nm);
  }
  return new_e - old_e;
}
inline void commit_far(State& S, const Aff* aff) {
  for (int k=0;k<14;k++) S.st[aff[k].s] = aff[k].ns;
}

// Generic path: deduplicated affected set.
inline void add_aff(Aff* aff, int& na, int s, int which, int bit) {
  for (int i=0;i<na;i++) if (aff[i].s == s) {
    if (which == 0) aff[i].ba = static_cast<uint8_t>(bit+1);
    if (which == 1) aff[i].bb = static_cast<uint8_t>(bit+1);
    return;
  }
  aff[na].s = s;
  aff[na].ba = (which == 0) ? static_cast<uint8_t>(bit+1) : 0;
  aff[na].bb = (which == 1) ? static_cast<uint8_t>(bit+1) : 0;
  aff[na].ns = 0;
  ++na;
}
inline int delta_generic(State& S, int a, int b, Aff* aff, int& na) {
  na = 0;
  add_aff(aff, na, a, 2, 0);
  add_aff(aff, na, b, 2, 0);
  for (int j=0;j<6;j++) add_aff(aff, na, nb(a,j), 0, j^1);
  for (int j=0;j<6;j++) add_aff(aff, na, nb(b,j), 1, j^1);
  const int ta = S.st[a] & 3, tb = S.st[b] & 3;
  const int oa = (tb != 0), ob = (ta != 0);
  int old_e = 0, new_e = 0;
  for (int i=0;i<na;i++) {
    const int s = aff[i].s; const uint8_t v = S.st[s];
    const int ts = v & 3, ms = v >> 2;
    old_e += local_energy(ts, ms);
    int nm = ms;
    if (aff[i].ba) { const int bit = aff[i].ba-1; nm = (nm & ~(1<<bit)) | (oa<<bit); }
    if (aff[i].bb) { const int bit = aff[i].bb-1; nm = (nm & ~(1<<bit)) | (ob<<bit); }
    const int nt = (s == a) ? tb : ((s == b) ? ta : ts);
    aff[i].ns = static_cast<uint8_t>(nt | (nm << 2));
    new_e += local_energy(nt, nm);
  }
  return new_e - old_e;
}
inline void commit_generic(State& S, const Aff* aff, int na) {
  for (int i=0;i<na;i++) S.st[aff[i].s] = aff[i].ns;
}

// ---------------------------------------------------------------------------
// Metropolis move
// ---------------------------------------------------------------------------
struct ExpTable {
  double beta;
  std::array<double, 1024> t;
  explicit ExpTable(double b = 1.0) : beta(b) {
    for (int i=0;i<1024;i++) t[i] = std::exp(-b * static_cast<double>(i - 512));
  }
  bool accept(int dE, double u) const {
    if (dE <= 0) return true;
    const int idx = dE + 512;
    if (idx < 1024) return u < t[idx];
    return u < std::exp(-beta * static_cast<double>(dE));
  }
};

inline bool try_move(State& S, Rng& rng, const ExpTable& table, int local_pct,
                     int& energy) {
  const int a = static_cast<int>(rng.bounded(N));
  int b;
  bool local = false;
  if (local_pct > 0 && static_cast<int>(rng.bounded(100)) < local_pct) {
    b = nb(a, static_cast<int>(rng.bounded(6)));
    local = true;
  } else {
    do { b = (a + 1 + static_cast<int>(rng.bounded(N - 1))) % N; }
    while (manhattan(a, b) < 3);
  }
  if ((S.st[a] & 3) == (S.st[b] & 3)) return false;

  int dE;
  Aff aff[14];
  int na = 0;
  if (!local && manhattan(a, b) >= 3) {
    dE = delta_far(S, a, b, aff);
  } else {
    dE = delta_generic(S, a, b, aff, na);
  }
  bool ok = (dE <= 0);
  if (!ok) ok = table.accept(dE, rng.uniform());
  if (!ok) return false;
  if (!local && manhattan(a, b) >= 3) commit_far(S, aff);
  else commit_generic(S, aff, na);
  energy += dE;
  return true;
}

// ---------------------------------------------------------------------------
// Fixed-beta single-chain sampler
// ---------------------------------------------------------------------------
struct SamplerConfig {
  double beta = 2.0;
  int local_pct = 50;
  int sample_every = 10;
  int hist_bins = 8192;
  int hist_width = 64;
  uint64_t seed = 1;
  uint64_t total_sweeps = 1000000;
};

struct Sampler {
  SamplerConfig cfg;
  State state;
  Rng rng;
  ExpTable table;
  int energy = 0;
  int accepted_total = 0;
  int accepted_last = 0;
  std::vector<uint32_t> hist;
  uint64_t sweeps_done = 0;
  uint64_t samples_done = 0;

  explicit Sampler(const SamplerConfig& c)
      : cfg(c), state(c.seed), rng(c.seed ^ 0x9e3779b97f4a7c15ULL),
        table(c.beta), hist(c.hist_bins, 0) {
    energy = state.full_energy();
  }

  int sweep() {
    int accepted = 0;
    for (int i=0;i<N;i++) if (try_move(state, rng, table, cfg.local_pct, energy)) ++accepted;
    accepted_last = accepted;
    accepted_total += accepted;
    ++sweeps_done;
    if (cfg.sample_every > 0 && sweeps_done % cfg.sample_every == 0) sample();
    return accepted;
  }

  void sample() {
    int bin = energy / cfg.hist_width;
    if (bin < 0) bin = 0;
    if (bin >= cfg.hist_bins) bin = cfg.hist_bins - 1;
    ++hist[bin];
    ++samples_done;
  }
};

// ---------------------------------------------------------------------------
// Checkpoint (atomic, checksummed)
// ---------------------------------------------------------------------------
inline uint64_t fnv1a(const uint8_t* p, size_t n) {
  uint64_t h = 1469598103934665603ULL;
  for (size_t i=0;i<n;i++) { h ^= p[i]; h *= 1099511628211ULL; }
  return h;
}

struct CheckpointHeader {
  uint64_t magic = 0x4c47534331ULL; // LGSC1
  uint32_t version = 4;
  uint32_t hist_bins = 0;
  uint64_t sweeps_done = 0;
  uint64_t samples_done = 0;
  uint64_t seed = 0;
  uint32_t hist_width = 0;
  int32_t local_pct = 0;
  double beta = 0.0;
  uint64_t total_sweeps = 0;
  uint64_t payload_bytes = 0;
  uint64_t checksum = 0;
};

inline void write_all(int fd, const void* data, size_t n) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  while (n) {
    ssize_t w = ::write(fd, p, n);
    if (w <= 0) throw std::runtime_error("write failed");
    p += w; n -= static_cast<size_t>(w);
  }
}
inline void read_all(int fd, void* data, size_t n) {
  uint8_t* p = static_cast<uint8_t*>(data);
  while (n) {
    ssize_t r = ::read(fd, p, n);
    if (r <= 0) throw std::runtime_error("read failed");
    p += r; n -= static_cast<size_t>(r);
  }
}

inline void save_checkpoint(const std::string& path, const Sampler& s) {
  std::vector<uint8_t> payload;
  auto put = [&](const void* p, size_t n) {
    const uint8_t* q = static_cast<const uint8_t*>(p);
    payload.insert(payload.end(), q, q + n);
  };
  const int32_t energy = s.energy;
  const int64_t accepted_total = s.accepted_total;
  put(&energy, sizeof(energy));
  put(&accepted_total, sizeof(accepted_total));
  put(s.state.st.data(), s.state.st.size());
  put(s.rng.s, sizeof(s.rng.s));
  put(s.hist.data(), s.hist.size() * sizeof(uint32_t));

  CheckpointHeader h;
  h.hist_bins = static_cast<uint32_t>(s.cfg.hist_bins);
  h.sweeps_done = s.sweeps_done;
  h.samples_done = s.samples_done;
  h.seed = s.cfg.seed;
  h.hist_width = static_cast<uint32_t>(s.cfg.hist_width);
  h.local_pct = s.cfg.local_pct;
  h.beta = s.cfg.beta;
  h.total_sweeps = s.cfg.total_sweeps;
  h.payload_bytes = payload.size();
  h.checksum = fnv1a(payload.data(), payload.size());

  const std::string tmp = path + ".tmp";
  const std::string prev = path + ".prev";
  int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
  if (fd < 0) throw std::runtime_error("open temp checkpoint failed");
  try {
    write_all(fd, &h, sizeof(h));
    write_all(fd, payload.data(), payload.size());
    ::fsync(fd);
    ::close(fd);
  } catch (...) {
    ::close(fd); ::unlink(tmp.c_str()); throw;
  }
  ::rename(path.c_str(), prev.c_str());
  if (::rename(tmp.c_str(), path.c_str()) != 0) throw std::runtime_error("rename checkpoint failed");
  const std::string dir = std::filesystem::path(path).parent_path().string();
  int dfd = ::open(dir.empty() ? "." : dir.c_str(), O_RDONLY);
  if (dfd >= 0) { ::fsync(dfd); ::close(dfd); }
}

inline bool load_checkpoint(const std::string& path, Sampler& s) {
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) return false;
  CheckpointHeader h;
  try {
    read_all(fd, &h, sizeof(h));
    if (h.magic != 0x4c47534331ULL || h.version != 4) { ::close(fd); return false; }
    std::vector<uint8_t> payload(h.payload_bytes);
    read_all(fd, payload.data(), payload.size());
    ::close(fd);
    if (fnv1a(payload.data(), payload.size()) != h.checksum) return false;

    SamplerConfig c;
    c.beta = h.beta;
    c.local_pct = h.local_pct;
    c.hist_bins = static_cast<int>(h.hist_bins);
    c.hist_width = static_cast<int>(h.hist_width);
    c.seed = h.seed;
    c.total_sweeps = h.total_sweeps;
    s = Sampler(c);
    s.sweeps_done = h.sweeps_done;
    s.samples_done = h.samples_done;
    size_t off = 0;
    auto get = [&](void* p, size_t n) {
      if (off + n > payload.size()) throw std::runtime_error("truncated checkpoint");
      std::memcpy(p, payload.data() + off, n); off += n;
    };
    get(&s.energy, sizeof(int32_t));
    int64_t accepted_total = 0;
    get(&accepted_total, sizeof(accepted_total));
    s.accepted_total = static_cast<int>(accepted_total);
    get(s.state.st.data(), s.state.st.size());
    get(s.rng.s, sizeof(s.rng.s));
    get(s.hist.data(), s.hist.size() * sizeof(uint32_t));
    return true;
  } catch (...) {
    ::close(fd);
    return false;
  }
}

// ---------------------------------------------------------------------------
// Output helpers
// ---------------------------------------------------------------------------
inline void save_configs(const std::string& dir, const Sampler& s) {
  std::filesystem::create_directories(dir);
  const std::string p = dir + "/config.bin";
  const std::string tmp = p + ".tmp";
  int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
  if (fd < 0) throw std::runtime_error("open config failed");
  write_all(fd, s.state.st.data(), s.state.st.size());
  ::fsync(fd); ::close(fd);
  ::rename(tmp.c_str(), p.c_str());
}

inline void save_histograms(const std::string& path, const Sampler& s) {
  std::ofstream out(path);
  out << "beta,bin,energy,count\n";
  for (int b=0;b<s.cfg.hist_bins;b++) {
    if (s.hist[b] == 0) continue;
    out << s.cfg.beta << ',' << b << ','
        << (static_cast<double>(b) + 0.5) * s.cfg.hist_width << ','
        << s.hist[b] << '\n';
  }
}

inline bool verify_kernel(int checks, uint64_t seed) {
  State S(seed);
  Rng r(seed + 1);
  for (int c=0;c<checks;c++) {
    const int a = static_cast<int>(r.bounded(N));
    int b;
    do { b = (a + 1 + static_cast<int>(r.bounded(N-1))) % N; } while (manhattan(a,b) < 3);
    const int e0 = S.full_energy();
    Aff aff[14];
    const int dE = delta_far(S, a, b, aff);
    commit_far(S, aff);
    if (S.full_energy() - e0 != dE) return false;
    delta_far(S, a, b, aff); commit_far(S, aff);
    if (S.full_energy() != e0) return false;
  }
  return true;
}

} // namespace lg
