// Time-boxed kernel research for the NH lattice glass.
// Compares the current packed/bit-twiddling move against a byte + neighbor-mask
// + local-energy-table move, and compares RNG / acceptance-table choices.
//
// Build: clang++ -O3 -march=native -std=c++17 -DNDEBUG bench.cpp -o bench

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// RNGs. All expose: uint32_t bounded(uint32_t range); double uniform();
// ---------------------------------------------------------------------------

struct MtRng {
  std::mt19937 g;
  uint32_t n;
  std::uniform_int_distribution<uint32_t> dn, d6;
  std::uniform_real_distribution<double> du;
  MtRng(uint64_t seed, uint32_t n_)
      : g(static_cast<uint32_t>(seed)), n(n_), dn(0, n_ - 1), d6(0, 5),
        du(0.0, 1.0) {}
  uint32_t bounded(uint32_t range) {
    if (range == n) return dn(g);
    if (range == 6) return d6(g);
    return std::uniform_int_distribution<uint32_t>(0, range - 1)(g);
  }
  double uniform() { return du(g); }
};

struct XoRng {
  uint64_t s[4];
  static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
  static uint64_t splitmix(uint64_t& x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
  explicit XoRng(uint64_t seed) {
    uint64_t x = seed;
    for (auto& v : s) v = splitmix(x);
  }
  uint64_t next64() {
    uint64_t result = rotl(s[0] + s[3], 23) + s[0];
    uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return result;
  }
  uint32_t bounded(uint32_t range) {
    uint64_t x = next64();
    __uint128_t m = static_cast<__uint128_t>(x) * range;
    uint64_t l = static_cast<uint64_t>(m);
    if (l < range) {
      uint64_t t = (0 - static_cast<uint64_t>(range)) % range;
      while (l < t) {
        x = next64();
        m = static_cast<__uint128_t>(x) * range;
        l = static_cast<uint64_t>(m);
      }
    }
    return static_cast<uint32_t>(m >> 64);
  }
  double uniform() { return static_cast<double>(next64() >> 11) * 0x1.0p-53; }
};

struct PhRng {
  uint32_t key[2];
  uint32_t ctr[2];
  uint32_t buf[4];
  int idx = 4;
  static uint32_t mulhilo(uint32_t a, uint32_t b, uint32_t& hi) {
    uint64_t p = static_cast<uint64_t>(a) * b;
    hi = static_cast<uint32_t>(p >> 32);
    return static_cast<uint32_t>(p);
  }
  static uint64_t splitmix(uint64_t& x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
  explicit PhRng(uint64_t seed) {
    uint64_t x = seed;
    uint64_t k0 = splitmix(x), k1 = splitmix(x);
    key[0] = static_cast<uint32_t>(k0);
    key[1] = static_cast<uint32_t>(k1);
    ctr[0] = 0;
    ctr[1] = 0;
  }
  void generate() {
    constexpr uint32_t M0 = 0xD2511F53u, M1 = 0xCD9E8D57u;
    constexpr uint32_t W0 = 0x9E3779B9u, W1 = 0xBB67AE85u;
    uint32_t c0 = ctr[0], c1 = ctr[1], c2 = 0, c3 = 0;
    for (int r = 0; r < 10; ++r) {
      uint32_t hi0, hi1;
      uint32_t lo0 = mulhilo(M0, c0, hi0);
      uint32_t lo1 = mulhilo(M1, c2, hi1);
      uint32_t n0 = hi1 ^ c1 ^ key[0];
      uint32_t n1 = lo1;
      uint32_t n2 = hi0 ^ c3 ^ key[1];
      uint32_t n3 = lo0;
      c0 = n0;
      c1 = n1;
      c2 = n2;
      c3 = n3;
      key[0] += W0;
      key[1] += W1;
    }
    buf[0] = c0;
    buf[1] = c1;
    buf[2] = c2;
    buf[3] = c3;
    idx = 0;
    if (++ctr[0] == 0) ++ctr[1];
  }
  uint32_t next32() {
    if (idx == 4) generate();
    return buf[idx++];
  }
  uint64_t next64() {
    uint64_t lo = next32();
    return (static_cast<uint64_t>(next32()) << 32) | lo;
  }
  uint32_t bounded(uint32_t range) {
    uint64_t m = static_cast<uint64_t>(next32()) * range;
    uint32_t l = static_cast<uint32_t>(m);
    if (l < range) {
      uint32_t t = (0u - range) % range;
      while (l < t) {
        m = static_cast<uint64_t>(next32()) * range;
        l = static_cast<uint32_t>(m);
      }
    }
    return static_cast<uint32_t>(m >> 32);
  }
  double uniform() { return static_cast<double>(next32()) * 0x1.0p-32; }
};

// ---------------------------------------------------------------------------
// Shared geometry
// ---------------------------------------------------------------------------

struct Geometry {
  int L, N;
  std::vector<int> nn;       // N * 6
  std::vector<uint8_t> bit;  // N * 6: bit index of this site in neighbor mask
  explicit Geometry(int L_) : L(L_), N(L_ * L_ * L_), nn(N * 6), bit(N * 6) {
    for (int i = 0; i < L; ++i)
      for (int j = 0; j < L; ++j)
        for (int k = 0; k < L; ++k) {
          const int s = k + L * (j + i * L);
          const int f = k + L * (j + ((i + L - 1) % L) * L);
          const int b = k + L * (j + ((i + 1) % L) * L);
          const int u = k + L * (((j + L - 1) % L) + i * L);
          const int d = k + L * (((j + 1) % L) + i * L);
          const int l = ((k + L - 1) % L) + L * (j + i * L);
          const int r = ((k + 1) % L) + L * (j + i * L);
          nn[s * 6 + 0] = f;
          nn[s * 6 + 1] = b;
          nn[s * 6 + 2] = u;
          nn[s * 6 + 3] = d;
          nn[s * 6 + 4] = l;
          nn[s * 6 + 5] = r;
        }
    for (int a = 0; a < N; ++a)
      for (int j = 0; j < 6; ++j) {
        const int n = nn[a * 6 + j];
        int k = 0;
        while (nn[n * 6 + k] != a) ++k;
        bit[a * 6 + j] = static_cast<uint8_t>(k);
      }
  }
};

struct ExpTable {
  static constexpr int kOffset = 512;
  static constexpr int kSize = 1024;
  double beta;
  std::array<double, kSize> table;
  explicit ExpTable(double beta_) : beta(beta_) {
    for (int i = 0; i < kSize; ++i)
      table[i] = std::exp(-beta * static_cast<double>(i - kOffset));
  }
  bool accept(int dE, double u) const {
    if (dE <= 0) return true;
    const int idx = dE + kOffset;
    if (idx < kSize) return u < table[idx];
    return u < std::exp(-beta * static_cast<double>(dE));
  }
};

// ---------------------------------------------------------------------------
// Byte + mask + local-energy table
// ---------------------------------------------------------------------------

struct ByteState {
  const Geometry& g;
  std::vector<uint8_t> type, mask;
  int le[3][64];

  struct Aff {
    int s;
    uint8_t bit_a, bit_b;  // 0 = none, else bit+1
    uint8_t nt, nm;
  };

  ByteState(const Geometry& g_, uint64_t seed) : g(g_), type(g_.N), mask(g_.N) {
    for (int t = 0; t < 3; ++t)
      for (int m = 0; m < 64; ++m) {
        if (t == 0) {
          le[t][m] = 0;
        } else {
          const int conn = (t == 1) ? 3 : 5;
          const int d = __builtin_popcount(static_cast<unsigned>(m));
          const int v = d - conn;
          le[t][m] = v * v;
        }
      }
    std::vector<int> sites(g.N);
    std::iota(sites.begin(), sites.end(), 0);
    std::mt19937 gen(static_cast<uint32_t>(seed));
    std::shuffle(sites.begin(), sites.end(), gen);
    const int n1 = static_cast<int>(std::llround(0.75 * 0.40 * g.N));
    const int n2 = static_cast<int>(std::llround(0.75 * 0.60 * g.N));
    for (int i = 0; i < n1; ++i) type[sites[i]] = 1;
    for (int i = 0; i < n2; ++i) type[sites[n1 + i]] = 2;
    for (int s = 0; s < g.N; ++s) {
      uint8_t m = 0;
      for (int j = 0; j < 6; ++j)
        if (type[g.nn[s * 6 + j]] != 0) m |= static_cast<uint8_t>(1u << j);
      mask[s] = m;
    }
  }

  int full_energy() const {
    int e = 0;
    for (int s = 0; s < g.N; ++s) e += le[type[s]][mask[s]];
    return e;
  }

  // Returns dE. If apply and dE is accepted by the caller, call apply().
  // Here "apply" means commit the already-computed affected-site state.
  int delta(int a, int b, Aff* aff_out, int& na_out) const {
    Aff aff[14];
    int na = 0;
    auto add = [&](int s, int which, int bit_index) {
      for (int i = 0; i < na; ++i)
        if (aff[i].s == s) {
          if (which == 0) aff[i].bit_a = static_cast<uint8_t>(bit_index + 1);
          if (which == 1) aff[i].bit_b = static_cast<uint8_t>(bit_index + 1);
          return;
        }
      aff[na].s = s;
      aff[na].bit_a = (which == 0) ? static_cast<uint8_t>(bit_index + 1) : 0;
      aff[na].bit_b = (which == 1) ? static_cast<uint8_t>(bit_index + 1) : 0;
      ++na;
    };
    add(a, 2, 0);
    add(b, 2, 0);
    for (int j = 0; j < 6; ++j) add(g.nn[a * 6 + j], 0, g.bit[a * 6 + j]);
    for (int j = 0; j < 6; ++j) add(g.nn[b * 6 + j], 1, g.bit[b * 6 + j]);

    const uint8_t ta = type[a], tb = type[b];
    const uint8_t oa = (ta != 0), ob = (tb != 0);
    const uint8_t na_occ = (tb != 0), nb_occ = (ta != 0);
    int old_e = 0, new_e = 0;
    for (int i = 0; i < na; ++i) {
      const int s = aff[i].s;
      old_e += le[type[s]][mask[s]];
      uint8_t nm = mask[s];
      if (aff[i].bit_a) {
        const uint8_t bit = static_cast<uint8_t>(aff[i].bit_a - 1);
        nm = static_cast<uint8_t>((nm & ~(1u << bit)) | (na_occ << bit));
      }
      if (aff[i].bit_b) {
        const uint8_t bit = static_cast<uint8_t>(aff[i].bit_b - 1);
        nm = static_cast<uint8_t>((nm & ~(1u << bit)) | (nb_occ << bit));
      }
      const uint8_t nt = (s == a) ? tb : (s == b) ? ta : type[s];
      aff[i].nt = nt;
      aff[i].nm = nm;
      new_e += le[nt][nm];
    }
    for (int i = 0; i < na; ++i) aff_out[i] = aff[i];
    na_out = na;
    return new_e - old_e;
  }

  void commit(const Aff* aff, int na) {
    for (int i = 0; i < na; ++i) {
      type[aff[i].s] = aff[i].nt;
      mask[aff[i].s] = aff[i].nm;
    }
  }
};

int verify_byte(ByteState& st, uint64_t seed, int checks) {
  std::mt19937 gen(static_cast<uint32_t>(seed));
  std::uniform_int_distribution<int> site(0, st.g.N - 1);
  ByteState::Aff aff[14];
  int na = 0;
  for (int c = 0; c < checks; ++c) {
    int a = site(gen), b = site(gen);
    const int e0 = st.full_energy();
    const int dE = st.delta(a, b, aff, na);
    st.commit(aff, na);
    const int e1 = st.full_energy();
    if (e1 - e0 != dE) {
      std::printf("BYTE MISMATCH a=%d b=%d dE=%d full=%d\n", a, b, dE, e1 - e0);
      return 1;
    }
    const int dE2 = st.delta(a, b, aff, na);
    st.commit(aff, na);
    const int e2 = st.full_energy();
    if (e2 != e0) {
      std::printf("BYTE RESTORE MISMATCH a=%d b=%d\n", a, b);
      return 1;
    }
    (void)dE2;
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Packed 2-bit baseline (current design)
// ---------------------------------------------------------------------------

struct PackedState {
  const Geometry& g;
  std::vector<uint8_t> lat;
  explicit PackedState(const Geometry& g_, uint64_t seed) : g(g_), lat((g_.N + 3) / 4, 0) {
    std::vector<int> sites(g.N);
    std::iota(sites.begin(), sites.end(), 0);
    std::mt19937 gen(static_cast<uint32_t>(seed));
    std::shuffle(sites.begin(), sites.end(), gen);
    const int n1 = static_cast<int>(std::llround(0.75 * 0.40 * g.N));
    const int n2 = static_cast<int>(std::llround(0.75 * 0.60 * g.N));
    for (int i = 0; i < n1; ++i) set(sites[i], 1);
    for (int i = 0; i < n2; ++i) set(sites[n1 + i], 2);
  }
  uint8_t get(int s) const {
    const int idx = s >> 2, off = (s & 3) * 2;
    return static_cast<uint8_t>((lat[idx] >> off) & 3u);
  }
  void set(int s, uint8_t v) {
    const int idx = s >> 2, off = (s & 3) * 2;
    lat[idx] = static_cast<uint8_t>((lat[idx] & ~(3u << off)) | ((v & 3u) << off));
  }
  int local(int s) const {
    const int t = get(s);
    if (t == 0) return 0;
    const int conn = (t == 1) ? 3 : 5;
    int e = 0;
    for (int j = 0; j < 6; ++j) e += (get(g.nn[s * 6 + j]) != 0);
    const int v = e - conn;
    return v * v;
  }
  int full_energy() const {
    int e = 0;
    for (int s = 0; s < g.N; ++s) e += local(s);
    return e;
  }
  void exchange(int a, int b) {
    const uint8_t ta = get(a), tb = get(b);
    set(a, tb);
    set(b, ta);
  }
};

int verify_packed(PackedState& st, uint64_t seed, int checks) {
  std::mt19937 gen(static_cast<uint32_t>(seed));
  std::uniform_int_distribution<int> site(0, st.g.N - 1);
  for (int c = 0; c < checks; ++c) {
    int a = site(gen), b = site(gen);
    const int e0 = st.full_energy();
    st.exchange(a, b);
    const int e1 = st.full_energy();
    st.exchange(a, b);
    if (st.full_energy() != e0) {
      std::printf("PACKED RESTORE MISMATCH a=%d b=%d dE=%d\n", a, b, e1 - e0);
      return 1;
    }
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Move loops
// ---------------------------------------------------------------------------

enum class Mix { LocalOnly, NonlocalOnly, Mixed };

template <class RNG>
inline void pick_sites(const Geometry& g, RNG& rng, Mix mix, double local_fraction,
                       int& a, int& b) {
  bool local = (mix == Mix::LocalOnly) ||
               (mix == Mix::Mixed && rng.uniform() < local_fraction);
  a = static_cast<int>(rng.bounded(static_cast<uint32_t>(g.N)));
  if (local) {
    b = g.nn[a * 6 + static_cast<int>(rng.bounded(6))];
  } else {
    b = (a + 1 + static_cast<int>(rng.bounded(static_cast<uint32_t>(g.N - 1)))) % g.N;
  }
}

template <class RNG>
int run_packed(PackedState& st, RNG& rng, double beta, int proposals, Mix mix,
               double local_fraction, int& accepted) {
  const Geometry& g = st.g;
  accepted = 0;
  for (int i = 0; i < proposals; ++i) {
    int a, b;
    pick_sites(g, rng, mix, local_fraction, a, b);
    if (st.get(a) == st.get(b)) continue;
    int aff[14];
    int na = 0;
    auto add = [&](int s) {
      for (int i = 0; i < na; ++i)
        if (aff[i] == s) return;
      aff[na++] = s;
    };
    add(a);
    add(b);
    for (int j = 0; j < 6; ++j) add(g.nn[a * 6 + j]);
    for (int j = 0; j < 6; ++j) add(g.nn[b * 6 + j]);
    int e0 = 0;
    for (int i = 0; i < na; ++i) e0 += st.local(aff[i]);
    st.exchange(a, b);
    int e1 = 0;
    for (int i = 0; i < na; ++i) e1 += st.local(aff[i]);
    const int dE = e1 - e0;
    bool acc = (dE <= 0) || (rng.uniform() < std::exp(-beta * dE));
    if (acc) {
      ++accepted;
    } else {
      st.exchange(a, b);
    }
  }
  return st.full_energy();
}

template <class RNG>
int run_byte(ByteState& st, RNG& rng, double beta, int proposals, Mix mix,
             double local_fraction, bool use_table, int& accepted) {
  const Geometry& g = st.g;
  ExpTable table(beta);
  ByteState::Aff aff[14];
  accepted = 0;
  for (int i = 0; i < proposals; ++i) {
    int a, b;
    pick_sites(g, rng, mix, local_fraction, a, b);
    if (st.type[a] == st.type[b]) continue;
    int na = 0;
    const int dE = st.delta(a, b, aff, na);
    bool acc = (dE <= 0);
    if (!acc) {
      const double u = rng.uniform();
      acc = use_table ? table.accept(dE, u) : (u < std::exp(-beta * dE));
    }
    if (acc) {
      st.commit(aff, na);
      ++accepted;
    }
  }
  return st.full_energy();
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

struct Case {
  const char* name;
  int kind;  // 0 packed, 1 byte
  int rng;   // 0 mt, 1 xoshiro, 2 philox
  int mix;
  bool table;
};

template <class Fn>
double time_case(Fn&& fn, int reps) {
  double best = 1e9;
  for (int r = 0; r < reps; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    fn();
    const auto t1 = std::chrono::steady_clock::now();
    const double s = std::chrono::duration<double>(t1 - t0).count();
    if (s < best) best = s;
  }
  return best;
}

int main(int argc, char** argv) {
  const int L = (argc > 1) ? std::atoi(argv[1]) : 20;
  const int proposals = (argc > 2) ? std::atoi(argv[2]) : 2000000;
  const double beta = (argc > 3) ? std::atof(argv[3]) : 3.5;
  const int reps = (argc > 4) ? std::atoi(argv[4]) : 3;
  const uint64_t seed = 12345;

  Geometry g(L);
  std::printf("L=%d N=%d proposals=%d beta=%.2f reps=%d\n", L, g.N, proposals, beta, reps);

  ByteState vb(g, seed);
  PackedState vp(g, seed);
  if (verify_byte(vb, seed + 1, 20000) != 0) return 1;
  if (verify_packed(vp, seed + 2, 2000) != 0) return 1;
  std::printf("correctness: PASS\n");

  const Case cases[] = {
      {"packed/mt/exp/mixed", 0, 0, 2, false},
      {"byte/mt/exp/mixed", 1, 0, 2, false},
      {"byte/mt/table/mixed", 1, 0, 2, true},
      {"byte/xo/table/mixed", 1, 1, 2, true},
      {"byte/ph/table/mixed", 1, 2, 2, true},
      {"byte/ph/table/local", 1, 2, 0, true},
      {"byte/ph/table/nonlocal", 1, 2, 1, true},
  };

  for (const auto& c : cases) {
    int last_accepted = 0;
    auto run = [&]() {
      int accepted = 0;
      int sink = 0;
      if (c.kind == 0) {
        PackedState st(g, seed);
        if (c.rng == 0) {
          MtRng rng(seed, g.N);
          sink = run_packed(st, rng, beta, proposals, static_cast<Mix>(c.mix), 0.5, accepted);
        }
      } else {
        ByteState st(g, seed);
        if (c.rng == 0) {
          MtRng rng(seed, g.N);
          sink = run_byte(st, rng, beta, proposals, static_cast<Mix>(c.mix), 0.5, c.table, accepted);
        } else if (c.rng == 1) {
          XoRng rng(seed);
          sink = run_byte(st, rng, beta, proposals, static_cast<Mix>(c.mix), 0.5, c.table, accepted);
        } else {
          PhRng rng(seed);
          sink = run_byte(st, rng, beta, proposals, static_cast<Mix>(c.mix), 0.5, c.table, accepted);
        }
      }
      volatile int keep = sink + accepted;
      (void)keep;
      last_accepted = accepted;
    };
    const double s = time_case(run, reps);
    std::printf("%-28s %8.2f ns/proposal  %8.2f Mprop/s  acc %6.2f%%\n", c.name,
                s * 1e9 / proposals, proposals / s / 1e6,
                100.0 * last_accepted / proposals);
  }
  return 0;
}
