// Per-sweep speed benchmark: teaching-harness swap vs an optimized production
// swap (xoshiro256++ RNG, integer-dE exp lookup table, O(1) remove/insert
// deltas, no dedup rescan). Reports us/sweep and ns/attempt at a given L and
// beta, and the implied wall-time for a 2e10-sweep dataset on 1 core.
//
// The optimized kernel is the SAME Markov chain as blind_occupancy_sweep
// (uniform occupied source, uniform vacant destination, Metropolis on the exact
// region energy) -- only the implementation is faster. Composition is asserted.

#include "../informed_swap.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace fp = lattice_glass::fp;
namespace informed = lattice_glass::informed;
namespace cluster = lattice_glass::cluster;

// xoshiro256++ : fast, high-quality.
struct Xoshiro {
  uint64_t s[4];
  explicit Xoshiro(uint64_t seed) {
    for (int i = 0; i < 4; ++i) {
      seed += 0x9E3779B97F4A7C15ULL;
      uint64_t z = seed;
      z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
      z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
      s[i] = z ^ (z >> 31);
    }
  }
  static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
  uint64_t next() {
    const uint64_t result = rotl(s[0] + s[3], 23) + s[0];
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = rotl(s[3], 45);
    return result;
  }
  // uniform in [0,n) via Lemire's multiplication trick (no modulo).
  uint32_t bounded(uint32_t n) {
    uint64_t m = static_cast<uint64_t>(static_cast<uint32_t>(next())) * n;
    return static_cast<uint32_t>(m >> 32);
  }
  double uni() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
};

// Optimized blind occupancy swap sweep. `etab` is exp(-beta*d) for integer
// d in [-OFF, OFF]; out-of-range falls back to std::exp.
static constexpr int kOff = 160;
struct SweepCount { long long attempts = 0, accepted = 0; };

static SweepCount fast_swap_sweep(std::vector<uint8_t> &lat, std::vector<int> &m,
                                  informed::SiteLists &lists, int attempts,
                                  double beta, const double *etab,
                                  Xoshiro &rng, const int *nn) {
  SweepCount c;
  const int np = static_cast<int>(lists.occ.size());
  const int nv = static_cast<int>(lists.vac.size());
  for (int a = 0; a < attempts; ++a) {
    const int i = lists.occ[rng.bounded(np)];
    const int j = lists.vac[rng.bounded(nv)];
    const int d_rem = informed::remove_particle_delta(lat, m, i, nn);
    const uint8_t label = lat[i];
    const int *ni = nn + fp::kNumNeighbors * i;
    lat[i] = cluster::kEmpty;
    for (int k = 0; k < fp::kNumNeighbors; ++k) --m[ni[k]];
    const int d = d_rem + informed::insert_particle_delta(lat, m, j, label, nn);
    ++c.attempts;
    bool acc;
    if (d <= 0) acc = true;
    else if (d <= kOff) acc = rng.uni() < etab[d + kOff];
    else acc = rng.uni() < std::exp(-beta * d);
    if (acc) {
      const int *nj = nn + fp::kNumNeighbors * j;
      lat[j] = label;
      for (int k = 0; k < fp::kNumNeighbors; ++k) ++m[nj[k]];
      informed::commit_move(lists, i, j);
      ++c.accepted;
    } else {
      lat[i] = label;
      for (int k = 0; k < fp::kNumNeighbors; ++k) ++m[ni[k]];
    }
  }
  return c;
}

static std::array<int, 3> counts(const std::vector<uint8_t> &l) {
  std::array<int, 3> c{};
  for (uint8_t v : l) c[v]++;
  return c;
}

static void run(int L, double beta) {
  const int n = L * L * L;
  const int np = static_cast<int>(0.75 * n);
  const int n1 = static_cast<int>(0.30 * n);
  const std::vector<int> nn = fp::cubic_neighbors(L);
  std::mt19937 seed_gen(12345);
  std::vector<uint8_t> lat0 = fp::random_lattice(n1, np - n1, n, seed_gen);

  std::vector<double> etab(2 * kOff + 1);
  for (int d = -kOff; d <= kOff; ++d) etab[d + kOff] = std::exp(-beta * d);

  const int sweeps = (L <= 16) ? 400 : 150;
  const int warm = 40;

  // --- optimized kernel ---
  {
    std::vector<uint8_t> lat = lat0;
    std::vector<int> m = informed::build_neighbor_counts(lat, nn.data());
    informed::SiteLists lists = informed::build_site_lists(lat);
    Xoshiro rng(999);
    const auto c0 = counts(lat);
    for (int s = 0; s < warm; ++s)
      fast_swap_sweep(lat, m, lists, np, beta, etab.data(), rng, nn.data());
    auto t0 = std::chrono::steady_clock::now();
    SweepCount acc;
    for (int s = 0; s < sweeps; ++s) {
      SweepCount c = fast_swap_sweep(lat, m, lists, np, beta, etab.data(), rng, nn.data());
      acc.attempts += c.attempts; acc.accepted += c.accepted;
    }
    auto t1 = std::chrono::steady_clock::now();
    assert(counts(lat) == c0);
    double secs = std::chrono::duration<double>(t1 - t0).count();
    double us_sweep = secs / sweeps * 1e6;
    double ns_att = secs / acc.attempts * 1e9;
    double week_days = 2e10 * (us_sweep * 1e-6) / 86400.0;
    std::printf("L=%d beta=%.2f OPT : %8.1f us/sweep  %6.2f ns/attempt  acc=%.1e  "
                "2e10 sweeps = %6.1f core-days (%.1f wk)\n",
                L, beta, us_sweep, ns_att,
                (double)acc.accepted / acc.attempts, week_days, week_days / 7);
  }
  // --- teaching harness (move_delta dedup + std::exp + mt19937) ---
  {
    std::vector<uint8_t> lat = lat0;
    std::vector<int> m = informed::build_neighbor_counts(lat, nn.data());
    informed::SiteLists lists = informed::build_site_lists(lat);
    std::mt19937 gen(999);
    for (int s = 0; s < warm; ++s)
      informed::blind_occupancy_sweep(lat, (long double)beta, np, lists, m, gen, nn.data());
    auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < sweeps; ++s)
      informed::blind_occupancy_sweep(lat, (long double)beta, np, lists, m, gen, nn.data());
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    double us_sweep = secs / sweeps * 1e6;
    std::printf("L=%d beta=%.2f BASE: %8.1f us/sweep  %6.2f ns/attempt  "
                "(harness)\n", L, beta, us_sweep, us_sweep * 1000.0 / np);
  }
}

int main() {
  for (int L : {16, 20}) {
    run(L, 3.0);
  }
  return 0;
}
