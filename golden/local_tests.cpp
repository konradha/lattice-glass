// Exhaustive checks for the local-move golden reference in local_model.hpp.
//
// Deterministic, no file I/O. Prints one line per contract and exits non-zero
// on the first failed check.

#include "local_model.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

using namespace lgl;

namespace {

void check_fail(const char* expr, const char* file, int line) {
  std::fprintf(stderr, "FAIL %s:%d: %s\n", file, line, expr);
  std::exit(1);
}

#define CHECK(cond) ((cond) ? static_cast<void>(0) : check_fail(#cond, __FILE__, __LINE__))

struct Rng {
  uint64_t x;
  explicit Rng(uint64_t seed) : x(seed) {}
  uint64_t next() { return splitmix64(x); }
  uint32_t below(uint32_t n) { return static_cast<uint32_t>(next() % n); }
  double uniform() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }
};

bool has(const int* a, int n, int v) {
  for (int i = 0; i < n; ++i) {
    if (a[i] == v) return true;
  }
  return false;
}

// Color of the positive edge anchored at site s, for every axis:
//   color = (x + 3y + 5z) mod 21.
// The plan states this formula for +x edges; this reference applies the same
// anchor formula to the +y and +z sweeps and validates all three exhaustively.
int edge_color(int s) {
  return (coord(s, 0) + 3 * coord(s, 1) + 5 * coord(s, 2)) % L;
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

void test_geometry() {
  for (int s = 0; s < N; ++s) {
    int nbr[6];
    for (int d = 0; d < 6; ++d) {
      const int q = neighbor(s, d);
      CHECK(q >= 0 && q < N);
      CHECK(q != s);
      CHECK(neighbor(q, d ^ 1) == s);
      CHECK(manhattan(s, q) == 1);
      nbr[d] = q;
    }
    for (int d = 0; d < 6; ++d) {
      for (int e = d + 1; e < 6; ++e) CHECK(nbr[d] != nbr[e]);
    }
  }
  for (int x = 0; x < L; ++x) {
    for (int y = 0; y < L; ++y) {
      for (int z = 0; z < L; ++z) {
        const int s = site(x, y, z);
        CHECK(s >= 0 && s < N);
        CHECK(coord(s, 0) == x);
        CHECK(coord(s, 1) == y);
        CHECK(coord(s, 2) == z);
      }
    }
  }
  CHECK(6LL * N == 55566LL);
  std::printf("[ok] geometry: %d sites, %lld directed bonds, involutive torus neighbors\n", N,
              6LL * N);
}

// ---------------------------------------------------------------------------
// All 55,566 directed bonds: the affected set is exactly the 12-site closure
// ---------------------------------------------------------------------------

void test_affected_sets() {
  long long bonds = 0;
  for (int s = 0; s < N; ++s) {
    for (int d = 0; d < 6; ++d) {
      const int a = s;
      const int b = neighbor(s, d);
      int aff[MAX_AFFECTED];
      const int n = affected_set(a, b, aff);
      CHECK(n == 12);
      for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) CHECK(aff[i] != aff[j]);
      }
      CHECK(has(aff, n, a));
      CHECK(has(aff, n, b));

      // Independent construction of {a,b} U N(a) U N(b).
      int expect[MAX_AFFECTED];
      int ne = 0;
      auto add = [&](int q) {
        if (!has(expect, ne, q)) expect[ne++] = q;
      };
      add(a);
      add(b);
      for (int k = 0; k < 6; ++k) add(neighbor(a, k));
      for (int k = 0; k < 6; ++k) add(neighbor(b, k));
      CHECK(ne == 12);
      for (int i = 0; i < n; ++i) CHECK(has(expect, ne, aff[i]));
      for (int i = 0; i < ne; ++i) CHECK(has(aff, n, expect[i]));

      for (int i = 0; i < n; ++i) {
        const int q = aff[i];
        CHECK(q == a || q == b || manhattan(q, a) == 1 || manhattan(q, b) == 1);
      }
      ++bonds;
    }
  }
  CHECK(bonds == 55566LL);
  std::printf("[ok] affected set: %lld directed bonds, |A(a,b)| = 12\n", bonds);
}

// ---------------------------------------------------------------------------
// delta against a full-energy recomputation on every directed bond
// ---------------------------------------------------------------------------

void test_delta_vs_full() {
  State S;
  S.init(0x5eed1234ULL);
  CHECK(S.invariant());
  const State base = S;
  const int E0 = S.full_energy();
  long long nonnull = 0;
  long long nulls = 0;

  for (int s = 0; s < N; ++s) {
    for (int d = 0; d < 6; ++d) {
      const int a = s;
      const int b = neighbor(s, d);
      int aff[MAX_AFFECTED];
      const int n = affected_set(a, b, aff);

      if (S.type(a) == S.type(b)) {
        ++nulls;
        uint8_t old[MAX_AFFECTED];
        for (int k = 0; k < n; ++k) old[k] = S.s[aff[k]];
        CHECK(delta(S, a, b) == 0);
        commit(S, a, b);
        for (int k = 0; k < n; ++k) CHECK(S.s[aff[k]] == old[k]);
        continue;
      }

      ++nonnull;
      uint8_t old[MAX_AFFECTED];
      for (int k = 0; k < n; ++k) old[k] = S.s[aff[k]];
      const int dE = delta(S, a, b);
      CHECK(dE == evaluate(S, a, b, 0.0, 2.8).dE);
      commit(S, a, b);
      CHECK(S.full_energy() - E0 == dE);
      commit(S, b, a); // the swap is an involution: exact inverse
      for (int k = 0; k < n; ++k) CHECK(S.s[aff[k]] == old[k]);
    }
  }
  CHECK(S.s == base.s);
  CHECK(nonnull + nulls == 55566LL);
  std::printf("[ok] delta vs full energy: %lld non-null bonds (%lld null restored)\n", nonnull,
              nulls);
}

// ---------------------------------------------------------------------------
// Fixed composition
// ---------------------------------------------------------------------------

void test_fixed_composition() {
  const uint64_t seeds[4] = {0ULL, 1ULL, 0xdeadbeefULL, 0x9e3779b97f4a7c15ULL};
  for (int i = 0; i < 4; ++i) {
    State S;
    S.init(seeds[i]);
    CHECK(S.invariant());
    const std::array<int, 3> c = S.composition();
    CHECK(c[0] == N0);
    CHECK(c[1] == N1);
    CHECK(c[2] == N2);
  }
  std::printf("[ok] fixed composition: N0=%d N1=%d N2=%d\n", N0, N1, N2);
}

// ---------------------------------------------------------------------------
// Small forced induced-matching example
// ---------------------------------------------------------------------------

void test_induced_matching_small() {
  const int anchor_x[4] = {0, 5, 10, 15};
  int a[4];
  int b[4];
  int aff[4][MAX_AFFECTED];
  int na[4];

  State S;
  S.s.fill(0);
  for (int i = 0; i < 4; ++i) {
    a[i] = site(anchor_x[i], 0, 0);
    b[i] = neighbor(a[i], 0);
    S.set_site(a[i], 1, 0);
    S.set_site(b[i], 2, 0);
  }
  S.recompute_masks();
  CHECK(S.masks_ok());

  for (int i = 0; i < 4; ++i) {
    na[i] = affected_set(a[i], b[i], aff[i]);
    CHECK(na[i] == 12);
  }
  for (int i = 0; i < 4; ++i) {
    for (int j = i + 1; j < 4; ++j) {
      for (int u = 0; u < na[i]; ++u) {
        for (int v = 0; v < na[j]; ++v) CHECK(aff[i][u] != aff[j][v]);
      }
      CHECK(manhattan(a[i], a[j]) >= 2);
      CHECK(manhattan(a[i], b[j]) >= 2);
      CHECK(manhattan(b[i], a[j]) >= 2);
      CHECK(manhattan(b[i], b[j]) >= 2);
    }
  }

  int dE[4];
  for (int i = 0; i < 4; ++i) dE[i] = delta(S, a[i], b[i]);
  const int E0 = S.full_energy();

  State seq = S;
  for (int i = 0; i < 4; ++i) commit(seq, a[i], b[i]);
  State rev = S;
  for (int i = 3; i >= 0; --i) commit(rev, a[i], b[i]);
  State par = S;
  for (int i = 0; i < 4; ++i) {
    State tmp = S;
    commit(tmp, a[i], b[i]);
    for (int k = 0; k < na[i]; ++k) par.s[aff[i][k]] = tmp.s[aff[i][k]];
  }

  CHECK(seq.s == rev.s);
  CHECK(seq.s == par.s);
  int sum = 0;
  for (int i = 0; i < 4; ++i) sum += dE[i];
  CHECK(seq.full_energy() - E0 == sum);
  CHECK(seq.masks_ok());
  std::printf("[ok] forced induced matching: 4 moves, orders and parallel apply agree (dE=%d)\n",
              sum);
}

// ---------------------------------------------------------------------------
// Full color-class induced matching: all 441 moves of color 0 (+x) commute
// ---------------------------------------------------------------------------

void test_induced_matching_color_class() {
  std::array<std::vector<int>, 21> cls;
  for (int s = 0; s < N; ++s) cls[edge_color(s)].push_back(s);

  const std::vector<int>& anchors = cls[0];
  const int m = static_cast<int>(anchors.size());
  CHECK(m == L * L);

  std::vector<int> a(m);
  std::vector<int> b(m);
  std::vector<std::array<int, MAX_AFFECTED>> aff(m);
  std::vector<int> na(m);
  std::vector<int> ends;
  ends.reserve(static_cast<std::size_t>(2 * m));
  for (int i = 0; i < m; ++i) {
    a[i] = anchors[i];
    b[i] = neighbor(a[i], 0);
    na[i] = affected_set(a[i], b[i], aff[i].data());
    CHECK(na[i] == 12);
    ends.push_back(a[i]);
    ends.push_back(b[i]);
  }
  std::sort(ends.begin(), ends.end());
  for (std::size_t i = 1; i < ends.size(); ++i) CHECK(ends[i] != ends[i - 1]);
  for (int i = 0; i < m; ++i) {
    for (int j = i + 1; j < m; ++j) {
      for (int u = 0; u < na[i]; ++u) {
        for (int v = 0; v < na[j]; ++v) CHECK(aff[i][u] != aff[j][v]);
      }
    }
  }

  State S;
  S.s.fill(0);
  for (int i = 0; i < m; ++i) {
    CHECK(S.type(a[i]) == 0);
    CHECK(S.type(b[i]) == 0);
    S.set_site(a[i], 1, 0);
    S.set_site(b[i], 2, 0);
  }
  S.recompute_masks();
  CHECK(S.masks_ok());

  std::vector<int> dE(m);
  for (int i = 0; i < m; ++i) dE[i] = delta(S, a[i], b[i]);
  const int E0 = S.full_energy();

  State seq = S;
  for (int i = 0; i < m; ++i) commit(seq, a[i], b[i]);

  State rev = S;
  for (int i = m - 1; i >= 0; --i) commit(rev, a[i], b[i]);

  State shuf = S;
  {
    std::vector<int> order(m);
    for (int i = 0; i < m; ++i) order[i] = i;
    Rng rng(0xabcdef0123456789ULL);
    for (int i = m - 1; i > 0; --i) {
      const int j = static_cast<int>(rng.below(static_cast<uint32_t>(i + 1)));
      std::swap(order[static_cast<std::size_t>(i)], order[static_cast<std::size_t>(j)]);
    }
    for (int i = 0; i < m; ++i) commit(shuf, a[order[i]], b[order[i]]);
  }

  State par = S;
  for (int i = 0; i < m; ++i) {
    State tmp = S;
    commit(tmp, a[i], b[i]);
    for (int k = 0; k < na[i]; ++k) par.s[aff[i][k]] = tmp.s[aff[i][k]];
  }

  CHECK(seq.s == rev.s);
  CHECK(seq.s == shuf.s);
  CHECK(seq.s == par.s);
  int sum = 0;
  for (int i = 0; i < m; ++i) sum += dE[i];
  CHECK(seq.full_energy() - E0 == sum);
  CHECK(seq.masks_ok());
  std::printf("[ok] full induced matching: color 0 (+x) has %d disjoint moves, all orders agree "
              "(dE=%d)\n",
              m, sum);
}

// ---------------------------------------------------------------------------
// Exhaustive 21-color schedule validation
//
// Two same-axis bonds scheduled in the same color may run concurrently only if
// their 12-site affected sets are disjoint. For anchors differing by delta this
// is equivalent to delta not being a difference of two affected offsets. The
// check below enumerates every same-color pair of every axis; color and the
// offset difference set are translation invariant modulo L, and L equals the
// color modulus, so this covers all 27,783 bonds.
// ---------------------------------------------------------------------------

void test_color_schedule() {
  std::array<std::array<int, 3>, N> xyz{};
  for (int s = 0; s < N; ++s) {
    xyz[s][0] = coord(s, 0);
    xyz[s][1] = coord(s, 1);
    xyz[s][2] = coord(s, 2);
  }

  long long total_pairs = 0;
  for (int axis = 0; axis < 3; ++axis) {
    int off_sites[MAX_AFFECTED];
    const int n_off = affected_set(0, neighbor(0, axis), off_sites);
    CHECK(n_off == 12);

    // The plan's claim: the 12 affected offsets have distinct colors.
    std::array<bool, L> residue{};
    for (int k = 0; k < n_off; ++k) {
      const int r = edge_color(off_sites[k]);
      CHECK(!residue[static_cast<std::size_t>(r)]);
      residue[static_cast<std::size_t>(r)] = true;
    }

    // Exact conflict table over delta in Z_L^3.
    std::array<bool, N> in_diff{};
    for (int u = 0; u < n_off; ++u) {
      const int ux = xyz[off_sites[u]][0];
      const int uy = xyz[off_sites[u]][1];
      const int uz = xyz[off_sites[u]][2];
      for (int v = 0; v < n_off; ++v) {
        const int dx = (ux - xyz[off_sites[v]][0] + L) % L;
        const int dy = (uy - xyz[off_sites[v]][1] + L) % L;
        const int dz = (uz - xyz[off_sites[v]][2] + L) % L;
        in_diff[static_cast<std::size_t>(site(dx, dy, dz))] = true;
      }
    }

    std::array<std::vector<int>, 21> cls;
    for (int s = 0; s < N; ++s) cls[edge_color(s)].push_back(s);

    long long axis_pairs = 0;
    for (int c = 0; c < 21; ++c) {
      CHECK(static_cast<int>(cls[c].size()) == L * L);
      for (std::size_t i = 0; i < cls[c].size(); ++i) {
        for (std::size_t j = i + 1; j < cls[c].size(); ++j) {
          const int u = cls[c][i];
          const int v = cls[c][j];
          const int dx = (xyz[u][0] - xyz[v][0] + L) % L;
          const int dy = (xyz[u][1] - xyz[v][1] + L) % L;
          const int dz = (xyz[u][2] - xyz[v][2] + L) % L;
          CHECK(!in_diff[static_cast<std::size_t>(site(dx, dy, dz))]);
          ++axis_pairs;
        }
      }
    }
    CHECK(axis_pairs == 21LL * (L * L) * (L * L - 1) / 2);
    total_pairs += axis_pairs;
    std::printf("[ok] color schedule axis %d: 21 colors x %d bonds, %lld same-color pairs "
                "disjoint\n",
                axis, L * L, axis_pairs);
  }
  CHECK(total_pairs == 3LL * 21 * (L * L) * (L * L - 1) / 2);
  std::printf("[ok] 21-color schedule exhaustive: %lld same-color same-axis pairs\n", total_pairs);
}

// ---------------------------------------------------------------------------
// 100,000 random proposals against a full-energy recomputation
// ---------------------------------------------------------------------------

void test_random_100k() {
  const double beta = 2.8;
  const int proposals = 100000;

  State S;
  S.init(0x9e3779b97f4a7c15ULL);
  CHECK(S.invariant());
  Rng rng(0x0123456789abcdefULL);

  long long nulls = 0;
  long long accepts = 0;
  int E = S.full_energy();

  for (int i = 0; i < proposals; ++i) {
    const int a = static_cast<int>(rng.below(static_cast<uint32_t>(N)));
    const int d = static_cast<int>(rng.below(6));
    const int b = neighbor(a, d);
    const double u = rng.uniform();

    const bool same = (S.type(a) == S.type(b));
    const Eval ev = evaluate(S, a, b, u, beta);
    CHECK(ev.null_move == same);

    if (same) {
      ++nulls;
      CHECK(!ev.accept);
      CHECK(ev.dE == 0);
    } else {
      const int dE = delta(S, a, b);
      CHECK(ev.dE == dE);
      CHECK(ev.accept == (u < acceptance_probability(dE, beta)));
      if (ev.accept) {
        CHECK(apply(S, a, b, ev));
        ++accepts;
      }
    }

    const int Enew = S.full_energy();
    CHECK(Enew == E + (ev.accept ? ev.dE : 0));
    E = Enew;

    if (i % 1000 == 0) CHECK(S.invariant());
  }

  CHECK(S.invariant());
  CHECK(S.full_energy() == E);
  const std::array<int, 3> c = S.composition();
  CHECK(c[0] == N0);
  CHECK(c[1] == N1);
  CHECK(c[2] == N2);
  std::printf("[ok] %d random proposals: %lld accepted, %lld null, every step checked against a "
              "full-energy recomputation\n",
              proposals, accepts, nulls);
}

} // namespace

int main() {
  test_geometry();
  test_affected_sets();
  test_delta_vs_full();
  test_fixed_composition();
  test_induced_matching_small();
  test_induced_matching_color_class();
  test_color_schedule();
  test_random_100k();
  std::printf("all local-model checks passed\n");
  return 0;
}
