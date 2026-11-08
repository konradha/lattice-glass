#pragma once
// ---------------------------------------------------------------------------
// Local-move golden reference for the L=21 lattice glass.
//
// N = 21^3 = 9261 sites. This is the reference the local/colored kernels are
// checked against; it is self-contained and has no dependency on the nonlocal
// windowed model in lattice_model.hpp.
//
// State byte (same layout as cpu/lattice.hpp), one byte per site:
//   bits 1:0   type: 0 = vacancy, 1 and 2 = occupied species
//   bits 7:2   six-bit neighbor occupancy mask; bit d is set iff the site at
//              neighbor(s, d) has type != 0
//
// Site indexing matches cpu/lattice.hpp: s = z + L*y + L^2*x, and direction d
// is 0=+x, 1=-x, 2=+y, 3=-y, 4=+z, 5=-z, with d^1 the opposite direction.
//
// Local energy of one site: 0 for a vacancy, otherwise
//   (popcount(mask) - conn)^2,  conn = 3 for type 1, conn = 5 for type 2.
//
// Move: a proposal is an ordered pair (a, b = neighbor(a, d)) plus a uniform
// u in [0, 1). If type(a) == type(b) the proposal is null (this includes
// vacancy-vacancy). Otherwise the two types are exchanged and the masks of the
// affected set
//   A(a,b) = {a, b} U N(a) U N(b)
// are updated. For a bond (adjacent a, b) |A(a,b)| = 12 exactly: the two
// endpoints plus their ten other neighbors.
//
// Metropolis convention (declared): dE is an exact integer. Accept iff
//   dE <= 0  or  u < exp(-beta * dE)
// with exp evaluated by std::exp in double. This is full double precision,
// mathematically u < min(1, exp(-beta*dE)) since u < 1. No integer fixed-point
// table and no other rounding is used.
//
// Fixed composition: NOCC = round(3N/4) occupied sites, split into exactly N1
// type-1 and N2 type-2 sites (N0 vacancies), matching cpu/lattice.hpp.
// ---------------------------------------------------------------------------

#include <array>
#include <cmath>
#include <cstdint>

namespace lgl {

constexpr int L = 21;
constexpr int N = L * L * L;                    // 9261
constexpr int NOCC = (3 * N + 2) / 4;           // 6946
constexpr int N1 = (2 * NOCC + 2) / 5;          // 2778
constexpr int N2 = NOCC - N1;                   // 4168
constexpr int N0 = N - NOCC;                    // 2315

// A nonlocal pair can touch at most 14 distinct sites; a lattice bond always
// touches exactly 12 (checked exhaustively by local_tests.cpp).
constexpr int MAX_AFFECTED = 14;

static_assert(N == 9261, "L=21 reference is fixed at N=9261");
static_assert(N0 + N1 + N2 == N, "fixed composition must cover the lattice");

// ---------------------------------------------------------------------------
// Geometry and energy
// ---------------------------------------------------------------------------

inline int site(int x, int y, int z) { return z + L * (y + L * x); }

inline int coord(int s, int axis) {
  if (axis == 0) return s / (L * L);
  if (axis == 1) return (s / L) % L;
  return s % L;
}

inline int neighbor(int s, int d) {
  const int z = s % L;
  const int y = (s / L) % L;
  const int x = s / (L * L);
  const int dx = (d == 1) - (d == 0);
  const int dy = (d == 3) - (d == 2);
  const int dz = (d == 5) - (d == 4);
  const int nx = (x + dx + L) % L;
  const int ny = (y + dy + L) % L;
  const int nz = (z + dz + L) % L;
  return nz + L * (ny + L * nx);
}

inline int manhattan(int a, int b) {
  int d = 0;
  for (int axis = 0; axis < 3; ++axis) {
    int delta = coord(a, axis) - coord(b, axis);
    if (delta < 0) delta = -delta;
    if (delta > L - delta) delta = L - delta;
    d += delta;
  }
  return d;
}

inline int local_energy(int t, int m) {
  if (t == 0) return 0;
  const int conn = (t == 1) ? 3 : 5;
  const int v = __builtin_popcount(static_cast<unsigned>(m)) - conn;
  return v * v;
}

// splitmix64, used only to build deterministic fixed-composition starts.
inline uint64_t splitmix64(uint64_t& x) {
  uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

struct State {
  std::array<uint8_t, N> s{};

  int type(int i) const { return s[i] & 3; }
  int mask(int i) const { return s[i] >> 2; }

  void set_site(int i, int t, int m) {
    s[i] = static_cast<uint8_t>((t & 3) | ((m & 63) << 2));
  }

  // Recompute every mask from the current types. Deterministic.
  void recompute_masks() {
    for (int i = 0; i < N; ++i) {
      int m = 0;
      for (int d = 0; d < 6; ++d) {
        if (type(neighbor(i, d)) != 0) m |= 1 << d;
      }
      set_site(i, type(i), m);
    }
  }

  // Fixed-composition start: uniform random ordering (Fisher-Yates driven by
  // splitmix64), then N1 type-1 sites, N2 type-2 sites, rest vacant.
  void init(uint64_t seed) {
    std::array<int, N> order;
    for (int i = 0; i < N; ++i) order[i] = i;
    uint64_t x = seed;
    for (int i = N - 1; i > 0; --i) {
      const int j = static_cast<int>(splitmix64(x) % static_cast<uint64_t>(i + 1));
      const int t = order[i];
      order[i] = order[j];
      order[j] = t;
    }
    s.fill(0);
    for (int i = 0; i < N1; ++i) set_site(order[i], 1, 0);
    for (int i = 0; i < N2; ++i) set_site(order[N1 + i], 2, 0);
    recompute_masks();
  }

  int full_energy() const {
    int e = 0;
    for (int i = 0; i < N; ++i) e += local_energy(type(i), mask(i));
    return e;
  }

  std::array<int, 3> composition() const {
    std::array<int, 3> c{0, 0, 0};
    for (int i = 0; i < N; ++i) {
      const int t = type(i);
      if (t < 3) ++c[t];
    }
    return c;
  }

  // Every stored mask equals the occupancy of the six neighbors, and every
  // type is in {0,1,2}.
  bool masks_ok() const {
    for (int i = 0; i < N; ++i) {
      if (type(i) > 2) return false;
      int m = 0;
      for (int d = 0; d < 6; ++d) {
        if (type(neighbor(i, d)) != 0) m |= 1 << d;
      }
      if (mask(i) != m) return false;
    }
    return true;
  }

  bool composition_ok() const {
    const std::array<int, 3> c = composition();
    return c[0] == N0 && c[1] == N1 && c[2] == N2;
  }

  bool invariant() const { return masks_ok() && composition_ok(); }
};

// ---------------------------------------------------------------------------
// Affected set, delta, commit
// ---------------------------------------------------------------------------

// Deduplicated A(a,b) = {a,b} U N(a) U N(b), in deterministic order.
// For a bond b == neighbor(a, d) with a != b the returned size is exactly 12;
// for a nonlocal pair the size is at most MAX_AFFECTED.
inline int affected_set(int a, int b, int* out) {
  int n = 0;
  auto add = [&](int q) {
    for (int k = 0; k < n; ++k) {
      if (out[k] == q) return;
    }
    out[n++] = q;
  };
  add(a);
  add(b);
  for (int d = 0; d < 6; ++d) add(neighbor(a, d));
  for (int d = 0; d < 6; ++d) add(neighbor(b, d));
  return n;
}

// Mask of q after the proposed swap. Only the occupancy of a and b changes,
// so a mask bit is rewritten only when it points at a or b.
inline int updated_mask(const State& S, int q, int a, int b, int oa, int ob) {
  int m = S.mask(q);
  for (int d = 0; d < 6; ++d) {
    const int r = neighbor(q, d);
    if (r == a) {
      m = (m & ~(1 << d)) | (oa << d);
    } else if (r == b) {
      m = (m & ~(1 << d)) | (ob << d);
    }
  }
  return m;
}

// Exact integer energy change of the swap on the affected set. a and b must be
// distinct; for a local move b == neighbor(a, d). Caller rejects null moves.
inline int delta(const State& S, int a, int b) {
  int aff[MAX_AFFECTED];
  const int n = affected_set(a, b, aff);
  const int ta = S.type(a);
  const int tb = S.type(b);
  const int oa = (tb != 0); // new occupancy at a
  const int ob = (ta != 0); // new occupancy at b
  int dE = 0;
  for (int k = 0; k < n; ++k) {
    const int q = aff[k];
    const int nt = (q == a) ? tb : ((q == b) ? ta : S.type(q));
    const int nm = updated_mask(S, q, a, b, oa, ob);
    dE += local_energy(nt, nm) - local_energy(S.type(q), S.mask(q));
  }
  return dE;
}

// Apply the swap exactly. All new bytes are computed from the pre-move state
// before any write. A null move (equal types) is a no-op here; the proposal
// layer rejects it instead.
inline void commit(State& S, int a, int b) {
  int aff[MAX_AFFECTED];
  const int n = affected_set(a, b, aff);
  const int ta = S.type(a);
  const int tb = S.type(b);
  const int oa = (tb != 0);
  const int ob = (ta != 0);
  uint8_t nv[MAX_AFFECTED];
  for (int k = 0; k < n; ++k) {
    const int q = aff[k];
    const int nt = (q == a) ? tb : ((q == b) ? ta : S.type(q));
    const int nm = updated_mask(S, q, a, b, oa, ob);
    nv[k] = static_cast<uint8_t>(nt | (nm << 2));
  }
  for (int k = 0; k < n; ++k) S.s[aff[k]] = nv[k];
}

// ---------------------------------------------------------------------------
// Proposal evaluation
// ---------------------------------------------------------------------------

struct Eval {
  bool null_move = false;
  bool accept = false;
  int dE = 0;
};

// Full-precision Metropolis acceptance probability; 1 for dE <= 0.
inline double acceptance_probability(int dE, double beta) {
  if (dE <= 0) return 1.0;
  return std::exp(-beta * static_cast<double>(dE));
}

// Evaluate one proposal with supplied endpoints a, b and uniform u in [0,1).
// The proposal is null iff type(a) == type(b).
inline Eval evaluate(const State& S, int a, int b, double u, double beta) {
  Eval e;
  if (S.type(a) == S.type(b)) {
    e.null_move = true;
    return e;
  }
  e.dE = delta(S, a, b);
  e.accept = (u < acceptance_probability(e.dE, beta));
  return e;
}

// Commit an accepted, non-null evaluation. Returns whether the state changed.
inline bool apply(State& S, int a, int b, const Eval& e) {
  if (e.null_move || !e.accept) return false;
  commit(S, a, b);
  return true;
}

// Evaluate and, if accepted, apply one proposal. Returns whether it changed
// the state.
inline bool step(State& S, int a, int b, double u, double beta) {
  return apply(S, a, b, evaluate(S, a, b, u, beta));
}

} // namespace lgl
