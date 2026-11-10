#pragma once
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>

namespace solid {

inline uint64_t splitmix64(uint64_t &x) {
  x += 0x9e3779b97f4a7c15ULL;
  uint64_t z = x;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

inline uint64_t mix(uint64_t x) {
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

// ---------------------------------------------------------------------------
// Packed site state.
//
// A site is one byte: bits 0-1 hold the occupancy type (0 empty, 1 and 2 the
// two species), bits 2-7 hold a bitmask of which of the six neighbors is
// occupied. A 2-bit type plus a 6-bit mask fill a byte exactly, so this is the
// compact representation: splitting types and masks into separate arrays would
// save no memory, and every state boundary (raw seed files, snapshots, .npz
// output, the Metal backend) already uses this byte layout. Keeping the byte
// layout therefore makes the representation exact by construction; there is no
// wider or lossy alternative to fall back to.
// ---------------------------------------------------------------------------
inline uint8_t packSite(int t, int mask) {
  return uint8_t((t & 3) | ((mask & 63) << 2));
}
inline int siteType(uint8_t v) { return int(v & 3); }
inline int siteMask(uint8_t v) { return int(v >> 2); }

// Local energy of a packed site byte: zero for empty sites, otherwise the
// squared excess of occupied neighbors over the species ideal (3 or 5).
// Written as a constexpr so the 256-entry table below is built at compile time
// and cannot drift from the arithmetic it replaces.
constexpr uint8_t energyFromTypeMask(int t, int cMask) {
  if ((t & 3) == 0) return 0;
  int c = 0;
  for (int b = 0; b < 6; ++b) c += (cMask >> b) & 1;
  const int d = c - (((t & 3) == 1) ? 3 : 5);
  return uint8_t(d * d);
}

struct PackedEnergyTable {
  uint8_t v[256];
  constexpr PackedEnergyTable() : v{} {
    for (int i = 0; i < 256; ++i) v[i] = energyFromTypeMask(i & 3, i >> 2);
  }
};
inline constexpr PackedEnergyTable kPackedEnergy{};

inline int packedEnergy(uint8_t v) { return int(kPackedEnergy.v[v]); }

struct Lattice {
  int L = 21, N = 9261;
  uint32_t lemN = 0, lem6 = 0;
  std::vector<uint8_t> st;
  std::vector<uint16_t> nn;
  std::vector<uint32_t> pc;

  explicit Lattice(int l = 21) : L(l), N(l * l * l), st(l * l * l, 0),
                                 nn(size_t(l * l * l) * 6), pc(size_t(l * l * l)) {
    lemN = (0u - uint32_t(N)) % uint32_t(N);
    lem6 = (0u - 6u) % 6u;
    for (int z = 0; z < L; ++z) for (int y = 0; y < L; ++y) for (int x = 0; x < L; ++x) {
      const int s = x + L * y + L * L * z;
      pc[size_t(s)] = uint32_t(x | (y << 5) | (z << 10));
      nn[size_t(s) * 6 + 0] = uint16_t(x == L - 1 ? s - (L - 1) : s + 1);
      nn[size_t(s) * 6 + 1] = uint16_t(x == 0 ? s + (L - 1) : s - 1);
      nn[size_t(s) * 6 + 2] = uint16_t(y == L - 1 ? s - L * (L - 1) : s + L);
      nn[size_t(s) * 6 + 3] = uint16_t(y == 0 ? s + L * (L - 1) : s - L);
      nn[size_t(s) * 6 + 4] = uint16_t(z == L - 1 ? s - L * L * (L - 1) : s + L * L);
      nn[size_t(s) * 6 + 5] = uint16_t(z == 0 ? s + L * L * (L - 1) : s - L * L);
    }
  }

  int nb(int s, int d) const { return int(nn[size_t(s) * 6 + d]); }

  int energy() const {
    int e = 0;
    const uint8_t* p = st.data();
    for (int s = 0; s < N; ++s) e += kPackedEnergy.v[p[s]];
    return e;
  }

  // Torus L1 distance at most 2? Only that predicate is needed: the far-pair
  // affected set is duplicate free while near pairs must be deduplicated. The
  // check short-circuits after the first coordinate in the common case.
  bool nearPair(int a, int b) const {
    const uint32_t pa = pc[size_t(a)], pb = pc[size_t(b)];
    int dx = std::abs(int(pa & 31u) - int(pb & 31u));
    dx = std::min(dx, L - dx);
    if (dx > 2) return false;
    int dy = std::abs(int((pa >> 5) & 31u) - int((pb >> 5) & 31u));
    dy = std::min(dy, L - dy);
    if (dx + dy > 2) return false;
    int dz = std::abs(int((pa >> 10) & 31u) - int((pb >> 10) & 31u));
    dz = std::min(dz, L - dz);
    return dx + dy + dz <= 2;
  }

  void init(uint64_t seed, int n1, int n2) {
    if (n1 < 0) n1 = 0;
    if (n2 < 0) n2 = 0;
    if (n1 + n2 > N) { n2 = std::max(0, N - n1); }
    std::vector<int> p(N);
    for (int i = 0; i < N; ++i) p[i] = i;
    uint64_t x = seed;
    for (int i = N - 1; i >= 1; --i) {
      int j = int(splitmix64(x) % uint64_t(i + 1));
      std::swap(p[i], p[j]);
    }
    std::fill(st.begin(), st.end(), 0);
    for (int i = 0; i < n1; ++i) st[p[i]] = 1;
    for (int i = 0; i < n2; ++i) st[p[n1 + i]] = 2;
    for (int s = 0; s < N; ++s) {
      int mask = 0;
      for (int d = 0; d < 6; ++d) if ((st[nb(s, d)] & 3) != 0) mask |= (1 << d);
      st[s] = packSite(st[s] & 3, mask);
    }
  }
};

} // namespace solid
