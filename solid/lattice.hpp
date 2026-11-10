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

struct Lattice {
  int L = 21, N = 9261;
  uint32_t lemN = 0, lem6 = 0;
  std::vector<uint8_t> st;

  explicit Lattice(int l = 21) : L(l), N(l * l * l), st(l * l * l, 0) {
    lemN = (0u - uint32_t(N)) % uint32_t(N);
    lem6 = (0u - 6u) % 6u;
  }

  int site(int x, int y, int z) const { return x + L * y + L * L * z; }
  int nb(int s, int d) const {
    int x = s % L, y = (s / L) % L, z = s / (L * L);
    if (d == 0) x = (x + 1) % L;
    else if (d == 1) x = (x + L - 1) % L;
    else if (d == 2) y = (y + 1) % L;
    else if (d == 3) y = (y + L - 1) % L;
    else if (d == 4) z = (z + 1) % L;
    else z = (z + L - 1) % L;
    return site(x, y, z);
  }
  int type(int s) const { return st[s] & 3; }
  int count(int s) const { return __builtin_popcount(unsigned(st[s] >> 2)); }
  // cMask is the six-neighbour occupancy bit mask.
  int localEnergy(int t, int cMask) const {
    if (t == 0) return 0;
    int c = __builtin_popcount(unsigned(cMask) & 63u);
    int d = c - (t == 1 ? 3 : 5);
    return d * d;
  }
  int energy() const {
    int e = 0;
    for (int s = 0; s < N; ++s) e += localEnergy(type(s), int(st[s] >> 2));
    return e;
  }
  int torusL1(int a, int b) const {
    int ax = a % L, ay = (a / L) % L, az = a / (L * L);
    int bx = b % L, by = (b / L) % L, bz = b / (L * L);
    int dx = std::abs(ax - bx), dy = std::abs(ay - by), dz = std::abs(az - bz);
    dx = std::min(dx, L - dx);
    dy = std::min(dy, L - dy);
    dz = std::min(dz, L - dz);
    return dx + dy + dz;
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
      st[s] = uint8_t((st[s] & 3) | (mask << 2));
    }
  }
};

} // namespace solid
