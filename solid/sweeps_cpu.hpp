#pragma once
#include "lattice.hpp"
#include <vector>
#include <cstdint>

namespace solid {

struct Aff { int s; uint8_t ba, bb, ns; };

inline void addAff(Aff* a, int& n, int s, int which, int bit) {
  for (int i = 0; i < n; ++i) if (a[i].s == s) {
    if (which == 0) a[i].ba = uint8_t(bit + 1);
    if (which == 1) a[i].bb = uint8_t(bit + 1);
    return;
  }
  a[n].s = s;
  a[n].ba = (which == 0) ? uint8_t(bit + 1) : 0;
  a[n].bb = (which == 1) ? uint8_t(bit + 1) : 0;
  a[n].ns = 0;
  ++n;
}

struct Proposal { int a, b; uint32_t u; bool valid; };

inline Proposal drawProposal(const Lattice& lat, uint64_t seed, int chain,
                             uint64_t index, bool local) {
  Proposal p{0, 0, 0, false};
  uint64_t r0 = mix(seed ^ (uint64_t(chain) * 0x9e3779b97f4a7c15ULL) ^
                    (index * 0xd1342543de82ef95ULL) ^ (local ? 0x1234u : 0x5678u));
  uint64_t r1 = mix(r0 ^ 0xa0761d6478bd642fULL);
  uint64_t r2 = mix(r1 ^ 0xe7037ed1a0b428dbULL);
  p.u = uint32_t(r2 >> 32);
  const int a = int(r0 % uint64_t(lat.N));
  if (local) {
    p.a = a;
    p.b = lat.nb(a, int(r1 % 6));
    p.valid = true;
  } else {
    const int b = int(r1 % uint64_t(lat.N));
    p.a = a; p.b = b;
    p.valid = (a != b) && lat.torusL1(a, b) >= 3;
  }
  return p;
}

inline bool tryProposal(Lattice& lat, const Proposal& p,
                        const std::vector<uint32_t>& th, int& energy) {
  if (!p.valid) return false;
  const int a = p.a, b = p.b;
  const int ta = lat.type(a), tb = lat.type(b);
  if (ta == tb) return false;
  Aff aff[14]; int na = 0;
  addAff(aff, na, a, 2, 0);
  addAff(aff, na, b, 2, 0);
  for (int j = 0; j < 6; ++j) addAff(aff, na, lat.nb(a, j), 0, j ^ 1);
  for (int j = 0; j < 6; ++j) addAff(aff, na, lat.nb(b, j), 1, j ^ 1);
  const int oa = (tb != 0), ob = (ta != 0);
  int oldE = 0, newE = 0;
  for (int i = 0; i < na; ++i) {
    const int s = aff[i].s;
    const uint8_t v = lat.st[s];
    const int ts = v & 3, ms = v >> 2;
    oldE += lat.localEnergy(ts, ms);
    int nm = ms;
    if (aff[i].ba) { int bit = aff[i].ba - 1; nm = (nm & ~(1 << bit)) | (oa << bit); }
    if (aff[i].bb) { int bit = aff[i].bb - 1; nm = (nm & ~(1 << bit)) | (ob << bit); }
    const int nt = (s == a) ? tb : ((s == b) ? ta : ts);
    aff[i].ns = uint8_t(nt | (nm << 2));
    newE += lat.localEnergy(nt, nm);
  }
  const int dE = newE - oldE;
  bool ok = dE <= 0;
  if (!ok) {
    const int idx = dE + 512;
    if (idx >= 0 && idx < int(th.size())) ok = p.u < th[idx];
  }
  if (!ok) return false;
  for (int i = 0; i < na; ++i) lat.st[aff[i].s] = aff[i].ns;
  energy += dE;
  return true;
}

inline std::vector<uint32_t> makeThresholds(double beta) {
  std::vector<uint32_t> t(1024);
  for (int i = 0; i < 1024; ++i) {
    int dE = i - 512;
    if (dE <= 0) t[i] = 0xffffffffu;
    else {
      double p = std::exp(-beta * double(dE)) * 4294967296.0;
      t[i] = uint32_t(std::min(std::max(p, 0.0), 4294967295.0));
    }
  }
  return t;
}

inline int nonlocalSweep(Lattice& lat, uint64_t seed, int chain, uint64_t sweep,
                         const std::vector<uint32_t>& th, int& energy) {
  int accepted = 0;
  const uint64_t index0 = (2 * sweep + 0) * uint64_t(lat.N);
  for (int t = 0; t < lat.N; ++t) {
    Proposal p = drawProposal(lat, seed, chain, index0 + uint64_t(t), false);
    if (tryProposal(lat, p, th, energy)) ++accepted;
  }
  return accepted;
}

inline int localSweep(Lattice& lat, uint64_t seed, int chain, uint64_t sweep,
                      const std::vector<uint32_t>& th, int& energy) {
  int accepted = 0;
  const uint64_t index0 = (2 * sweep + 1) * uint64_t(lat.N);
  for (int t = 0; t < lat.N; ++t) {
    Proposal p = drawProposal(lat, seed, chain, index0 + uint64_t(t), true);
    if (tryProposal(lat, p, th, energy)) ++accepted;
  }
  return accepted;
}

} // namespace solid
