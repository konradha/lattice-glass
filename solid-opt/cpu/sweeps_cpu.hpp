#pragma once
#include "lattice.hpp"
#include "efma.hpp"
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

// One proposal: endpoints, exact index stream, and the first acceptance word.
// The low acceptance word is drawn lazily by tryProposal only on a hi-word tie.
struct Proposal {
  int a = 0, b = 0;
  bool valid = false;
  bool local = false;
  uint32_t u_hi = 0;
  WordStream rng;
};

inline Proposal drawProposal(const Lattice& lat, uint64_t seed, int chain,
                             uint64_t index, bool local) {
  Proposal p;
  p.local = local;
  p.rng.key = proposalKey(seed, uint64_t(chain), index, local);
  const int a = int(uniformIndex(p.rng, uint32_t(lat.N), lat.lemN));
  if (local) {
    const int dir = int(uniformIndex(p.rng, 6u, lat.lem6));
    p.a = a;
    p.b = lat.nb(a, dir);
    p.valid = true;
    p.u_hi = p.rng.next();
  } else {
    const int b = int(uniformIndex(p.rng, uint32_t(lat.N), lat.lemN));
    p.a = a;
    p.b = b;
    p.valid = (a != b);
    if (p.valid) p.u_hi = p.rng.next();
  }
  return p;
}

inline void addFixed(Aff* a, int& n, int s, int which, int bit) {
  a[n].s = s;
  a[n].ba = (which == 0) ? uint8_t(bit + 1) : 0;
  a[n].bb = (which == 1) ? uint8_t(bit + 1) : 0;
  a[n].ns = 0;
  ++n;
}

inline bool tryProposal(Lattice& lat, Proposal& p,
                        const std::vector<Threshold>& th, int& energy) {
  if (!p.valid) return false;
  const int a = p.a, b = p.b;
  const int ta = lat.type(a), tb = lat.type(b);
  if (ta == tb) return false;
  Aff aff[14]; int na = 0;
  if (p.local) {
    addFixed(aff, na, a, 2, 0); addFixed(aff, na, b, 2, 0);
    for (int j = 0; j < 6; ++j) {
      int q = lat.nb(a, j);
      if (q == b) aff[1].ba = uint8_t((j ^ 1) + 1);
      else addFixed(aff, na, q, 0, j ^ 1);
    }
    for (int j = 0; j < 6; ++j) {
      int q = lat.nb(b, j);
      if (q == a) aff[0].bb = uint8_t((j ^ 1) + 1);
      else addFixed(aff, na, q, 1, j ^ 1);
    }
  } else if (lat.torusL1(a, b) > 2) {
    addFixed(aff, na, a, 2, 0); addFixed(aff, na, b, 2, 0);
    for (int j = 0; j < 6; ++j) addFixed(aff, na, lat.nb(a, j), 0, j ^ 1);
    for (int j = 0; j < 6; ++j) addFixed(aff, na, lat.nb(b, j), 1, j ^ 1);
  } else {
    addAff(aff, na, a, 2, 0);
    addAff(aff, na, b, 2, 0);
    for (int j = 0; j < 6; ++j) addAff(aff, na, lat.nb(a, j), 0, j ^ 1);
    for (int j = 0; j < 6; ++j) addAff(aff, na, lat.nb(b, j), 1, j ^ 1);
  }
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
  if (!acceptEFMA(p.u_hi, p.rng, th, dE)) return false;
  for (int i = 0; i < na; ++i) lat.st[aff[i].s] = aff[i].ns;
  energy += dE;
  return true;
}

inline int nonlocalSweep(Lattice& lat, uint64_t seed, int chain, uint64_t sweep,
                         const std::vector<Threshold>& th, int& energy) {
  int accepted = 0;
  const uint64_t index0 = (2 * sweep + 0) * uint64_t(lat.N);
  for (int t = 0; t < lat.N; ++t) {
    Proposal p = drawProposal(lat, seed, chain, index0 + uint64_t(t), false);
    if (tryProposal(lat, p, th, energy)) ++accepted;
  }
  return accepted;
}

inline int localSweep(Lattice& lat, uint64_t seed, int chain, uint64_t sweep,
                      const std::vector<Threshold>& th, int& energy) {
  int accepted = 0;
  const uint64_t index0 = (2 * sweep + 1) * uint64_t(lat.N);
  for (int t = 0; t < lat.N; ++t) {
    Proposal p = drawProposal(lat, seed, chain, index0 + uint64_t(t), true);
    if (tryProposal(lat, p, th, energy)) ++accepted;
  }
  return accepted;
}

} // namespace solid
