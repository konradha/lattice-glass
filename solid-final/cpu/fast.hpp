#pragma once
#include "sweeps_cpu.hpp"
#include <stdexcept>
#include <string>

// Verbatim opt3 cpu/fast.cpp helpers and Fast<K>; benchmark/check main omitted.
namespace {
using namespace solid;

void require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}
int ell(int type) { return type == 2 ? 5 : 3; }
int base(int type, int m) {
  return m * (m + 1) / 2 - (type != 0 ? (ell(type) + 1) * m : 0);
}
int qvalue(int type, int m) { return type != 0 ? m - ell(type) : 0; }
int ownEnergy(int type, int m) {
  const int d = m - ell(type);
  return type != 0 ? d * d : 0;
}

// This is the frozen drawProposal with only its signed-int chain interface
// widened. All actual random operations are the unchanged frozen helpers.
Proposal draw64(const Lattice& geometry, uint64_t seed, uint64_t chain,
                uint64_t index, bool local) {
  Proposal p;
  p.local = local;
  p.rng.key = proposalKey(seed, chain, index, local);
  p.a = int(uniformIndex(p.rng, uint32_t(geometry.N), geometry.lemN));
  p.b = local ? geometry.nb(p.a, int(uniformIndex(p.rng, 6, geometry.lem6)))
              : int(uniformIndex(p.rng, uint32_t(geometry.N), geometry.lemN));
  p.valid = local || p.a != p.b;
  if (p.valid) p.u_hi = p.rng.next();
  return p;
}
uint64_t indexAt(uint64_t sweep, bool local, int N, uint64_t offset) {
  return (2 * sweep + uint64_t(local)) * uint64_t(N) + offset;
}
struct Outcome {
  int delta = 0;
  uint32_t flags = 0;  // valid=1, equal-type no-op=2, accepted=4
};

template<bool K> class Fast {
  const Lattice& g;
  std::vector<uint16_t> cell;
  int cached(int s) const { return int(cell[size_t(s)] >> 5) - (K ? 64 : 32); }
  int potential(int s) const {
    return K ? cached(s) : cached(s) + base(type(s), count(s));
  }
  struct Owner { int site, dn, dq; };
  static void add(Owner* owners, int& n, int site, int dn) {
    for (int i = 0; i < n; ++i) if (owners[i].site == site) {
      owners[i].dn += dn;
      return;
    }
    owners[n++] = {site, dn, 0};
  }
  void commit(int a, int b) {
    const int ta = type(a), tb = type(b);
    if (ta != 0 && tb != 0) {
      // Occupancy/counts stay fixed; preserve S while changing type and owned B.
      const int dq = ell(ta) - ell(tb);
      cell[size_t(a)] = uint16_t(int(cell[size_t(a)]) + tb - ta +
                                (K ? count(a) * dq * 32 : 0));
      cell[size_t(b)] = uint16_t(int(cell[size_t(b)]) + ta - tb -
                                (K ? count(b) * dq * 32 : 0));
      // Additive scatters retain adjacent-endpoint and common-neighbor terms.
      for (int d = 0; d < 6; ++d) {
        const int sa = g.nb(a, d), sb = g.nb(b, d);
        cell[size_t(sa)] = uint16_t(int(cell[size_t(sa)]) + dq * 32);
        cell[size_t(sb)] = uint16_t(int(cell[size_t(sb)]) - dq * 32);
      }
      return;
    }
    Owner owners[14];
    int n = 0;
    add(owners, n, a, 0);
    add(owners, n, b, 0);
    const int dn = int(tb != 0) - int(ta != 0);
    if (dn != 0) {
      for (int d = 0; d < 6; ++d) add(owners, n, g.nb(a, d), dn);
      for (int d = 0; d < 6; ++d) add(owners, n, g.nb(b, d), -dn);
    }
    // Gathered net changes mean adjacent/common sites are updated only once.
    for (int i = 0; i < n; ++i) {
      const int s = owners[i].site;
      const int ot = type(s), om = count(s);
      const int nt = s == a ? tb : (s == b ? ta : ot);
      const int nm = om + owners[i].dn;
      owners[i].dq = qvalue(nt, nm) - qvalue(ot, om);
      const int shift = K ? (base(nt, nm) - base(ot, om)) * 32 : 0;
      cell[size_t(s)] = uint16_t(int(cell[size_t(s)]) + shift +
                                (nt + 4 * nm) - (ot + 4 * om));
    }
    for (int i = 0; i < n; ++i) if (owners[i].dq != 0) {
      for (int d = 0; d < 6; ++d) {
        const int s = g.nb(owners[i].site, d);
        cell[size_t(s)] = uint16_t(int(cell[size_t(s)]) + owners[i].dq * 32);
      }
    }
  }
public:
  int energy;
  uint64_t accepted = 0;
  explicit Fast(const Lattice& initial)
      : g(initial), cell(size_t(initial.N)), energy(initial.energy()) {
    for (int s = 0; s < g.N; ++s) {
      const int t = siteType(initial.st[size_t(s)]);
      const int m = __builtin_popcount(unsigned(siteMask(initial.st[size_t(s)])));
      int S = 0;
      for (int d = 0; d < 6; ++d) {
        const uint8_t v = initial.st[size_t(g.nb(s, d))];
        S += qvalue(siteType(v), __builtin_popcount(unsigned(siteMask(v))));
      }
      const int value = K ? S + base(t, m) + 64 : S + 32;
      cell[size_t(s)] = uint16_t(t + 4 * m + 32 * value);
    }
  }
  int type(int s) const { return cell[size_t(s)] & 3; }
  int count(int s) const { return (cell[size_t(s)] >> 2) & 7; }
  int delta(int a, int b, bool local) const {
    const int ta = type(a), tb = type(b);
    const int ma = count(a), mb = count(b);
    if (ta != 0 && tb != 0)
      return ownEnergy(tb, ma) + ownEnergy(ta, mb) -
             ownEnergy(ta, ma) - ownEnergy(tb, mb);
    if (ta == 0) std::swap(a, b);
    const int t = type(a), e = ell(t);
    int half = potential(b) - potential(a) - e * count(b);
    if (local || g.nearPair(a, b)) {
      bool adjacent = local;
      if (!adjacent) for (int d = 0; d < 6; ++d)
        if (g.nb(a, d) == b) adjacent = true;
      if (adjacent) {
        half -= count(a) + count(b) - 2 * e;
      } else {
        for (int d = 0; d < 6; ++d) {
          const int s = g.nb(a, d);
          if (type(s) == 0) continue;
          for (int j = 0; j < 6; ++j) if (s == g.nb(b, j)) --half;
        }
      }
    }
    return 2 * half;
  }
  Outcome attempt(Proposal& p, const std::vector<Threshold>& th) {
    Outcome out;
    if (!p.valid) return out;
    out.flags = 1;
    if (type(p.a) == type(p.b)) { out.flags |= 2; return out; }
    out.delta = delta(p.a, p.b, p.local);
    if (!acceptEFMA(p.u_hi, p.rng, th, out.delta)) return out;
    commit(p.a, p.b);
    energy += out.delta;
    ++accepted;
    out.flags |= 4;
    return out;
  }
  uint8_t byte(int s) const {
    int mask = 0;
    for (int d = 0; d < 6; ++d) if (type(g.nb(s, d)) != 0) mask |= 1 << d;
    return packSite(type(s), mask);
  }
  uint64_t checksum() const {
    uint64_t h = 14695981039346656037ULL;
    for (int s = 0; s < g.N; ++s) { h ^= byte(s); h *= 1099511628211ULL; }
    return h;
  }
  void audit(const Lattice& ref) const {
    int full = 0;
    for (int s = 0; s < g.N; ++s) {
      require(byte(s) == ref.st[size_t(s)], "canonical state byte at site " + std::to_string(s));
      const int m = __builtin_popcount(unsigned(siteMask(ref.st[size_t(s)])));
      require(count(s) == m, "cached count at site " + std::to_string(s));
      int S = 0;
      for (int d = 0; d < 6; ++d) {
        const uint8_t v = ref.st[size_t(g.nb(s, d))];
        S += qvalue(siteType(v), __builtin_popcount(unsigned(siteMask(v))));
      }
      require(cached(s) == S + (K ? base(type(s), m) : 0),
              "cached potential at site " + std::to_string(s));
      full += packedEnergy(ref.st[size_t(s)]);
    }
    require(full == energy && full == ref.energy(), "full recomputed energy");
  }
};

} // namespace
