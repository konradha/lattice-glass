#ifndef LATTICE_GLASS_LIFTED_SWAP_H
#define LATTICE_GLASS_LIFTED_SWAP_H

// Irreversible lifted nonlocal occupancy swap via skew detailed balance
// (Turitsyn-Chertkov-Vucelja 2011; Vucelja review arXiv:1412.8762).
//
// Lifting variable: an active vacancy h together with a momentum
// (axis a in {0,1,2}, sign s in {+1,-1}), encoded as dir = 2*a + (s>0). One
// event proposes teleporting the active vacancy a fixed `stride` along the
// momentum axis (periodic), swapping it with the particle found there:
//   n = h shifted by s*stride along axis a;   particle(n) -> h, vacancy -> n.
// On Metropolis acceptance the vacancy advances persistently (h <- n, momentum
// kept); on rejection the momentum is reversed (dir <- dir^1) and the
// configuration is unchanged. stride = 1 reduces to nearest-neighbour lifted
// transport; stride > 1 is a genuinely nonlocal swap that remains ballistic.
//
// Skew detailed balance: pair each accepting event with its momentum mirror.
//   forward  (x, h, (a,+s)) -> (x', n, (a,+s))   [vacancy h -> n]
//   mirror   (x', n, (a,-s)) -> (x, h, (a,-s))   [vacancy n -> h]
// On a periodic axis, shifting n by -s*stride returns h exactly, so the mirror
// reconstructs x. Since dH(x->x') = -dH(x'->x), the Metropolis factors satisfy
//   pi(x) A(x->x') = pi(x') A(x'->x) = min(pi(x), pi(x')),
// i.e. the within-momentum flow in +s mirrors the reversed flow in -s. The
// rejection mass is routed into the momentum flip dir^1 in place, and a
// configuration-independent refresh of (h, dir) leaves the configuration
// marginal invariant while restoring irreducibility. Hence the configuration
// marginal is the Gibbs measure at fixed composition. Verified empirically by
// tests/test_lifted_swap.cpp (conservation + stationarity against swap).
//
// Note (why this is the tractable skew-DB lever here): the cSwap cascade of
// Ghimenti-Berthier is collective only for many size families; on the 3-symbol
// NH alphabet it degenerates, and the label sector is in any case unfrustrated.
// The occupancy channel is the slow one, so we lift occupancy transport.

#include "fp_sampler.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace lattice_glass {
namespace axis_lift {

struct EventStats {
  long long events = 0;
  long long accepted = 0;
  long long flips = 0;
  long long blocked = 0;
};

inline int random_vacancy(const std::vector<uint8_t> &lattice,
                          std::mt19937 &gen) {
  std::uniform_int_distribution<int> site_dist(
      0, static_cast<int>(lattice.size()) - 1);
  int site = site_dist(gen);
  while (lattice[site] != cluster::kEmpty)
    site = site_dist(gen);
  return site;
}

// Site reached from `site` by shifting `signed_stride` along `axis` on an
// L^3 periodic cubic lattice with index k + L*(j + i*L).
inline int shift_site(int site, int axis, int signed_stride, int L) {
  const int k = site % L;
  const int j = (site / L) % L;
  const int i = site / (L * L);
  int ni = i, nj = j, nk = k;
  if (axis == 0)
    ni = ((i + signed_stride) % L + L) % L;
  else if (axis == 1)
    nj = ((j + signed_stride) % L + L) % L;
  else
    nk = ((k + signed_stride) % L + L) % L;
  return nk + L * (nj + ni * L);
}

// Run a lifted nonlocal swap chain for `events` elementary events.
inline EventStats lifted_swap_chain(std::vector<uint8_t> &lattice,
                                    long double beta, long long events,
                                    int stride, int refresh_period, int L,
                                    std::mt19937 &gen, const int *nn) {
  EventStats stats;
  std::uniform_int_distribution<int> dir_dist(0, 5); // 2*axis + (sign>0)
  std::uniform_real_distribution<double> uni(0.0, 1.0);

  int h = random_vacancy(lattice, gen);
  int dir = dir_dist(gen);

  for (long long e = 0; e < events; ++e) {
    if (refresh_period > 0 && e % refresh_period == 0) {
      h = random_vacancy(lattice, gen);
      dir = dir_dist(gen);
    }
    ++stats.events;

    const int axis = dir >> 1;
    const int sign = (dir & 1) ? +1 : -1;
    const int n = shift_site(h, axis, sign * stride, L);

    if (lattice[n] == cluster::kEmpty) {
      dir ^= 1; // cannot push a vacancy onto a vacancy; reverse momentum
      ++stats.blocked;
      continue;
    }

    const int before = fp::swap_region_energy(lattice, h, n, nn);
    std::swap(lattice[h], lattice[n]);
    const int after = fp::swap_region_energy(lattice, h, n, nn);
    const long double delta = static_cast<long double>(after - before);

    if (delta <= 0.0L ||
        uni(gen) < std::exp(-static_cast<double>(beta * delta))) {
      h = n; // vacancy advances, momentum preserved
      ++stats.accepted;
    } else {
      std::swap(lattice[h], lattice[n]); // revert
      dir ^= 1;                          // reverse momentum
      ++stats.flips;
    }
  }
  return stats;
}

} // namespace axis_lift
} // namespace lattice_glass

#endif // LATTICE_GLASS_LIFTED_SWAP_H
