#ifndef LATTICE_GLASS_LIFTED_VACANCY_H
#define LATTICE_GLASS_LIFTED_VACANCY_H

// Irreversible lifted vacancy chain (lifted Metropolis / TCV skewed detailed
// balance) for the occupancy field.
//
// Lifting variable: an active vacancy site h and a direction d (one of the 6
// lattice axes). One event proposes pushing the active vacancy along d, i.e.
// the particle at n = h+d hops into h and the vacancy follows to n. The event
// is accepted with the Metropolis factor min(1, exp(-beta dH)); on acceptance
// the vacancy advances persistently (h<-n, keep d); on rejection the momentum
// is reversed (d <- -d) and the configuration is unchanged.
//
// Correctness (global balance via skewed detailed balance, TCV 2011):
// pair each forward event with its mirror under d -> -d. The accepting
// transition (s,h,d) -> (s',n,d) has mirror (s',n,-d) -> (s,h,-d): pushing the
// vacancy at n along -d swaps the particle at n+(-d)=h back, restoring s. Since
// dH(s->s') = -dH(s'->s), the Metropolis factors satisfy
//   pi(s) A(s->s') = pi(s') A(s'->s) = min(pi(s), pi(s')),
// and the rejection mass flips d in place. Marginalising the direction gives
// pi as stationary. Periodically refreshing (h,d) from a config-independent
// law leaves the configuration marginal invariant and restores irreducibility.
// The configuration marginal is therefore the Gibbs measure at fixed
// composition. This is verified empirically by tests/test_lifted_vacancy.cpp.

#include "fp_sampler.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace lattice_glass {
namespace lifted {

// Neighbour ordering in fp::cubic_neighbors is [i-1,i+1,j-1,j+1,k-1,k+1], so
// the opposite of direction d is d^1 (pairs (0,1),(2,3),(4,5)).
inline int opposite_direction(int d) { return d ^ 1; }

struct EventStats {
  long long events = 0;
  long long accepted = 0;
  long long flips = 0;
  long long blocked = 0; // neighbour was itself a vacancy
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

// Run a lifted vacancy chain for `events` elementary events. The lifting
// variable (active vacancy + direction) is refreshed every `refresh_period`
// events from a configuration-independent law.
inline EventStats lifted_vacancy_chain(std::vector<uint8_t> &lattice,
                                       long double beta, long long events,
                                       int refresh_period, std::mt19937 &gen,
                                       const int *nn) {
  EventStats stats;
  std::uniform_int_distribution<int> dir_dist(0, fp::kNumNeighbors - 1);
  std::uniform_real_distribution<double> uni(0.0, 1.0);

  int h = random_vacancy(lattice, gen);
  int d = dir_dist(gen);

  for (long long e = 0; e < events; ++e) {
    if (refresh_period > 0 && e % refresh_period == 0) {
      h = random_vacancy(lattice, gen); // config-independent refresh
      d = dir_dist(gen);
    }
    ++stats.events;

    const int n = nn[fp::kNumNeighbors * h + d];
    if (lattice[n] == cluster::kEmpty) {
      d = opposite_direction(d); // cannot push a vacancy; reverse momentum
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
      d = opposite_direction(d);         // reverse momentum
      ++stats.flips;
    }
  }
  return stats;
}

} // namespace lifted
} // namespace lattice_glass

#endif // LATTICE_GLASS_LIFTED_VACANCY_H
