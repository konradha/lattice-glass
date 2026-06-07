#ifndef LATTICE_GLASS_COLLECTIVE_MOVES_H
#define LATTICE_GLASS_COLLECTIVE_MOVES_H

// Collective occupancy moves for the lattice glass.
//
// The cooperative vacancy chain: pick a vacancy, then grow a self-avoiding
// directed path of k occupied sites; shifting each particle one step back
// along the path transports the vacancy from the start to the end of the path
// (a "vacancy soliton" dragging a line of particles). This is a single
// collective, non-local move on the occupancy field.
//
// Detailed balance: the proposal is symmetric. The forward move selects the
// start site (prob 1/N over all sites), a length k (uniform on [1,k_max]), and
// k directions (each 1/6 on the cubic lattice), subject to every stepped-onto
// site being occupied and the path self-avoiding. The reverse move starts at
// the path's end (now the vacancy), takes the opposite directions, and walks
// the shifted particles back; every stepped-onto site is occupied and the path
// is self-avoiding by construction. Both proposals therefore have probability
// (1/N)(1/k_max)(1/6)^k, so g(x->x') = g(x'->x). With Metropolis acceptance
// min(1, exp(-beta dH)) the move is reversible w.r.t. the Gibbs measure.

#include "fp_sampler.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace lattice_glass {
namespace collective {

struct ChainResult {
  bool proposed = false; // a valid self-avoiding occupied path was built
  bool accepted = false;
  int length = 0;
};

// Attempt one cooperative vacancy-chain move. Mutates lattice on acceptance.
inline ChainResult vacancy_chain_move(std::vector<uint8_t> &lattice,
                                      long double beta, int k_max,
                                      std::mt19937 &gen, const int *nn) {
  const int lat_size = static_cast<int>(lattice.size());
  std::uniform_int_distribution<int> site_dist(0, lat_size - 1);
  std::uniform_int_distribution<int> dir_dist(0, fp::kNumNeighbors - 1);
  std::uniform_int_distribution<int> len_dist(1, k_max);
  std::uniform_real_distribution<double> uni(0.0, 1.0);

  ChainResult result;
  const int v0 = site_dist(gen);
  if (lattice[v0] != cluster::kEmpty)
    return result; // start must be a vacancy (symmetric null otherwise)

  const int k = len_dist(gen);
  std::vector<int> path;
  path.reserve(k + 1);
  path.push_back(v0);

  int cur = v0;
  for (int step = 0; step < k; ++step) {
    const int dir = dir_dist(gen);
    const int nb = nn[fp::kNumNeighbors * cur + dir];
    if (lattice[nb] == cluster::kEmpty)
      return result; // must step onto an occupied site
    bool seen = false;
    for (const int p : path)
      if (p == nb) {
        seen = true;
        break;
      }
    if (seen)
      return result; // self-avoiding
    path.push_back(nb);
    cur = nb;
  }
  result.proposed = true;
  result.length = k;

  // Affected region: every path site plus its neighbourhood (deduplicated).
  std::vector<int> affected;
  affected.reserve(path.size() * (fp::kNumNeighbors + 1));
  auto push = [&](int site) {
    for (const int a : affected)
      if (a == site)
        return;
    affected.push_back(site);
  };
  for (const int site : path) {
    push(site);
    for (int d = 0; d < fp::kNumNeighbors; ++d)
      push(nn[fp::kNumNeighbors * site + d]);
  }

  long double e_before = 0.0L;
  for (const int site : affected)
    e_before += fp::local_energy_int(lattice, site, nn);

  // Shift particles one step back along the path: v_{t-1} <- v_t, end emptied.
  std::vector<uint8_t> original(path.size());
  for (int t = 0; t < static_cast<int>(path.size()); ++t)
    original[t] = lattice[path[t]];
  for (int t = 1; t < static_cast<int>(path.size()); ++t)
    lattice[path[t - 1]] = original[t];
  lattice[path.back()] = cluster::kEmpty;

  long double e_after = 0.0L;
  for (const int site : affected)
    e_after += fp::local_energy_int(lattice, site, nn);

  const long double delta = e_after - e_before;
  if (delta <= 0.0L ||
      uni(gen) < std::exp(-static_cast<double>(beta * delta))) {
    result.accepted = true;
    return result;
  }

  // Reject: restore.
  for (int t = 0; t < static_cast<int>(path.size()); ++t)
    lattice[path[t]] = original[t];
  return result;
}

struct SweepStats {
  int attempts = 0;
  int proposed = 0;
  int accepted = 0;
  long double accepted_length_sum = 0.0L;
};

inline SweepStats vacancy_chain_sweep(std::vector<uint8_t> &lattice,
                                      long double beta, int k_max, int attempts,
                                      std::mt19937 &gen, const int *nn) {
  SweepStats stats;
  for (int i = 0; i < attempts; ++i) {
    const ChainResult r = vacancy_chain_move(lattice, beta, k_max, gen, nn);
    ++stats.attempts;
    stats.proposed += r.proposed;
    if (r.accepted) {
      ++stats.accepted;
      stats.accepted_length_sum += r.length;
    }
  }
  return stats;
}

} // namespace collective
} // namespace lattice_glass

#endif // LATTICE_GLASS_COLLECTIVE_MOVES_H
