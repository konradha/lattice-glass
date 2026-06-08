#ifndef LATTICE_GLASS_INFORMED_SWAP_H
#define LATTICE_GLASS_INFORMED_SWAP_H

// Locally-balanced ("informed") nonlocal occupancy swap (Zanella, Informed
// proposals for local MCMC in discrete spaces, arXiv:1711.07424).
//
// The slow channel of the NH lattice glass at fixed composition is the
// OCCUPANCY field: a particle must hop into a vacancy, and the standard
// nonlocal swap proposes the destination vacancy UNIFORMLY, so at low T almost
// every proposal lands in a high-energy hole and is rejected (acceptance
// collapse). This kernel keeps the exact same move set -- relocate one particle
// to a vacancy -- but chooses the destination with the locally-balanced weight
//   g(pi(y)/pi(x)) = sqrt(exp(-beta * dE)),
// steering proposals toward the rare low-dE vacancies. dE is the EXACT O(1)
// region energy change (no gradient surrogate), so this is classical analytic
// MCMC, not ML.
//
// Exact Metropolis-Hastings with the sqrt balancing function g(t)=sqrt(t),
// which satisfies g(t)=t*g(1/t). Move: pick source i uniformly among the N_p
// occupied sites; pick destination j among the N_v vacant sites with
// probability w_x(i,j)/Z_i(x), w_x(i,v)=sqrt(exp(-beta dE_x(i->v))),
// Z_i(x)=sum_{v vacant} w_x(i,v). The reverse move (in the post-swap config y,
// site j is occupied and i is vacant) picks source j uniformly and destination
// i with probability w_y(j,i)/Z_j(y). Because composition is conserved the two
// uniform source factors 1/N_p cancel, the base proposal is symmetric, and with
// g=sqrt the pi and weight factors telescope:
//   alpha = min(1, [pi(y) q(y->x)] / [pi(x) q(x->y)]) = min(1, Z_i(x)/Z_j(y)).
// (dE_y(j->i) = -dE_x(i->j) makes the exp factors cancel exactly.) Both Z's are
// strictly positive -- Z_j(y) always contains the reverse term v=i with weight
// 1/w_x(i,j) > 0 -- so the ratio is well defined.
//
// Ergodicity: relocating labelled particles through vacancies generates every
// fixed-composition configuration (sliding-puzzle argument; there is always at
// least one vacancy at rho<1), so the kernel alone samples the full Gibbs
// measure -- species labels mix as they ride along with the particles.
//
// blind_occupancy_sweep is the matched control: identical move set (occupied ->
// vacant), UNIFORM destination, plain Metropolis min(1, exp(-beta dE)). The two
// sweeps differ only in the destination proposal, isolating the effect of
// informing it. Cost: blind is O(1) per attempt; informed is O(N_v) per attempt
// (it scans all vacancies twice to form Z_i(x) and Z_j(y)).
//
// Energy changes use an analytic O(1) routine (move_delta_energy) backed by a
// maintained occupied-neighbour-count array m[]; tests/test_informed_swap.cpp
// checks it equals the region-recompute reference exactly, then checks
// conservation and stationarity of the informed-only chain against the trusted
// nonlocal swap (energy and a pure-occupancy structural observable).

#include "fp_sampler.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace lattice_glass {
namespace informed {

struct SweepStats {
  long long attempts = 0;
  long long accepted = 0;
  long double accept_prob_sum = 0.0L; // sum of the Metropolis acceptance prob

  double acceptance() const {
    return attempts > 0 ? static_cast<double>(accepted) / attempts : 0.0;
  }
  double mean_accept_prob() const {
    return attempts > 0 ? static_cast<double>(accept_prob_sum / attempts) : 0.0;
  }
};

inline int preferred_coordination(uint8_t label) {
  return label == cluster::kType1 ? 3 : 5;
}

// Occupied-neighbour count per site, m[s] = #occupied nearest neighbours of s.
inline std::vector<int> build_neighbor_counts(const std::vector<uint8_t> &lattice,
                                              const int *nn) {
  const int n = static_cast<int>(lattice.size());
  std::vector<int> m(n, 0);
  for (int site = 0; site < n; ++site) {
    const int *neigh = nn + fp::kNumNeighbors * site;
    int c = 0;
    for (int k = 0; k < fp::kNumNeighbors; ++k)
      c += lattice[neigh[k]] != cluster::kEmpty;
    m[site] = c;
  }
  return m;
}

// Update m[] for an accepted relocation i (occupied -> empty) to j (empty ->
// occupied): every neighbour of i loses an occupied neighbour, every neighbour
// of j gains one. Correctly handles i,j adjacency and shared neighbours. The
// inverse of move i->j is move j->i, so reverting calls this with (j,i).
inline void apply_move_counts(std::vector<int> &m, int i, int j, const int *nn) {
  const int *ni = nn + fp::kNumNeighbors * i;
  const int *nj = nn + fp::kNumNeighbors * j;
  for (int k = 0; k < fp::kNumNeighbors; ++k)
    --m[ni[k]];
  for (int k = 0; k < fp::kNumNeighbors; ++k)
    ++m[nj[k]];
}

// Exact integer energy change of relocating the particle at occupied site i to
// vacant site j, given maintained counts m[] for the current configuration.
// O(1): touches only i, j and their neighbours (deduplicated). The moved
// particle keeps its label (lattice[i]).
inline int move_delta_energy(const std::vector<uint8_t> &lattice,
                             const std::vector<int> &m, int i, int j,
                             const int *nn) {
  constexpr int kCap = 2 * (fp::kNumNeighbors + 1);
  int region[kCap];
  int dm[kCap];
  int count = 0;
  auto index_of = [&](int s) -> int {
    for (int t = 0; t < count; ++t)
      if (region[t] == s)
        return t;
    region[count] = s;
    dm[count] = 0;
    return count++;
  };

  index_of(i);
  index_of(j);
  const int *ni = nn + fp::kNumNeighbors * i;
  const int *nj = nn + fp::kNumNeighbors * j;
  for (int k = 0; k < fp::kNumNeighbors; ++k)
    dm[index_of(ni[k])] -= 1; // neighbours of i lose occupied neighbour i
  for (int k = 0; k < fp::kNumNeighbors; ++k)
    dm[index_of(nj[k])] += 1; // neighbours of j gain occupied neighbour j

  const uint8_t moved_label = lattice[i];
  int delta = 0;
  for (int t = 0; t < count; ++t) {
    const int s = region[t];
    const uint8_t old_state = lattice[s];
    int e_old = 0;
    if (old_state != cluster::kEmpty) {
      const int d = m[s] - preferred_coordination(old_state);
      e_old = d * d;
    }
    const uint8_t new_state =
        s == i ? cluster::kEmpty : (s == j ? moved_label : old_state);
    int e_new = 0;
    if (new_state != cluster::kEmpty) {
      const int d = (m[s] + dm[t]) - preferred_coordination(new_state);
      e_new = d * d;
    }
    delta += e_new - e_old;
  }
  return delta;
}

// Energy change of removing the occupied particle at i: its own term vanishes
// and each occupied neighbour loses one occupied neighbour. O(1).
inline int remove_particle_delta(const std::vector<uint8_t> &lattice,
                                 const std::vector<int> &m, int i,
                                 const int *nn) {
  const int l = preferred_coordination(lattice[i]);
  int delta = -(m[i] - l) * (m[i] - l);
  const int *ni = nn + fp::kNumNeighbors * i;
  for (int k = 0; k < fp::kNumNeighbors; ++k) {
    const int n = ni[k];
    if (lattice[n] != cluster::kEmpty) {
      const int ln = preferred_coordination(lattice[n]);
      const int before = m[n] - ln;
      const int after = m[n] - 1 - ln;
      delta += after * after - before * before;
    }
  }
  return delta;
}

// Energy change of inserting a particle with the given label at vacant site v:
// its term appears and each occupied neighbour gains one occupied neighbour.
// O(1). For a relocation, dE(i->v) = remove_particle_delta(i) +
// insert_particle_delta(v) evaluated AFTER i has been removed.
inline int insert_particle_delta(const std::vector<uint8_t> &lattice,
                                 const std::vector<int> &m, int v, uint8_t label,
                                 const int *nn) {
  const int l = preferred_coordination(label);
  int delta = (m[v] - l) * (m[v] - l);
  const int *nv = nn + fp::kNumNeighbors * v;
  for (int k = 0; k < fp::kNumNeighbors; ++k) {
    const int n = nv[k];
    if (lattice[n] != cluster::kEmpty) {
      const int ln = preferred_coordination(lattice[n]);
      const int before = m[n] - ln;
      const int after = m[n] + 1 - ln;
      delta += after * after - before * before;
    }
  }
  return delta;
}

// Occupied/vacant partition with O(1) membership swaps. slot[site] is the index
// of `site` within whichever list (occ or vac) currently holds it.
struct SiteLists {
  std::vector<int> occ;
  std::vector<int> vac;
  std::vector<int> slot;
};

inline SiteLists build_site_lists(const std::vector<uint8_t> &lattice) {
  SiteLists lists;
  const int n = static_cast<int>(lattice.size());
  lists.slot.assign(n, -1);
  for (int site = 0; site < n; ++site) {
    if (lattice[site] == cluster::kEmpty) {
      lists.slot[site] = static_cast<int>(lists.vac.size());
      lists.vac.push_back(site);
    } else {
      lists.slot[site] = static_cast<int>(lists.occ.size());
      lists.occ.push_back(site);
    }
  }
  return lists;
}

// Move the occupied site `i` into the vacant site `j` in the partition: j joins
// occ in i's slot, i joins vac in j's slot. Must be paired with the matching
// lattice mutation and apply_move_counts.
inline void commit_move(SiteLists &lists, int i, int j) {
  const int si = lists.slot[i]; // index in occ
  const int sj = lists.slot[j]; // index in vac
  lists.occ[si] = j;
  lists.slot[j] = si;
  lists.vac[sj] = i;
  lists.slot[i] = sj;
}

// Reference (region-recompute) energy change of swapping a,b, leaving the
// lattice unchanged. Used to validate move_delta_energy in the test.
inline int swap_delta_energy(std::vector<uint8_t> &lattice, int a, int b,
                             const int *nn) {
  const int before = fp::swap_region_energy(lattice, a, b, nn);
  std::swap(lattice[a], lattice[b]);
  const int after = fp::swap_region_energy(lattice, a, b, nn);
  std::swap(lattice[a], lattice[b]);
  return after - before;
}

// Pure occupancy structural observable: number of occupied-occupied nearest
// neighbour bonds, sum_{occupied i} m_i / 2. Independent of species labels, so
// its equilibrium mean is identical for any composition-conserving sampler --
// a clean target for the stationarity gate of the occupancy channel.
inline long long occupied_bond_count(const std::vector<uint8_t> &lattice,
                                     const int *nn) {
  long long twice = 0;
  const int n = static_cast<int>(lattice.size());
  for (int site = 0; site < n; ++site) {
    if (lattice[site] == cluster::kEmpty)
      continue;
    const int *neigh = nn + fp::kNumNeighbors * site;
    for (int k = 0; k < fp::kNumNeighbors; ++k)
      twice += lattice[neigh[k]] != cluster::kEmpty;
  }
  return twice / 2;
}

// One sweep of `attempts` blind occupancy moves: uniform occupied source,
// uniform vacant destination, Metropolis on the exact region energy. Maintains
// `lists` and `m` in lockstep with the lattice.
inline SweepStats blind_occupancy_sweep(std::vector<uint8_t> &lattice,
                                        long double beta, int attempts,
                                        SiteLists &lists, std::vector<int> &m,
                                        std::mt19937 &gen, const int *nn) {
  SweepStats stats;
  const int np = static_cast<int>(lists.occ.size());
  const int nv = static_cast<int>(lists.vac.size());
  if (np == 0 || nv == 0)
    return stats;
  std::uniform_int_distribution<int> occ_dist(0, np - 1);
  std::uniform_int_distribution<int> vac_dist(0, nv - 1);
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  const double b = static_cast<double>(beta);

  for (int a = 0; a < attempts; ++a) {
    const int i = lists.occ[occ_dist(gen)];
    const int j = lists.vac[vac_dist(gen)];
    const int d = move_delta_energy(lattice, m, i, j, nn);
    const double accept = d <= 0 ? 1.0 : std::exp(-b * d);
    ++stats.attempts;
    stats.accept_prob_sum += accept;
    if (d <= 0 || uni(gen) < accept) {
      std::swap(lattice[i], lattice[j]);
      apply_move_counts(m, i, j, nn);
      commit_move(lists, i, j);
      ++stats.accepted;
    }
  }
  return stats;
}

// One sweep of `attempts` informed occupancy moves (see file header for the
// derivation). `weights` is reused scratch sized >= number of vacancies.
inline SweepStats informed_occupancy_sweep(std::vector<uint8_t> &lattice,
                                           long double beta, int attempts,
                                           SiteLists &lists, std::vector<int> &m,
                                           std::vector<double> &weights,
                                           std::mt19937 &gen, const int *nn) {
  SweepStats stats;
  const int np = static_cast<int>(lists.occ.size());
  const int nv = static_cast<int>(lists.vac.size());
  if (np == 0 || nv == 0)
    return stats;
  if (static_cast<int>(weights.size()) < nv)
    weights.assign(nv, 0.0);
  std::uniform_int_distribution<int> occ_dist(0, np - 1);
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  const double half_beta = 0.5 * static_cast<double>(beta);

  for (int a = 0; a < attempts; ++a) {
    const int i = lists.occ[occ_dist(gen)];

    const uint8_t label = lattice[i];

    // Removing i is shared by every destination: w_x(i,v) = w_remove * a_v with
    // a_v the insert weight in x' (i removed) and w_remove = exp(-beta/2 *
    // dE_remove). So sampling reduces to a_v / S and the reverse normalizer to a
    // closed form (see file header).
    const int dE_remove = remove_particle_delta(lattice, m, i, nn);
    const double w_remove = std::exp(-half_beta * dE_remove);

    // Mutate to x' (i removed).
    lattice[i] = cluster::kEmpty;
    const int *ni = nn + fp::kNumNeighbors * i;
    for (int k = 0; k < fp::kNumNeighbors; ++k)
      --m[ni[k]];

    // Insert weights a_v = exp(-beta/2 * dE_insert(v)) over original vacancies.
    double s_sum = 0.0;
    for (int t = 0; t < nv; ++t) {
      const int d = insert_particle_delta(lattice, m, lists.vac[t], label, nn);
      const double a = std::exp(-half_beta * d);
      weights[t] = a;
      s_sum += a;
    }

    // Sample destination j ~ a_v / S.
    double threshold = uni(gen) * s_sum;
    int tj = nv - 1;
    for (int t = 0; t < nv; ++t) {
      threshold -= weights[t];
      if (threshold <= 0.0) {
        tj = t;
        break;
      }
    }
    const int j = lists.vac[tj];
    const double a_j = weights[tj];

    // Z_i(x) = w_remove * S; Z_j(y) = (S - a_j + a_i)/a_j with a_i = 1/w_remove
    // (inserting at i in x' is the exact inverse of removal). alpha=min(1,Zi/Zj).
    double accept = 0.0;
    if (s_sum > 0.0 && a_j > 0.0) {
      const double a_i = 1.0 / w_remove;
      const double z_i = w_remove * s_sum;
      const double z_j = (s_sum - a_j + a_i) / a_j;
      accept = z_i >= z_j ? 1.0 : z_i / z_j;
    }
    ++stats.attempts;
    stats.accept_prob_sum += accept;

    if (accept >= 1.0 || (accept > 0.0 && uni(gen) < accept)) {
      // Commit: insert the particle at j (config y).
      lattice[j] = label;
      const int *nj = nn + fp::kNumNeighbors * j;
      for (int k = 0; k < fp::kNumNeighbors; ++k)
        ++m[nj[k]];
      commit_move(lists, i, j);
      ++stats.accepted;
    } else {
      // Reject: re-insert the particle at i (restore x).
      lattice[i] = label;
      for (int k = 0; k < fp::kNumNeighbors; ++k)
        ++m[ni[k]];
    }
  }
  return stats;
}

// Capped informed relocation via Multiple-Try Metropolis (Liu-Liang-Wong 2000)
// with the locally-balanced weight. Restores O(N) per sweep: each particle
// scores only `k` random candidate vacancies instead of all N_v, so cost is
// O(N_p * k) independent of system size. Exact (reversible w.r.t. the fixed-
// composition Gibbs measure) for any k.
//
// Procedure per move (see also the full-set derivation above):
//   - pick source i uniformly; remove it (shared dE_remove, mutate to x').
//   - forward: draw k vacancies v_1..v_k i.i.d. uniform from V; weight each by
//     a_c = exp(-beta/2 * insert_delta(v_c)); pick winner j ~ a_c. The true
//     MTM forward weight is w(x,y_c) = w_remove * a_c with w_remove common.
//   - reverse: the MTM reference set is the forced return move (j->i, weight
//     a_i = 1/w_remove) plus k-1 fresh vacancies drawn uniform from
//     V(y) = V\{j}u{i}. Using dE_y(j->v) = -insert_delta(j) + insert_delta(v),
//     every reverse weight is w(y,.) = (1/a_J) * a(.).
//   - accept min(1, sum_fwd / sum_rev) = min(1, w_remove * a_J * S_fwd / S_rev),
//     S_fwd = sum_c a_c, S_rev = (1/w_remove) + sum of k-1 fresh a(.).
// The base proposal is uniform (symmetric) so it cancels; the 1/N_p source
// factors cancel under fixed composition. k -> N_v recovers the full-set kernel.
// `cand_w`/`cand_v` are reused scratch of length >= k.
inline SweepStats informed_mtm_swap_sweep(
    std::vector<uint8_t> &lattice, long double beta, int attempts, int k,
    SiteLists &lists, std::vector<int> &m, std::vector<double> &cand_w,
    std::vector<int> &cand_v, std::mt19937 &gen, const int *nn) {
  SweepStats stats;
  const int np = static_cast<int>(lists.occ.size());
  int nv = static_cast<int>(lists.vac.size());
  if (np == 0 || nv == 0 || k <= 0)
    return stats;
  if (static_cast<int>(cand_w.size()) < k)
    cand_w.assign(k, 0.0);
  if (static_cast<int>(cand_v.size()) < k)
    cand_v.assign(k, 0);
  std::uniform_int_distribution<int> occ_dist(0, np - 1);
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  const double hb = 0.5 * static_cast<double>(beta);
  for (int a = 0; a < attempts; ++a) {
    const int i = lists.occ[occ_dist(gen)];
    const uint8_t label = lattice[i];
    const int dE_remove = remove_particle_delta(lattice, m, i, nn);
    const double w_remove = std::exp(-hb * dE_remove);

    // Mutate to x' (i removed).
    const int *ni = nn + fp::kNumNeighbors * i;
    lattice[i] = cluster::kEmpty;
    for (int t = 0; t < fp::kNumNeighbors; ++t)
      --m[ni[t]];
    std::uniform_int_distribution<int> vac_dist(0, nv - 1);

    // Forward: k candidate vacancies, locally-balanced insert weights.
    double s_fwd = 0.0;
    for (int c = 0; c < k; ++c) {
      const int v = lists.vac[vac_dist(gen)];
      const double aw =
          std::exp(-hb * insert_particle_delta(lattice, m, v, label, nn));
      cand_w[c] = aw;
      cand_v[c] = v;
      s_fwd += aw;
    }
    ++stats.attempts;
    if (!(s_fwd > 0.0)) { // degenerate underflow: reject, restore x
      lattice[i] = label;
      for (int t = 0; t < fp::kNumNeighbors; ++t)
        ++m[ni[t]];
      continue;
    }

    // Select winner j ~ cand_w / s_fwd.
    double threshold = uni(gen) * s_fwd;
    int win = k - 1;
    double acc = 0.0;
    for (int c = 0; c < k; ++c) {
      acc += cand_w[c];
      if (acc >= threshold) {
        win = c;
        break;
      }
    }
    const int j = cand_v[win];
    const double a_j = cand_w[win];

    // Reverse reference set: forced return (j->i, weight 1/w_remove) plus k-1
    // fresh vacancies uniform over V(y) = V with j replaced by i.
    double s_rev = 1.0 / w_remove;
    for (int c = 0; c < k - 1; ++c) {
      const int idx = vac_dist(gen);
      const int v = (lists.vac[idx] == j) ? i : lists.vac[idx];
      s_rev += std::exp(-hb * insert_particle_delta(lattice, m, v, label, nn));
    }

    const double num = w_remove * a_j * s_fwd;
    const double accept = num >= s_rev ? 1.0 : num / s_rev;
    stats.accept_prob_sum += accept;
    if (accept >= 1.0 || (accept > 0.0 && uni(gen) < accept)) {
      // Commit: insert the particle at j (config y).
      const int *nj = nn + fp::kNumNeighbors * j;
      lattice[j] = label;
      for (int t = 0; t < fp::kNumNeighbors; ++t)
        ++m[nj[t]];
      commit_move(lists, i, j);
      ++stats.accepted;
    } else {
      // Reject: re-insert the particle at i (restore x).
      lattice[i] = label;
      for (int t = 0; t < fp::kNumNeighbors; ++t)
        ++m[ni[t]];
    }
  }
  return stats;
}
} // namespace informed
} // namespace lattice_glass

#endif // LATTICE_GLASS_INFORMED_SWAP_H
