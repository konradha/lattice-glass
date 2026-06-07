#ifndef LATTICE_GLASS_FP_SAMPLER_H
#define LATTICE_GLASS_FP_SAMPLER_H

// Fast Franz-Parisi / reference-coupled sampler primitives shared by the
// efficiency harness and its tests.
//
// State is two same-temperature replicas (a cluster::ReplicaPair). Both may be
// symmetrically coupled to a common quenched reference s0 through the effective
// single-replica energy
//     E_eff(s) = H(s) - epsilon * (#sites where s agrees with s0).
// The coupling is symmetric under the replica exchange, so it cancels in the
// balanced cluster move's acceptance (proved separately, tested in
// tests/test_reference_cancellation.cpp) and only biases the base moves.

#include "balanced_cluster.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace lattice_glass {
namespace fp {

constexpr int kNumNeighbors = 6;

inline std::vector<int> cubic_neighbors(int L) {
  const int n = L * L * L;
  std::vector<int> nn(n * kNumNeighbors);
  for (int i = 0; i < L; ++i)
    for (int j = 0; j < L; ++j)
      for (int k = 0; k < L; ++k) {
        const int site = k + L * (j + i * L);
        nn[kNumNeighbors * site + 0] = k + L * (j + ((i + L - 1) % L) * L);
        nn[kNumNeighbors * site + 1] = k + L * (j + ((i + 1) % L) * L);
        nn[kNumNeighbors * site + 2] = k + L * (((j + L - 1) % L) + i * L);
        nn[kNumNeighbors * site + 3] = k + L * (((j + 1) % L) + i * L);
        nn[kNumNeighbors * site + 4] = ((k + L - 1) % L) + L * (j + i * L);
        nn[kNumNeighbors * site + 5] = ((k + 1) % L) + L * (j + i * L);
      }
  return nn;
}

inline int local_energy_int(const std::vector<uint8_t> &lattice, int site,
                            const int *nn) {
  const uint8_t type = lattice[site];
  if (type == cluster::kEmpty)
    return 0;
  const int preferred = type == cluster::kType1 ? 3 : 5;
  const int *neigh = nn + kNumNeighbors * site;
  int occupied = 0;
  for (int i = 0; i < kNumNeighbors; ++i)
    occupied += lattice[neigh[i]] != cluster::kEmpty;
  const int delta = occupied - preferred;
  return delta * delta;
}

inline long double total_energy_int(const std::vector<uint8_t> &lattice,
                                    const int *nn) {
  long double energy = 0.0L;
  for (int site = 0; site < static_cast<int>(lattice.size()); ++site)
    energy += local_energy_int(lattice, site, nn);
  return energy;
}

// Energy of the two swapped sites plus their (deduplicated) neighbourhoods.
inline int swap_region_energy(const std::vector<uint8_t> &lattice, int a, int b,
                              const int *nn) {
  int affected[2 * (kNumNeighbors + 1)];
  int count = 0;
  auto push = [&](int site) {
    for (int i = 0; i < count; ++i)
      if (affected[i] == site)
        return;
    affected[count++] = site;
  };
  push(a);
  push(b);
  for (int i = 0; i < kNumNeighbors; ++i) {
    push(nn[kNumNeighbors * a + i]);
    push(nn[kNumNeighbors * b + i]);
  }
  int energy = 0;
  for (int i = 0; i < count; ++i)
    energy += local_energy_int(lattice, affected[i], nn);
  return energy;
}

inline int agreement_count(const std::vector<uint8_t> &a,
                           const std::vector<uint8_t> &b) {
  int count = 0;
  for (int site = 0; site < static_cast<int>(a.size()); ++site)
    count += a[site] == b[site];
  return count;
}

inline int disagreement_count(const std::vector<uint8_t> &a,
                              const std::vector<uint8_t> &b) {
  return static_cast<int>(a.size()) - agreement_count(a, b);
}

// One sweep (lat_size attempts) of random-pair exchanges, Metropolis on the
// field-tilted effective energy. reference == nullptr disables the field.
inline int nonlocal_swap_sweep(std::vector<uint8_t> &lattice, long double beta,
                               long double epsilon,
                               const std::vector<uint8_t> *reference,
                               std::mt19937 &gen, const int *nn) {
  const int lat_size = static_cast<int>(lattice.size());
  std::uniform_int_distribution<int> site_dist(0, lat_size - 1);
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  int accepted = 0;
  for (int step = 0; step < lat_size; ++step) {
    const int a = site_dist(gen);
    const int b = site_dist(gen);
    if (lattice[a] == lattice[b])
      continue;

    const int before = swap_region_energy(lattice, a, b, nn);
    int field_delta = 0;
    if (reference != nullptr) {
      const uint8_t ra = (*reference)[a];
      const uint8_t rb = (*reference)[b];
      const int agree_before = (lattice[a] == ra) + (lattice[b] == rb);
      const int agree_after = (lattice[b] == ra) + (lattice[a] == rb);
      field_delta = agree_after - agree_before;
    }
    std::swap(lattice[a], lattice[b]);
    const int after = swap_region_energy(lattice, a, b, nn);

    const long double delta = static_cast<long double>(after - before) -
                              epsilon * static_cast<long double>(field_delta);
    if (delta <= 0.0L ||
        uni(gen) < std::exp(-static_cast<double>(beta * delta))) {
      ++accepted;
    } else {
      std::swap(lattice[a], lattice[b]);
    }
  }
  return accepted;
}

inline void equilibrate(std::vector<uint8_t> &lattice, long double beta,
                        long double epsilon,
                        const std::vector<uint8_t> *reference, int sweeps,
                        std::mt19937 &gen, const int *nn) {
  for (int sweep = 0; sweep < sweeps; ++sweep)
    nonlocal_swap_sweep(lattice, beta, epsilon, reference, gen, nn);
}

inline std::vector<uint8_t> random_lattice(int num_type1, int num_type2,
                                           int lat_size, std::mt19937 &gen) {
  std::vector<uint8_t> lattice;
  lattice.reserve(lat_size);
  lattice.insert(lattice.end(), num_type1, cluster::kType1);
  lattice.insert(lattice.end(), num_type2, cluster::kType2);
  lattice.insert(lattice.end(), lat_size - num_type1 - num_type2,
                 cluster::kEmpty);
  std::shuffle(lattice.begin(), lattice.end(), gen);
  return lattice;
}

struct AutocorrResult {
  long double tau_int = 0.5L;
  long double mean = 0.0L;
  long double variance = 0.0L;
  int window = 0;
  bool resolved = false; // chain long enough relative to tau
};

// Integrated autocorrelation time with Madras-Sokal automatic windowing:
// the window M is the smallest lag with M >= c * tau_int(M).
inline AutocorrResult integrated_autocorrelation_time(
    const std::vector<long double> &samples, long double c = 6.0L) {
  AutocorrResult result;
  const int n = static_cast<int>(samples.size());
  if (n < 2)
    return result;

  long double mean = 0.0L;
  for (long double s : samples)
    mean += s;
  mean /= n;
  result.mean = mean;

  long double variance = 0.0L;
  for (long double s : samples) {
    const long double d = s - mean;
    variance += d * d;
  }
  variance /= n;
  result.variance = variance;
  if (variance == 0.0L) {
    result.resolved = true;
    return result;
  }

  long double tau = 0.5L;
  int window = n - 1;
  for (int lag = 1; lag < n; ++lag) {
    long double cov = 0.0L;
    for (int i = 0; i + lag < n; ++i)
      cov += (samples[i] - mean) * (samples[i + lag] - mean);
    cov /= (n - lag);
    tau += cov / variance;
    if (static_cast<long double>(lag) >= c * tau) {
      window = lag;
      break;
    }
  }
  if (tau < 0.5L)
    tau = 0.5L;
  result.tau_int = tau;
  result.window = window;
  // Madras-Sokal: a windowed estimate needs n >> tau (rule of thumb n > 50 tau).
  result.resolved = static_cast<long double>(n) > 50.0L * tau;
  return result;
}

inline long double effective_samples(int n, long double tau_int) {
  return static_cast<long double>(n) / (2.0L * tau_int);
}

} // namespace fp
} // namespace lattice_glass

#endif // LATTICE_GLASS_FP_SAMPLER_H
