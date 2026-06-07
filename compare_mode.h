#ifndef LATTICE_GLASS_COMPARE_MODE_H
#define LATTICE_GLASS_COMPARE_MODE_H

#include "balanced_cluster.h"
#include "species_reduction.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace lattice_glass {
namespace compare {

enum class KernelKind { Base, HeatBath, Cluster };
enum class ClusterKind { Full, ZeroOne, ZeroTwo };

struct Options {
  long double beta = 3.5L;
  long double rho = 0.75L;
  long double rho1 = 0.30L;
  int sweeps = 200000;
  int warmup = 50000;
  int sample_every = 10;
  int heatbath_every = 1;
  int cluster_every = 1;
  int cluster_max_size = 32;
  long double cluster_kappa = 0.5L;
  unsigned int seed = 20260606u;
  KernelKind kernel = KernelKind::Base;
  ClusterKind cluster_kind = ClusterKind::Full;
};

struct SweepStats {
  int attempts = 0;
  int accepted = 0;
};

struct ClusterStats {
  int attempts = 0;
  int proposed = 0;
  int accepted = 0;
  long double accepted_size_sum = 0.0L;
  long double accepted_delta_sum = 0.0L;
};

inline int parse_int(const char *value) { return std::stoi(std::string(value)); }

inline unsigned int parse_uint(const char *value) {
  return static_cast<unsigned int>(std::stoul(std::string(value)));
}

inline long double parse_long_double(const char *value) {
  return std::stold(std::string(value));
}

inline KernelKind parse_kernel_kind(const std::string &value) {
  if (value == "base")
    return KernelKind::Base;
  if (value == "heatbath")
    return KernelKind::HeatBath;
  if (value == "cluster")
    return KernelKind::Cluster;
  throw std::invalid_argument("unknown kernel kind " + value);
}

inline ClusterKind parse_cluster_kind(const std::string &value) {
  if (value == "full")
    return ClusterKind::Full;
  if (value == "01")
    return ClusterKind::ZeroOne;
  if (value == "02")
    return ClusterKind::ZeroTwo;
  throw std::invalid_argument("unknown cluster kind " + value);
}

inline const char *kernel_name(const KernelKind kernel) {
  switch (kernel) {
  case KernelKind::Base:
    return "base";
  case KernelKind::HeatBath:
    return "heatbath";
  case KernelKind::Cluster:
    return "cluster";
  }
  return "unknown";
}

inline const char *cluster_name(const ClusterKind kind) {
  switch (kind) {
  case ClusterKind::Full:
    return "full";
  case ClusterKind::ZeroOne:
    return "01";
  case ClusterKind::ZeroTwo:
    return "02";
  }
  return "unknown";
}

inline Options parse_options(const int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string key(argv[i]);
    if (key == "--mode") {
      if (i + 1 >= argc)
        throw std::invalid_argument("missing value for --mode");
      const std::string mode(argv[++i]);
      if (mode != "compare")
        throw std::invalid_argument("unsupported mode " + mode);
      continue;
    }

    if (i + 1 >= argc)
      throw std::invalid_argument("missing value for " + key);
    const char *value = argv[++i];

    if (key == "--beta")
      options.beta = parse_long_double(value);
    else if (key == "--rho")
      options.rho = parse_long_double(value);
    else if (key == "--rho1")
      options.rho1 = parse_long_double(value);
    else if (key == "--sweeps")
      options.sweeps = parse_int(value);
    else if (key == "--warmup")
      options.warmup = parse_int(value);
    else if (key == "--sample-every")
      options.sample_every = parse_int(value);
    else if (key == "--heatbath-every")
      options.heatbath_every = parse_int(value);
    else if (key == "--cluster-every")
      options.cluster_every = parse_int(value);
    else if (key == "--cluster-max-size")
      options.cluster_max_size = parse_int(value);
    else if (key == "--cluster-kappa")
      options.cluster_kappa = parse_long_double(value);
    else if (key == "--seed")
      options.seed = parse_uint(value);
    else if (key == "--kernel")
      options.kernel = parse_kernel_kind(value);
    else if (key == "--cluster-kind")
      options.cluster_kind = parse_cluster_kind(value);
    else
      throw std::invalid_argument("unknown option " + key);
  }

  if (options.beta < 0.0L)
    throw std::invalid_argument("beta must be non-negative");
  if (options.rho < 0.0L || options.rho > 1.0L)
    throw std::invalid_argument("rho must be in [0, 1]");
  if (options.rho1 < 0.0L || options.rho1 > options.rho)
    throw std::invalid_argument("rho1 must be in [0, rho]");
  if (options.sweeps <= 0 || options.warmup < 0 || options.sample_every <= 0)
    throw std::invalid_argument("sweeps, warmup, sample cadence must be valid");
  if (options.heatbath_every <= 0 || options.cluster_every <= 0 ||
      options.cluster_max_size <= 0)
    throw std::invalid_argument("kernel cadences and cluster size must be positive");
  if (options.warmup >= options.sweeps)
    throw std::invalid_argument("warmup must be smaller than sweeps");
  return options;
}

inline std::vector<uint8_t> random_lattice(const int num_type1, const int num_type2,
                                           std::mt19937 &generator) {
  std::vector<uint8_t> lattice;
  lattice.reserve(lat_size);
  lattice.insert(lattice.end(), num_type1, species::kType1);
  lattice.insert(lattice.end(), num_type2, species::kType2);
  lattice.insert(lattice.end(), lat_size - num_type1 - num_type2,
                 species::kEmpty);
  std::shuffle(lattice.begin(), lattice.end(), generator);
  return lattice;
}

inline long double replica_energy(const std::vector<uint8_t> &replica,
                                  const int *nearest_neighbors) {
  return cluster::total_energy(replica, nearest_neighbors, NUM_NN);
}

inline bool metropolis_neighbor_exchange(std::vector<uint8_t> &replica,
                                         const int site, const int mv,
                                         const long double beta,
                                         std::mt19937 &generator,
                                         const int *nearest_neighbors) {
  if (replica[site] == replica[mv])
    return false;

  const std::vector<int> moved = {site, mv};
  const std::vector<int> affected =
      cluster::cluster_affected_sites(moved, nearest_neighbors, NUM_NN);
  const long double before =
      cluster::affected_energy(replica, affected, nearest_neighbors, NUM_NN);
  std::swap(replica[site], replica[mv]);
  const long double after =
      cluster::affected_energy(replica, affected, nearest_neighbors, NUM_NN);
  const long double delta = after - before;

  std::uniform_real_distribution<long double> uniform(0.0L, 1.0L);
  if (delta <= 0.0L || uniform(generator) < std::expl(-beta * delta))
    return true;

  std::swap(replica[site], replica[mv]);
  return false;
}

inline SweepStats local_neighbor_sweep(std::vector<uint8_t> &replica,
                                       const long double beta,
                                       std::mt19937 &generator,
                                       const int *nearest_neighbors) {
  SweepStats stats;
  std::uniform_int_distribution<int> site_dist(0, lat_size - 1);
  std::uniform_int_distribution<int> nb_dist(0, NUM_NN - 1);
  for (int step = 0; step < lat_size; ++step) {
    const int site = site_dist(generator);
    const int mv = nearest_neighbors[NUM_NN * site + nb_dist(generator)];
    stats.attempts++;
    stats.accepted +=
        metropolis_neighbor_exchange(replica, site, mv, beta, generator,
                                     nearest_neighbors);
  }
  return stats;
}

inline void apply_species_heatbath(std::vector<uint8_t> &replica,
                                   const long double beta, const int num_type1,
                                   std::mt19937 &generator,
                                   const int *nearest_neighbors) {
  replica = species::sample_species_given_occupancy(replica, nearest_neighbors,
                                                    NUM_NN, beta, num_type1,
                                                    generator);
}

inline long double rao_blackwell_overlap(const std::vector<uint8_t> &first,
                                         const std::vector<uint8_t> &second,
                                         const long double beta,
                                         const int num_type1,
                                         const int *nearest_neighbors) {
  const auto first_stats =
      species::compute_occupancy_stats(first, nearest_neighbors, NUM_NN);
  const auto second_stats =
      species::compute_occupancy_stats(second, nearest_neighbors, NUM_NN);
  const auto first_probs =
      species::site_type1_marginals(first_stats, beta, num_type1);
  const auto second_probs =
      species::site_type1_marginals(second_stats, beta, num_type1);
  return species::rao_blackwell_overlap_sum(first, first_probs, second,
                                            second_probs) /
         static_cast<long double>(lat_size);
}

inline long double mean(const std::vector<long double> &values) {
  long double total = 0.0L;
  for (const long double value : values)
    total += value;
  return total / static_cast<long double>(values.size());
}

inline long double integrated_autocorrelation_time(
    const std::vector<long double> &samples) {
  if (samples.size() < 2)
    return 0.5L;

  const long double sample_mean = mean(samples);
  long double variance = 0.0L;
  for (const long double sample : samples) {
    const long double delta = sample - sample_mean;
    variance += delta * delta;
  }
  variance /= static_cast<long double>(samples.size());
  if (variance == 0.0L)
    return 0.5L;

  long double tau = 0.5L;
  const int max_lag = static_cast<int>(samples.size()) - 1;
  for (int lag = 1; lag <= max_lag; ++lag) {
    long double covariance = 0.0L;
    for (int i = 0; i + lag < static_cast<int>(samples.size()); ++i)
      covariance += (samples[i] - sample_mean) *
                    (samples[i + lag] - sample_mean);
    covariance /= static_cast<long double>(samples.size() - lag);
    const long double rho = covariance / variance;
    if (rho <= 0.0L && lag > 5)
      break;
    tau += rho;
    if (static_cast<long double>(lag) > 5.0L * tau)
      break;
  }
  return tau;
}

inline cluster::GrowthRule make_growth_rule(const Options &options) {
  cluster::GrowthRule rule;
  rule.kappa = options.cluster_kappa;
  switch (options.cluster_kind) {
  case ClusterKind::Full:
    break;
  case ClusterKind::ZeroOne:
    rule.restrict_unordered_type = true;
    rule.restricted_a = cluster::kEmpty;
    rule.restricted_b = cluster::kType1;
    break;
  case ClusterKind::ZeroTwo:
    rule.restrict_unordered_type = true;
    rule.restricted_a = cluster::kEmpty;
    rule.restricted_b = cluster::kType2;
    break;
  }
  return rule;
}

inline int run_compare_mode(const int argc, char **argv) {
  const Options options = parse_options(argc, argv);
  const int num_particles = static_cast<int>(options.rho * lat_size);
  const int num_type1 = static_cast<int>(options.rho1 * lat_size);
  const int num_type2 = num_particles - num_type1;
  if (num_type1 < 0 || num_type2 < 0 || num_particles > lat_size)
    throw std::invalid_argument("invalid compare composition");

  generate_tables();
  const int *nearest_neighbors = thread_nn[0];

  std::mt19937 generator1(options.seed);
  std::mt19937 generator2(options.seed ^ 0x9E3779B9u);
  std::mt19937 generator_species(options.seed ^ 0x85EBCA6Bu);
  std::mt19937 generator_cluster(options.seed ^ 0xC2B2AE35u);

  cluster::ReplicaPair pair;
  pair.first = random_lattice(num_type1, num_type2, generator1);
  pair.second = random_lattice(num_type1, num_type2, generator2);

  SweepStats sweep_first{};
  SweepStats sweep_second{};
  int heatbath_applications = 0;
  ClusterStats cluster_stats{};
  std::vector<long double> energy_samples;
  std::vector<long double> overlap_samples;
  energy_samples.reserve((options.sweeps + options.sample_every - 1) /
                         options.sample_every);
  overlap_samples.reserve(energy_samples.capacity());

  const cluster::GrowthRule cluster_rule = make_growth_rule(options);

  for (int sweep = 1; sweep <= options.sweeps; ++sweep) {
    const SweepStats first = local_neighbor_sweep(pair.first, options.beta,
                                                  generator1, nearest_neighbors);
    const SweepStats second = local_neighbor_sweep(pair.second, options.beta,
                                                   generator2, nearest_neighbors);
    sweep_first.attempts += first.attempts;
    sweep_first.accepted += first.accepted;
    sweep_second.attempts += second.attempts;
    sweep_second.accepted += second.accepted;

    if (options.kernel != KernelKind::Base &&
        sweep % options.heatbath_every == 0) {
      apply_species_heatbath(pair.first, options.beta, num_type1,
                             generator_species, nearest_neighbors);
      apply_species_heatbath(pair.second, options.beta, num_type1,
                             generator_species, nearest_neighbors);
      heatbath_applications += 2;
    }

    if (options.kernel == KernelKind::Cluster &&
        sweep % options.cluster_every == 0) {
      cluster_stats.attempts++;
      cluster::MetropolisResult result = cluster::metropolis_cluster_exchange(
          pair, options.beta, nearest_neighbors, NUM_NN,
          options.cluster_max_size, cluster_rule, generator_cluster);
      cluster_stats.proposed += result.proposed;
      cluster_stats.accepted += result.accepted;
      if (result.accepted) {
        cluster_stats.accepted_size_sum += result.cluster_size;
        cluster_stats.accepted_delta_sum += result.delta_energy;
      }
    }

    if (sweep <= options.warmup || sweep % options.sample_every != 0)
      continue;

    const long double energy =
        (replica_energy(pair.first, nearest_neighbors) +
         replica_energy(pair.second, nearest_neighbors)) /
        2.0L;
    const long double overlap = rao_blackwell_overlap(
        pair.first, pair.second, options.beta, num_type1, nearest_neighbors);
    energy_samples.push_back(energy);
    overlap_samples.push_back(overlap);
  }

  if (energy_samples.empty())
    throw std::runtime_error("no samples collected; adjust warmup/sample_every");

  const long double energy_tau = integrated_autocorrelation_time(energy_samples);
  const long double overlap_tau =
      integrated_autocorrelation_time(overlap_samples);
  const long double sample_count = static_cast<long double>(energy_samples.size());

  std::cout << std::setprecision(10);
  std::cout << "compare.L " << L << "\n";
  std::cout << "compare.kernel " << kernel_name(options.kernel) << "\n";
  std::cout << "compare.cluster_kind " << cluster_name(options.cluster_kind)
            << "\n";
  std::cout << "compare.beta " << static_cast<double>(options.beta) << "\n";
  std::cout << "compare.rho " << static_cast<double>(options.rho) << "\n";
  std::cout << "compare.rho1 " << static_cast<double>(options.rho1) << "\n";
  std::cout << "compare.sweeps " << options.sweeps << "\n";
  std::cout << "compare.warmup " << options.warmup << "\n";
  std::cout << "compare.sample_every " << options.sample_every << "\n";
  std::cout << "compare.heatbath_every " << options.heatbath_every << "\n";
  std::cout << "compare.cluster_every " << options.cluster_every << "\n";
  std::cout << "compare.cluster_max_size " << options.cluster_max_size << "\n";
  std::cout << "compare.cluster_kappa "
            << static_cast<double>(options.cluster_kappa) << "\n";
  std::cout << "compare.local_acceptance_first "
            << (static_cast<long double>(sweep_first.accepted) /
                static_cast<long double>(sweep_first.attempts))
            << "\n";
  std::cout << "compare.local_acceptance_second "
            << (static_cast<long double>(sweep_second.accepted) /
                static_cast<long double>(sweep_second.attempts))
            << "\n";
  std::cout << "compare.heatbath_applications " << heatbath_applications
            << "\n";
  std::cout << "compare.cluster_attempts " << cluster_stats.attempts << "\n";
  std::cout << "compare.cluster_proposed " << cluster_stats.proposed << "\n";
  std::cout << "compare.cluster_accepted " << cluster_stats.accepted << "\n";
  std::cout << "compare.cluster_accept_per_attempt "
            << (cluster_stats.attempts > 0
                    ? static_cast<long double>(cluster_stats.accepted) /
                          static_cast<long double>(cluster_stats.attempts)
                    : 0.0L)
            << "\n";
  std::cout << "compare.cluster_mean_accepted_size "
            << (cluster_stats.accepted > 0
                    ? cluster_stats.accepted_size_sum /
                          static_cast<long double>(cluster_stats.accepted)
                    : 0.0L)
            << "\n";
  std::cout << "compare.cluster_mean_accepted_delta "
            << (cluster_stats.accepted > 0
                    ? cluster_stats.accepted_delta_sum /
                          static_cast<long double>(cluster_stats.accepted)
                    : 0.0L)
            << "\n";
  std::cout << "compare.samples " << energy_samples.size() << "\n";
  std::cout << "energy.mean " << mean(energy_samples) << "\n";
  std::cout << "energy.tau_int " << energy_tau << "\n";
  std::cout << "energy.effective_samples " << sample_count / (2.0L * energy_tau)
            << "\n";
  std::cout << "rb_overlap.mean " << mean(overlap_samples) << "\n";
  std::cout << "rb_overlap.tau_int " << overlap_tau << "\n";
  std::cout << "rb_overlap.effective_samples "
            << sample_count / (2.0L * overlap_tau) << "\n";
  return 0;
}

} // namespace compare
} // namespace lattice_glass

#endif // LATTICE_GLASS_COMPARE_MODE_H
