#include "../balanced_cluster.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace cluster = lattice_glass::cluster;

struct Options {
  int L = 8;
  int steps = 10000;
  int max_cluster = 128;
  unsigned int seed = 20260606;
  long double beta = 5.0L;
  long double kappa = 0.0L;
  long double rho = 0.75L;
  long double rho1 = 0.30L;
};

struct BucketStats {
  int proposed = 0;
  int accepted = 0;
  long double delta_sum = 0.0L;
  long double boundary_sum = 0.0L;
};

struct PilotStats {
  int attempts = 0;
  int proposed = 0;
  int abandoned = 0;
  int accepted = 0;
  int rejected = 0;
  int initial_disagreements = 0;
  int final_disagreements = 0;
  long double proposed_size_sum = 0.0L;
  long double accepted_size_sum = 0.0L;
  long double proposed_boundary_sum = 0.0L;
  long double accepted_boundary_sum = 0.0L;
  long double proposed_delta_sum = 0.0L;
  long double accepted_delta_sum = 0.0L;
  std::vector<BucketStats> size_buckets;
};

static constexpr int kMinPilotL = 8;
static constexpr int kNumSizeBuckets = 8;

static int parse_int(const char *value) { return std::stoi(std::string(value)); }

static unsigned int parse_uint(const char *value) {
  return static_cast<unsigned int>(std::stoul(std::string(value)));
}

static long double parse_long_double(const char *value) {
  return std::stold(std::string(value));
}

static Options parse_options(const int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string key(argv[i]);
    if (i + 1 >= argc)
      throw std::invalid_argument("missing value for " + key);
    const char *value = argv[++i];
    if (key == "--L")
      options.L = parse_int(value);
    else if (key == "--steps")
      options.steps = parse_int(value);
    else if (key == "--max-cluster")
      options.max_cluster = parse_int(value);
    else if (key == "--seed")
      options.seed = parse_uint(value);
    else if (key == "--beta")
      options.beta = parse_long_double(value);
    else if (key == "--kappa")
      options.kappa = parse_long_double(value);
    else if (key == "--rho")
      options.rho = parse_long_double(value);
    else if (key == "--rho1")
      options.rho1 = parse_long_double(value);
    else
      throw std::invalid_argument("unknown option " + key);
  }

  if (options.L < kMinPilotL)
    throw std::invalid_argument("pilot L must be at least 8");
  if (options.steps <= 0)
    throw std::invalid_argument("steps must be positive");
  if (options.max_cluster <= 0)
    throw std::invalid_argument("max cluster must be positive");
  if (options.beta < 0.0L)
    throw std::invalid_argument("beta must be non-negative");
  if (options.rho < 0.0L || options.rho > 1.0L)
    throw std::invalid_argument("rho must be in [0, 1]");
  if (options.rho1 < 0.0L || options.rho1 > options.rho)
    throw std::invalid_argument("rho1 must be in [0, rho]");
  return options;
}

static std::vector<int> cubic_neighbors(const int L) {
  const int lat_size = L * L * L;
  std::vector<int> neighbors(lat_size * 6);
  for (int i = 0; i < L; ++i) {
    for (int j = 0; j < L; ++j) {
      for (int k = 0; k < L; ++k) {
        const int site = k + L * (j + i * L);
        const int im = (i + L - 1) % L;
        const int ip = (i + 1) % L;
        const int jm = (j + L - 1) % L;
        const int jp = (j + 1) % L;
        const int km = (k + L - 1) % L;
        const int kp = (k + 1) % L;
        neighbors[6 * site + 0] = k + L * (j + im * L);
        neighbors[6 * site + 1] = k + L * (j + ip * L);
        neighbors[6 * site + 2] = k + L * (jm + i * L);
        neighbors[6 * site + 3] = k + L * (jp + i * L);
        neighbors[6 * site + 4] = km + L * (j + i * L);
        neighbors[6 * site + 5] = kp + L * (j + i * L);
      }
    }
  }
  return neighbors;
}

static std::vector<uint8_t> random_lattice(const Options &options,
                                           std::mt19937 &generator) {
  const int lat_size = options.L * options.L * options.L;
  const int occupied = static_cast<int>(options.rho * lat_size);
  const int type1 = static_cast<int>(options.rho1 * lat_size);
  const int type2 = occupied - type1;
  if (type1 < 0 || type2 < 0 || occupied > lat_size)
    throw std::invalid_argument("invalid composition");

  std::vector<uint8_t> lattice;
  lattice.reserve(lat_size);
  lattice.insert(lattice.end(), type1, cluster::kType1);
  lattice.insert(lattice.end(), type2, cluster::kType2);
  lattice.insert(lattice.end(), lat_size - occupied, cluster::kEmpty);
  std::shuffle(lattice.begin(), lattice.end(), generator);
  return lattice;
}

static int disagreement_count(const cluster::ReplicaPair &pair) {
  int count = 0;
  for (int site = 0; site < static_cast<int>(pair.first.size()); ++site)
    count += pair.first[site] != pair.second[site];
  return count;
}

static int size_bucket_index(const int size) {
  if (size <= 2)
    return 0;
  if (size <= 4)
    return 1;
  if (size <= 8)
    return 2;
  if (size <= 16)
    return 3;
  if (size <= 32)
    return 4;
  if (size <= 64)
    return 5;
  if (size <= 128)
    return 6;
  return 7;
}

static const char *size_bucket_name(const int bucket) {
  static const char *names[kNumSizeBuckets] = {"le2",   "3_4",   "5_8",
                                               "9_16", "17_32", "33_64",
                                               "65_128", "gt128"};
  return names[bucket];
}

static int boundary_size(const std::vector<int> &cluster_sites,
                         const std::vector<int> &nearest_neighbors) {
  const std::vector<int> affected =
      cluster::cluster_affected_sites(cluster_sites, nearest_neighbors.data(), 6);
  return static_cast<int>(affected.size() - cluster_sites.size());
}

static void add_result(PilotStats &stats, const cluster::GrowthAttempt &attempt,
                       const bool accepted, const long double delta_energy,
                       const int boundary) {
  stats.attempts++;
  if (attempt.abandoned) {
    stats.abandoned++;
    return;
  }
  if (!attempt.proposed)
    return;

  const int cluster_size = static_cast<int>(attempt.cluster_sites.size());
  const int bucket_index = size_bucket_index(cluster_size);
  BucketStats &bucket = stats.size_buckets[bucket_index];

  stats.proposed++;
  stats.proposed_size_sum += cluster_size;
  stats.proposed_boundary_sum += boundary;
  stats.proposed_delta_sum += delta_energy;
  bucket.proposed++;
  bucket.delta_sum += delta_energy;
  bucket.boundary_sum += boundary;

  if (accepted) {
    stats.accepted++;
    stats.accepted_size_sum += cluster_size;
    stats.accepted_boundary_sum += boundary;
    stats.accepted_delta_sum += delta_energy;
    bucket.accepted++;
  } else {
    stats.rejected++;
  }
}

static PilotStats run_case(cluster::ReplicaPair pair,
                           const std::vector<int> &nearest_neighbors,
                           const Options &options,
                           const cluster::GrowthRule &rule,
                           std::mt19937 &generator) {
  const auto first_counts = cluster::composition_counts(pair.first);
  const auto second_counts = cluster::composition_counts(pair.second);
  PilotStats stats;
  stats.size_buckets.assign(kNumSizeBuckets, BucketStats{});
  stats.initial_disagreements = disagreement_count(pair);
  std::uniform_real_distribution<long double> uniform(0.0L, 1.0L);

  for (int step = 0; step < options.steps; ++step) {
    const auto attempt = cluster::grow_balanced_cluster(
        pair, nearest_neighbors.data(), 6, options.max_cluster, rule,
        generator);
    long double delta_energy = 0.0L;
    int boundary = 0;
    bool accepted = false;
    if (attempt.proposed) {
      boundary = boundary_size(attempt.cluster_sites, nearest_neighbors);
      delta_energy = cluster::cluster_exchange_delta(
          pair, attempt.cluster_sites, nearest_neighbors.data(), 6);
      if (delta_energy <= 0.0L ||
          uniform(generator) < std::expl(-options.beta * delta_energy)) {
        cluster::exchange_cluster(pair, attempt.cluster_sites);
        accepted = true;
      }
    }

    add_result(stats, attempt, accepted, delta_energy, boundary);
    if (cluster::composition_counts(pair.first) != first_counts ||
        cluster::composition_counts(pair.second) != second_counts)
      throw std::runtime_error("cluster move changed replica composition");
  }

  stats.final_disagreements = disagreement_count(pair);
  return stats;
}

static void print_stats(const char *name, const PilotStats &stats) {
  const long double attempts = stats.attempts;
  const long double proposed = stats.proposed;
  const long double accepted = stats.accepted;

  std::cout << name << ".attempts " << stats.attempts << "\n";
  std::cout << name << ".initial_disagreements " << stats.initial_disagreements
            << "\n";
  std::cout << name << ".final_disagreements " << stats.final_disagreements
            << "\n";
  std::cout << name << ".closure_rate " << (proposed / attempts) << "\n";
  std::cout << name << ".abandon_rate " << (stats.abandoned / attempts) << "\n";
  std::cout << name << ".rejection_rate "
            << (proposed > 0.0L ? stats.rejected / proposed : 0.0L) << "\n";
  std::cout << name << ".acceptance_rate "
            << (proposed > 0.0L ? accepted / proposed : 0.0L) << "\n";
  std::cout << name << ".mean_proposed_size "
            << (proposed > 0.0L ? stats.proposed_size_sum / proposed : 0.0L)
            << "\n";
  std::cout << name << ".mean_accepted_size "
            << (accepted > 0.0L ? stats.accepted_size_sum / accepted : 0.0L)
            << "\n";
  std::cout << name << ".mean_proposed_boundary "
            << (proposed > 0.0L ? stats.proposed_boundary_sum / proposed
                                 : 0.0L)
            << "\n";
  std::cout << name << ".mean_accepted_boundary "
            << (accepted > 0.0L ? stats.accepted_boundary_sum / accepted
                                 : 0.0L)
            << "\n";
  std::cout << name << ".mean_proposed_delta "
            << (proposed > 0.0L ? stats.proposed_delta_sum / proposed : 0.0L)
            << "\n";
  std::cout << name << ".mean_accepted_delta "
            << (accepted > 0.0L ? stats.accepted_delta_sum / accepted : 0.0L)
            << "\n";

  for (int bucket = 0; bucket < kNumSizeBuckets; ++bucket) {
    const BucketStats &bucket_stats = stats.size_buckets[bucket];
    const long double bucket_proposed = bucket_stats.proposed;
    const long double bucket_accepted = bucket_stats.accepted;
    const std::string prefix = std::string(name) + ".size_" +
                               size_bucket_name(bucket);
    std::cout << prefix << ".proposed " << bucket_stats.proposed << "\n";
    std::cout << prefix << ".accepted " << bucket_stats.accepted << "\n";
    std::cout << prefix << ".acceptance_rate "
              << (bucket_proposed > 0.0L ? bucket_accepted / bucket_proposed
                                          : 0.0L)
              << "\n";
    std::cout << prefix << ".mean_delta "
              << (bucket_proposed > 0.0L
                      ? bucket_stats.delta_sum / bucket_proposed
                      : 0.0L)
              << "\n";
    std::cout << prefix << ".mean_boundary "
              << (bucket_proposed > 0.0L
                      ? bucket_stats.boundary_sum / bucket_proposed
                      : 0.0L)
              << "\n";
  }
}

int main(const int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    std::mt19937 generator(options.seed);
    const std::vector<int> nearest_neighbors = cubic_neighbors(options.L);

    cluster::ReplicaPair pair;
    pair.first = random_lattice(options, generator);
    pair.second = random_lattice(options, generator);

    cluster::GrowthRule full_rule;
    full_rule.kappa = options.kappa;

    cluster::GrowthRule zero_one_rule = full_rule;
    zero_one_rule.restrict_unordered_type = true;
    zero_one_rule.restricted_a = cluster::kEmpty;
    zero_one_rule.restricted_b = cluster::kType1;

    cluster::GrowthRule zero_two_rule = full_rule;
    zero_two_rule.restrict_unordered_type = true;
    zero_two_rule.restricted_a = cluster::kEmpty;
    zero_two_rule.restricted_b = cluster::kType2;

    std::cout << std::setprecision(10);
    std::cout << "pilot.L " << options.L << "\n";
    std::cout << "pilot.steps " << options.steps << "\n";
    std::cout << "pilot.beta " << static_cast<double>(options.beta) << "\n";
    std::cout << "pilot.kappa " << static_cast<double>(options.kappa) << "\n";
    std::cout << "pilot.max_cluster " << options.max_cluster << "\n";

    print_stats("full", run_case(pair, nearest_neighbors, options, full_rule,
                                  generator));
    print_stats("zero_one", run_case(pair, nearest_neighbors, options,
                                      zero_one_rule, generator));
    print_stats("zero_two", run_case(pair, nearest_neighbors, options,
                                      zero_two_rule, generator));
  } catch (const std::exception &error) {
    std::cerr << "pilot_balanced_cluster: " << error.what() << "\n";
    return 1;
  }
  return 0;
}
