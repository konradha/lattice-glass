// Tests the balanced replica cluster move in its intended regime: two
// equilibrated replicas whose disagreement set D concentrates on mobile
// regions. Reports D, cluster closure/acceptance, and E[dE | |C|] / boundary
// scaling, with a random-configuration control at the same composition.

#include "../balanced_cluster.h"
#include "../species_reduction.h"

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
namespace species = lattice_glass::species;

constexpr int kNumSizeBuckets = 7;

struct Options {
  int L = 10;
  long double beta = 2.0L;
  long double rho = 0.75L;
  long double rho1 = 0.30L;
  int equil_sweeps = 40000;
  int decorrelate_sweeps = 4000;
  int rounds = 3;
  int growth_samples = 4000;
  int cluster_max_size = 32;
  long double kappa = 0.5L;
  int heatbath_every = 200;
  unsigned int seed = 20260606u;
};

struct Diagnostics {
  int disagreements = 0;
  long double closure_rate = 0.0L;
  long double abandon_rate = 0.0L;
  long double mean_accept_prob = 0.0L;
  long double mean_size = 0.0L;
  long double mean_boundary = 0.0L;
  long double mean_delta = 0.0L;
  std::array<long double, kNumSizeBuckets> bucket_delta{};
  std::array<long double, kNumSizeBuckets> bucket_boundary{};
  std::array<long double, kNumSizeBuckets> bucket_accept{};
  std::array<int, kNumSizeBuckets> bucket_count{};
};

static int parse_int(const char *v) { return std::stoi(std::string(v)); }
static unsigned int parse_uint(const char *v) {
  return static_cast<unsigned int>(std::stoul(std::string(v)));
}
static long double parse_ld(const char *v) { return std::stold(std::string(v)); }

static Options parse_options(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string key(argv[i]);
    if (i + 1 >= argc)
      throw std::invalid_argument("missing value for " + key);
    const char *value = argv[++i];
    if (key == "--L")
      o.L = parse_int(value);
    else if (key == "--beta")
      o.beta = parse_ld(value);
    else if (key == "--rho")
      o.rho = parse_ld(value);
    else if (key == "--rho1")
      o.rho1 = parse_ld(value);
    else if (key == "--equil-sweeps")
      o.equil_sweeps = parse_int(value);
    else if (key == "--decorrelate-sweeps")
      o.decorrelate_sweeps = parse_int(value);
    else if (key == "--rounds")
      o.rounds = parse_int(value);
    else if (key == "--growth-samples")
      o.growth_samples = parse_int(value);
    else if (key == "--cluster-max-size")
      o.cluster_max_size = parse_int(value);
    else if (key == "--kappa")
      o.kappa = parse_ld(value);
    else if (key == "--heatbath-every")
      o.heatbath_every = parse_int(value);
    else if (key == "--seed")
      o.seed = parse_uint(value);
    else
      throw std::invalid_argument("unknown option " + key);
  }
  if (o.L < 8)
    throw std::invalid_argument("L must be at least 8");
  if (o.beta < 0.0L)
    throw std::invalid_argument("beta must be non-negative");
  if (o.rho < 0.0L || o.rho > 1.0L)
    throw std::invalid_argument("rho must be in [0,1]");
  if (o.rho1 < 0.0L || o.rho1 > o.rho)
    throw std::invalid_argument("rho1 must be in [0,rho]");
  if (o.rounds <= 0 || o.growth_samples <= 0 || o.cluster_max_size <= 0)
    throw std::invalid_argument("rounds/growth/max-size must be positive");
  return o;
}

static std::vector<int> cubic_neighbors(int L) {
  const int n = L * L * L;
  std::vector<int> nn(n * 6);
  for (int i = 0; i < L; ++i)
    for (int j = 0; j < L; ++j)
      for (int k = 0; k < L; ++k) {
        const int site = k + L * (j + i * L);
        nn[6 * site + 0] = k + L * (j + ((i + L - 1) % L) * L);
        nn[6 * site + 1] = k + L * (j + ((i + 1) % L) * L);
        nn[6 * site + 2] = k + L * (((j + L - 1) % L) + i * L);
        nn[6 * site + 3] = k + L * (((j + 1) % L) + i * L);
        nn[6 * site + 4] = ((k + L - 1) % L) + L * (j + i * L);
        nn[6 * site + 5] = ((k + 1) % L) + L * (j + i * L);
      }
  return nn;
}

static std::vector<uint8_t> random_lattice(int num_type1, int num_type2,
                                           int lat_size, std::mt19937 &gen) {
  std::vector<uint8_t> lattice;
  lattice.reserve(lat_size);
  lattice.insert(lattice.end(), num_type1, cluster::kType1);
  lattice.insert(lattice.end(), num_type2, cluster::kType2);
  lattice.insert(lattice.end(), lat_size - num_type1 - num_type2, cluster::kEmpty);
  std::shuffle(lattice.begin(), lattice.end(), gen);
  return lattice;
}

static inline int local_energy_int(const std::vector<uint8_t> &lattice,
                                   int site, const int *nn) {
  const uint8_t type = lattice[site];
  if (type == cluster::kEmpty)
    return 0;
  const int preferred = type == cluster::kType1 ? 3 : 5;
  const int *neigh = nn + 6 * site;
  int occupied = 0;
  for (int i = 0; i < 6; ++i)
    occupied += lattice[neigh[i]] != cluster::kEmpty;
  const int delta = occupied - preferred;
  return delta * delta;
}

static long double total_energy_int(const std::vector<uint8_t> &lattice,
                                    const int *nn) {
  long double energy = 0.0L;
  for (int site = 0; site < static_cast<int>(lattice.size()); ++site)
    energy += local_energy_int(lattice, site, nn);
  return energy;
}

// Energy contribution of the two swapped sites plus their neighbourhoods.
static int swap_region_energy(const std::vector<uint8_t> &lattice, int a, int b,
                              const int *nn) {
  int affected[14];
  int count = 0;
  auto push = [&](int site) {
    for (int i = 0; i < count; ++i)
      if (affected[i] == site)
        return;
    affected[count++] = site;
  };
  push(a);
  push(b);
  for (int i = 0; i < 6; ++i) {
    push(nn[6 * a + i]);
    push(nn[6 * b + i]);
  }
  int energy = 0;
  for (int i = 0; i < count; ++i)
    energy += local_energy_int(lattice, affected[i], nn);
  return energy;
}

static int nonlocal_swap_sweep(std::vector<uint8_t> &lattice,
                               long double beta, std::mt19937 &gen,
                               const int *nn) {
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
    std::swap(lattice[a], lattice[b]);
    const int after = swap_region_energy(lattice, a, b, nn);
    const int delta = after - before;
    if (delta <= 0 ||
        uni(gen) < std::exp(-static_cast<double>(beta) * delta)) {
      ++accepted;
    } else {
      std::swap(lattice[a], lattice[b]);
    }
  }
  return accepted;
}

static void equilibrate(std::vector<uint8_t> &lattice, long double beta,
                        int sweeps, int num_type1, int heatbath_every,
                        std::mt19937 &gen, std::mt19937 &heat_gen,
                        const std::vector<int> &nn) {
  for (int sweep = 1; sweep <= sweeps; ++sweep) {
    nonlocal_swap_sweep(lattice, beta, gen, nn.data());
    if (heatbath_every > 0 && sweep % heatbath_every == 0)
      lattice = species::sample_species_given_occupancy(
          lattice, nn.data(), 6, beta, num_type1, heat_gen);
  }
}

static int disagreement_count(const cluster::ReplicaPair &pair) {
  int count = 0;
  for (int site = 0; site < static_cast<int>(pair.first.size()); ++site)
    count += pair.first[site] != pair.second[site];
  return count;
}

static int size_bucket(int size) {
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
  return 6;
}

static const char *bucket_name(int b) {
  static const char *names[kNumSizeBuckets] = {"le2",  "3_4",   "5_8", "9_16",
                                               "17_32", "33_64", "gt64"};
  return names[b];
}

// Measures cluster-growth statistics on a FIXED pair without accepting moves,
// so the disagreement set stays constant across all samples.
static Diagnostics measure(cluster::ReplicaPair &pair,
                           const std::vector<int> &nn, const Options &o,
                           const cluster::GrowthRule &rule, std::mt19937 &gen) {
  Diagnostics d;
  d.disagreements = disagreement_count(pair);

  int proposed = 0;
  int abandoned = 0;
  long double size_sum = 0.0L, boundary_sum = 0.0L, delta_sum = 0.0L;
  long double accept_sum = 0.0L;
  std::array<long double, kNumSizeBuckets> bdelta{}, bbound{}, baccept{};
  std::array<int, kNumSizeBuckets> bcount{};

  for (int sample = 0; sample < o.growth_samples; ++sample) {
    const cluster::GrowthAttempt attempt = cluster::grow_balanced_cluster(
        pair, nn.data(), 6, o.cluster_max_size, rule, gen);
    if (attempt.abandoned) {
      ++abandoned;
      continue;
    }
    if (!attempt.proposed)
      continue;

    const int size = static_cast<int>(attempt.cluster_sites.size());
    const std::vector<int> affected =
        cluster::cluster_affected_sites(attempt.cluster_sites, nn.data(), 6);
    const int boundary = static_cast<int>(affected.size() - size);
    const long double delta = cluster::cluster_exchange_delta(
        pair, attempt.cluster_sites, nn.data(), 6);
    const long double accept_prob =
        delta <= 0.0L ? 1.0L : std::expl(-o.beta * delta);

    ++proposed;
    size_sum += size;
    boundary_sum += boundary;
    delta_sum += delta;
    accept_sum += accept_prob;

    const int bucket = size_bucket(size);
    bdelta[bucket] += delta;
    bbound[bucket] += boundary;
    baccept[bucket] += accept_prob;
    bcount[bucket]++;
  }

  const long double samples = o.growth_samples;
  d.closure_rate = proposed / samples;
  d.abandon_rate = abandoned / samples;
  if (proposed > 0) {
    d.mean_size = size_sum / proposed;
    d.mean_boundary = boundary_sum / proposed;
    d.mean_delta = delta_sum / proposed;
    d.mean_accept_prob = accept_sum / proposed;
  }
  for (int b = 0; b < kNumSizeBuckets; ++b) {
    d.bucket_count[b] = bcount[b];
    if (bcount[b] > 0) {
      d.bucket_delta[b] = bdelta[b] / bcount[b];
      d.bucket_boundary[b] = bbound[b] / bcount[b];
      d.bucket_accept[b] = baccept[b] / bcount[b];
    }
  }
  return d;
}

static Diagnostics average(const std::vector<Diagnostics> &runs) {
  Diagnostics avg;
  const long double n = runs.size();
  for (const Diagnostics &r : runs) {
    avg.disagreements += r.disagreements;
    avg.closure_rate += r.closure_rate;
    avg.abandon_rate += r.abandon_rate;
    avg.mean_accept_prob += r.mean_accept_prob;
    avg.mean_size += r.mean_size;
    avg.mean_boundary += r.mean_boundary;
    avg.mean_delta += r.mean_delta;
    for (int b = 0; b < kNumSizeBuckets; ++b) {
      avg.bucket_delta[b] += r.bucket_delta[b];
      avg.bucket_boundary[b] += r.bucket_boundary[b];
      avg.bucket_accept[b] += r.bucket_accept[b];
      avg.bucket_count[b] += r.bucket_count[b];
    }
  }
  avg.disagreements = static_cast<int>(avg.disagreements / n);
  avg.closure_rate /= n;
  avg.abandon_rate /= n;
  avg.mean_accept_prob /= n;
  avg.mean_size /= n;
  avg.mean_boundary /= n;
  avg.mean_delta /= n;
  for (int b = 0; b < kNumSizeBuckets; ++b) {
    avg.bucket_delta[b] /= n;
    avg.bucket_boundary[b] /= n;
    avg.bucket_accept[b] /= n;
  }
  return avg;
}

static void print_diag(const std::string &prefix, const Diagnostics &d,
                       int lat_size) {
  std::cout << prefix << ".disagreements " << d.disagreements << "\n";
  std::cout << prefix << ".disagreement_fraction "
            << static_cast<long double>(d.disagreements) / lat_size << "\n";
  std::cout << prefix << ".closure_rate " << d.closure_rate << "\n";
  std::cout << prefix << ".abandon_rate " << d.abandon_rate << "\n";
  std::cout << prefix << ".mean_accept_prob " << d.mean_accept_prob << "\n";
  std::cout << prefix << ".mean_size " << d.mean_size << "\n";
  std::cout << prefix << ".mean_boundary " << d.mean_boundary << "\n";
  std::cout << prefix << ".mean_delta " << d.mean_delta << "\n";
  for (int b = 0; b < kNumSizeBuckets; ++b) {
    if (d.bucket_count[b] == 0)
      continue;
    std::cout << prefix << ".size_" << bucket_name(b) << ".count "
              << d.bucket_count[b] << "\n";
    std::cout << prefix << ".size_" << bucket_name(b) << ".mean_delta "
              << d.bucket_delta[b] << "\n";
    std::cout << prefix << ".size_" << bucket_name(b) << ".mean_boundary "
              << d.bucket_boundary[b] << "\n";
    std::cout << prefix << ".size_" << bucket_name(b) << ".mean_accept_prob "
              << d.bucket_accept[b] << "\n";
  }
}

int main(int argc, char **argv) {
  try {
    const Options o = parse_options(argc, argv);
    const int lat_size = o.L * o.L * o.L;
    const int num_particles = static_cast<int>(o.rho * lat_size);
    const int num_type1 = static_cast<int>(o.rho1 * lat_size);
    const int num_type2 = num_particles - num_type1;
    if (num_type1 < 0 || num_type2 < 0 || num_particles > lat_size)
      throw std::invalid_argument("invalid composition");

    const std::vector<int> nn = cubic_neighbors(o.L);

    std::mt19937 gen1(o.seed);
    std::mt19937 gen2(o.seed ^ 0x9E3779B9u);
    std::mt19937 heat1(o.seed ^ 0x85EBCA6Bu);
    std::mt19937 heat2(o.seed ^ 0xC2B2AE35u);
    std::mt19937 growth_gen(o.seed ^ 0x27D4EB2Fu);

    cluster::GrowthRule rule;
    rule.kappa = o.kappa;

    std::cout << std::setprecision(8);
    std::cout << "diag.L " << o.L << "\n";
    std::cout << "diag.beta " << static_cast<double>(o.beta) << "\n";
    std::cout << "diag.rho " << static_cast<double>(o.rho) << "\n";
    std::cout << "diag.rho1 " << static_cast<double>(o.rho1) << "\n";
    std::cout << "diag.equil_sweeps " << o.equil_sweeps << "\n";
    std::cout << "diag.rounds " << o.rounds << "\n";
    std::cout << "diag.growth_samples " << o.growth_samples << "\n";
    std::cout << "diag.cluster_max_size " << o.cluster_max_size << "\n";
    std::cout << "diag.kappa " << static_cast<double>(o.kappa) << "\n";

    // Control: random configurations at fixed composition.
    cluster::ReplicaPair random_pair;
    random_pair.first = random_lattice(num_type1, num_type2, lat_size, gen1);
    random_pair.second = random_lattice(num_type1, num_type2, lat_size, gen2);
    std::cout << "random.energy "
              << static_cast<double>(
                     (total_energy_int(random_pair.first, nn.data()) +
                      total_energy_int(random_pair.second, nn.data())) /
                     2.0L)
              << "\n";
    const Diagnostics random_diag =
        measure(random_pair, nn, o, rule, growth_gen);
    print_diag("random", random_diag, lat_size);

    // Relaxed: equilibrate both replicas, then measure over several rounds.
    cluster::ReplicaPair pair;
    pair.first = random_lattice(num_type1, num_type2, lat_size, gen1);
    pair.second = random_lattice(num_type1, num_type2, lat_size, gen2);
    equilibrate(pair.first, o.beta, o.equil_sweeps, num_type1, o.heatbath_every,
                gen1, heat1, nn);
    equilibrate(pair.second, o.beta, o.equil_sweeps, num_type1, o.heatbath_every,
                gen2, heat2, nn);
    std::cout << "relaxed.energy "
              << static_cast<double>(
                     (total_energy_int(pair.first, nn.data()) +
                      total_energy_int(pair.second, nn.data())) /
                     2.0L)
              << "\n";

    std::vector<Diagnostics> rounds;
    for (int round = 0; round < o.rounds; ++round) {
      if (round > 0) {
        equilibrate(pair.first, o.beta, o.decorrelate_sweeps, num_type1,
                    o.heatbath_every, gen1, heat1, nn);
        equilibrate(pair.second, o.beta, o.decorrelate_sweeps, num_type1,
                    o.heatbath_every, gen2, heat2, nn);
      }
      rounds.push_back(measure(pair, nn, o, rule, growth_gen));
    }
    print_diag("relaxed", average(rounds), lat_size);
  } catch (const std::exception &error) {
    std::cerr << "cluster_diagnostics: " << error.what() << "\n";
    return 1;
  }
  return 0;
}
