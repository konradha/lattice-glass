// Franz-Parisi / reference-coupled efficiency harness.
//
// Two same-temperature replicas are symmetrically coupled to a common quenched
// equilibrium reference s0 by a field eps. The field is tuned (per temperature)
// so the inter-replica disagreement fraction |D|/N sits just below the 3D
// simple-cubic SITE percolation threshold p_c ~ 0.3116 -- the regime where the
// balanced cluster move is sub-percolating and can be accepted.
//
// It then benchmarks two kernels at matched wall-clock, both sampling the same
// eps-tilted measure (the field cancels in the cluster acceptance, so the
// stationary law is identical):
//   arm 'base'    : reference-coupled non-local swap only
//   arm 'cluster' : reference-coupled non-local swap + balanced cluster move
// and reports integrated autocorrelation times, effective sample size, and
// ESS-per-second for energy, replica-reference overlap, and mutual overlap.

#include "../balanced_cluster.h"
#include "../fp_sampler.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace cluster = lattice_glass::cluster;
namespace fp = lattice_glass::fp;

// 3D simple-cubic site percolation threshold.
constexpr long double kSitePercolation = 0.3116L;

struct Options {
  int L = 10;
  long double beta = 3.0L;
  long double rho = 0.75L;
  long double rho1 = 0.30L;
  long double epsilon = -1.0L;     // <0 means calibrate
  long double target_d = 0.15L;    // target |D|/N (below p_c)
  int ref_equil_sweeps = 8000;
  int calib_equil_sweeps = 3000;
  int warmup_sweeps = 4000;
  int production_sweeps = 60000;
  int sample_every = 5;
  int cluster_every = 1;
  int cluster_max_size = 32;
  long double kappa = 0.5L;
  unsigned int seed = 20260606u;
};

struct ArmResult {
  double seconds = 0.0;
  int samples = 0;
  fp::AutocorrResult energy;
  fp::AutocorrResult q_ref;
  fp::AutocorrResult q_mutual;
  long long cluster_attempts = 0;
  long long cluster_accepted = 0;
  long double cluster_size_sum = 0.0L;
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
    else if (key == "--epsilon")
      o.epsilon = parse_ld(value);
    else if (key == "--target-d")
      o.target_d = parse_ld(value);
    else if (key == "--ref-equil-sweeps")
      o.ref_equil_sweeps = parse_int(value);
    else if (key == "--calib-equil-sweeps")
      o.calib_equil_sweeps = parse_int(value);
    else if (key == "--warmup-sweeps")
      o.warmup_sweeps = parse_int(value);
    else if (key == "--production-sweeps")
      o.production_sweeps = parse_int(value);
    else if (key == "--sample-every")
      o.sample_every = parse_int(value);
    else if (key == "--cluster-every")
      o.cluster_every = parse_int(value);
    else if (key == "--cluster-max-size")
      o.cluster_max_size = parse_int(value);
    else if (key == "--kappa")
      o.kappa = parse_ld(value);
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
  if (o.target_d <= 0.0L || o.target_d >= 1.0L)
    throw std::invalid_argument("target-d must be in (0,1)");
  if (o.production_sweeps <= o.sample_every)
    throw std::invalid_argument("production must exceed sample cadence");
  return o;
}

static cluster::GrowthRule make_rule(const Options &o) {
  cluster::GrowthRule rule;
  rule.kappa = o.kappa;
  return rule;
}

// Mean |D|/N for a fresh, briefly equilibrated pair at a given epsilon.
static long double measure_disagreement(const Options &o,
                                        const std::vector<int> &nn,
                                        const std::vector<uint8_t> &reference,
                                        int num_type1, int num_type2,
                                        long double epsilon,
                                        std::mt19937 &gen) {
  const int lat_size = o.L * o.L * o.L;
  std::vector<uint8_t> r1 = fp::random_lattice(num_type1, num_type2, lat_size, gen);
  std::vector<uint8_t> r2 = fp::random_lattice(num_type1, num_type2, lat_size, gen);
  const std::vector<uint8_t> *ref = epsilon > 0.0L ? &reference : nullptr;
  fp::equilibrate(r1, o.beta, epsilon, ref, o.calib_equil_sweeps, gen, nn.data());
  fp::equilibrate(r2, o.beta, epsilon, ref, o.calib_equil_sweeps, gen, nn.data());

  long double sum = 0.0L;
  const int probes = 4;
  for (int p = 0; p < probes; ++p) {
    fp::equilibrate(r1, o.beta, epsilon, ref, 200, gen, nn.data());
    fp::equilibrate(r2, o.beta, epsilon, ref, 200, gen, nn.data());
    sum += static_cast<long double>(fp::disagreement_count(r1, r2)) / lat_size;
  }
  return sum / probes;
}

// Coarse calibration: pick the epsilon whose |D|/N is closest to the target
// (and below the percolation threshold).
static long double calibrate_epsilon(const Options &o, const std::vector<int> &nn,
                                     const std::vector<uint8_t> &reference,
                                     int num_type1, int num_type2,
                                     std::mt19937 &gen) {
  const long double grid[] = {0.0L,  0.25L, 0.5L, 0.75L,
                              1.0L,  1.25L, 1.5L, 2.0L};
  long double best_eps = 0.0L;
  long double best_gap = 1e9L;
  for (long double eps : grid) {
    const long double d =
        measure_disagreement(o, nn, reference, num_type1, num_type2, eps, gen);
    std::cout << "calib.epsilon " << static_cast<double>(eps)
              << " disagreement_fraction " << static_cast<double>(d) << "\n";
    // Prefer points below threshold; penalise super-percolating D heavily.
    const long double penalty = d > kSitePercolation ? 10.0L : 0.0L;
    const long double gap = std::fabsl(d - o.target_d) + penalty;
    if (gap < best_gap) {
      best_gap = gap;
      best_eps = eps;
    }
  }
  return best_eps;
}

static ArmResult run_arm(const Options &o, const std::vector<int> &nn,
                         const std::vector<uint8_t> &reference, int num_type1,
                         int num_type2, long double epsilon, bool use_cluster,
                         std::mt19937 &gen) {
  const int lat_size = o.L * o.L * o.L;
  const std::vector<uint8_t> *ref = epsilon > 0.0L ? &reference : nullptr;
  const cluster::GrowthRule rule = make_rule(o);

  cluster::ReplicaPair pair;
  pair.first = fp::random_lattice(num_type1, num_type2, lat_size, gen);
  pair.second = fp::random_lattice(num_type1, num_type2, lat_size, gen);

  auto cluster_step = [&]() -> cluster::MetropolisResult {
    return cluster::metropolis_cluster_exchange(pair, o.beta, nn.data(),
                                                fp::kNumNeighbors,
                                                o.cluster_max_size, rule, gen);
  };

  // Warmup with the arm's own kernel.
  for (int sweep = 1; sweep <= o.warmup_sweeps; ++sweep) {
    fp::nonlocal_swap_sweep(pair.first, o.beta, epsilon, ref, gen, nn.data());
    fp::nonlocal_swap_sweep(pair.second, o.beta, epsilon, ref, gen, nn.data());
    if (use_cluster && sweep % o.cluster_every == 0)
      cluster_step();
  }

  std::vector<long double> energy, q_ref, q_mutual;
  ArmResult result;
  const auto start = std::chrono::steady_clock::now();
  for (int sweep = 1; sweep <= o.production_sweeps; ++sweep) {
    fp::nonlocal_swap_sweep(pair.first, o.beta, epsilon, ref, gen, nn.data());
    fp::nonlocal_swap_sweep(pair.second, o.beta, epsilon, ref, gen, nn.data());
    if (use_cluster && sweep % o.cluster_every == 0) {
      const cluster::MetropolisResult r = cluster_step();
      ++result.cluster_attempts;
      if (r.accepted) {
        ++result.cluster_accepted;
        result.cluster_size_sum += r.cluster_size;
      }
    }
    if (sweep % o.sample_every == 0) {
      const long double e = (fp::total_energy_int(pair.first, nn.data()) +
                             fp::total_energy_int(pair.second, nn.data())) /
                            (2.0L * lat_size);
      energy.push_back(e);
      q_ref.push_back(static_cast<long double>(
                          fp::agreement_count(pair.first, reference)) /
                      lat_size);
      q_mutual.push_back(static_cast<long double>(
                             fp::agreement_count(pair.first, pair.second)) /
                         lat_size);
    }
  }
  const auto end = std::chrono::steady_clock::now();
  result.seconds = std::chrono::duration<double>(end - start).count();
  result.samples = static_cast<int>(energy.size());
  result.energy = fp::integrated_autocorrelation_time(energy);
  result.q_ref = fp::integrated_autocorrelation_time(q_ref);
  result.q_mutual = fp::integrated_autocorrelation_time(q_mutual);
  return result;
}

static void print_arm(const std::string &name, const ArmResult &r) {
  auto ess_per_sec = [&](const fp::AutocorrResult &a) {
    const long double ess = fp::effective_samples(r.samples, a.tau_int);
    return r.seconds > 0.0 ? static_cast<double>(ess) / r.seconds : 0.0;
  };
  std::cout << name << ".seconds " << r.seconds << "\n";
  std::cout << name << ".samples " << r.samples << "\n";
  std::cout << name << ".energy_tau " << static_cast<double>(r.energy.tau_int)
            << " resolved " << (r.energy.resolved ? 1 : 0) << "\n";
  std::cout << name << ".q_ref_tau " << static_cast<double>(r.q_ref.tau_int)
            << " resolved " << (r.q_ref.resolved ? 1 : 0) << "\n";
  std::cout << name << ".q_mutual_tau "
            << static_cast<double>(r.q_mutual.tau_int) << " resolved "
            << (r.q_mutual.resolved ? 1 : 0) << "\n";
  std::cout << name << ".energy_ess_per_sec " << ess_per_sec(r.energy) << "\n";
  std::cout << name << ".q_ref_ess_per_sec " << ess_per_sec(r.q_ref) << "\n";
  std::cout << name << ".q_mutual_ess_per_sec " << ess_per_sec(r.q_mutual)
            << "\n";
  if (r.cluster_attempts > 0) {
    std::cout << name << ".cluster_accept_per_attempt "
              << static_cast<double>(r.cluster_accepted) / r.cluster_attempts
              << "\n";
    std::cout << name << ".cluster_mean_accepted_size "
              << (r.cluster_accepted > 0
                      ? static_cast<double>(r.cluster_size_sum /
                                            r.cluster_accepted)
                      : 0.0)
              << "\n";
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

    const std::vector<int> nn = fp::cubic_neighbors(o.L);

    std::mt19937 ref_gen(o.seed ^ 0x51ED2701u);
    std::mt19937 calib_gen(o.seed ^ 0x1B56C4E9u);
    std::mt19937 base_gen(o.seed ^ 0x9E3779B9u);
    std::mt19937 cluster_gen(o.seed ^ 0xC2B2AE35u);

    std::cout << std::setprecision(8);
    std::cout << "fp.L " << o.L << "\n";
    std::cout << "fp.beta " << static_cast<double>(o.beta) << "\n";
    std::cout << "fp.rho " << static_cast<double>(o.rho) << "\n";
    std::cout << "fp.rho1 " << static_cast<double>(o.rho1) << "\n";
    std::cout << "fp.target_d " << static_cast<double>(o.target_d) << "\n";
    std::cout << "fp.percolation_threshold "
              << static_cast<double>(kSitePercolation) << "\n";
    std::cout << "fp.production_sweeps " << o.production_sweeps << "\n";
    std::cout << "fp.cluster_every " << o.cluster_every << "\n";
    std::cout << "fp.cluster_max_size " << o.cluster_max_size << "\n";

    // Quenched reference: an equilibrium configuration of the plain model.
    std::vector<uint8_t> reference =
        fp::random_lattice(num_type1, num_type2, lat_size, ref_gen);
    fp::equilibrate(reference, o.beta, 0.0L, nullptr, o.ref_equil_sweeps,
                    ref_gen, nn.data());
    std::cout << "reference.energy_per_site "
              << static_cast<double>(fp::total_energy_int(reference, nn.data()) /
                                     lat_size)
              << "\n";

    long double epsilon = o.epsilon;
    if (epsilon < 0.0L)
      epsilon = calibrate_epsilon(o, nn, reference, num_type1, num_type2,
                                  calib_gen);
    const long double calibrated_d = measure_disagreement(
        o, nn, reference, num_type1, num_type2, epsilon, calib_gen);
    std::cout << "fp.epsilon " << static_cast<double>(epsilon) << "\n";
    std::cout << "fp.calibrated_disagreement_fraction "
              << static_cast<double>(calibrated_d) << "\n";
    std::cout << "fp.subpercolating "
              << (calibrated_d < kSitePercolation ? 1 : 0) << "\n";

    const ArmResult base =
        run_arm(o, nn, reference, num_type1, num_type2, epsilon, false, base_gen);
    print_arm("base", base);

    const ArmResult clustered = run_arm(o, nn, reference, num_type1, num_type2,
                                        epsilon, true, cluster_gen);
    print_arm("cluster", clustered);

    auto speedup = [&](const fp::AutocorrResult &b, const fp::AutocorrResult &c) {
      const long double base_eps =
          fp::effective_samples(base.samples, b.tau_int) /
          (base.seconds > 0.0 ? base.seconds : 1.0);
      const long double cl_eps =
          fp::effective_samples(clustered.samples, c.tau_int) /
          (clustered.seconds > 0.0 ? clustered.seconds : 1.0);
      return base_eps > 0.0L ? static_cast<double>(cl_eps / base_eps) : 0.0;
    };
    std::cout << "speedup.energy_ess_per_sec "
              << speedup(base.energy, clustered.energy) << "\n";
    std::cout << "speedup.q_ref_ess_per_sec "
              << speedup(base.q_ref, clustered.q_ref) << "\n";
    std::cout << "speedup.q_mutual_ess_per_sec "
              << speedup(base.q_mutual, clustered.q_mutual) << "\n";
  } catch (const std::exception &error) {
    std::cerr << "fp_efficiency: " << error.what() << "\n";
    return 1;
  }
  return 0;
}
