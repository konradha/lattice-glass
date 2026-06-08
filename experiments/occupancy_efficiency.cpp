// Single-system occupancy-sampling efficiency harness.
//
// Compares two detailed-balance-respecting samplers of the NH Gibbs measure at
// fixed composition, at matched wall-clock:
//   arm 'swap'      : non-local random-pair swap (the proven accelerator)
//   arm 'swapchain' : non-local swap + cooperative vacancy-chain moves
// and, on the same chains, contrasts two estimators of <H>:
//   plain energy   : actual H of the sampled config (occupancy + species noise)
//   RB energy      : exact E[H | occupancy], species integrated out
// reporting integrated autocorrelation time, ESS, and ESS-per-second.

#include "../collective_moves.h"
#include "../fp_sampler.h"
#include "../lifted_swap.h"
#include "../lifted_vacancy.h"
#include "../species_reduction.h"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fp = lattice_glass::fp;
namespace collective = lattice_glass::collective;
namespace species = lattice_glass::species;
namespace lifted = lattice_glass::lifted;
namespace axis_lift = lattice_glass::axis_lift;

struct Options {
  int L = 10;
  long double beta = 3.0L;
  long double rho = 0.75L;
  long double rho1 = 0.30L;
  int warmup = 20000;
  int production = 200000;
  int sample_every = 4;
  int k_max = 6;
  int chain_attempts = 120;
  int stride = 4;
  unsigned int seed = 20260606u;
};

struct ArmResult {
  double seconds = 0.0;
  int samples = 0;
  fp::AutocorrResult plain;
  fp::AutocorrResult rb;
  long long chain_attempts = 0;
  long long chain_accepted = 0;
};

static int parse_int(const char *v) { return std::stoi(std::string(v)); }
static unsigned parse_uint(const char *v) {
  return static_cast<unsigned>(std::stoul(std::string(v)));
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
    else if (key == "--warmup")
      o.warmup = parse_int(value);
    else if (key == "--production")
      o.production = parse_int(value);
    else if (key == "--sample-every")
      o.sample_every = parse_int(value);
    else if (key == "--k-max")
      o.k_max = parse_int(value);
    else if (key == "--chain-attempts")
      o.chain_attempts = parse_int(value);
    else if (key == "--stride")
      o.stride = parse_int(value);
    else if (key == "--seed")
      o.seed = parse_uint(value);
    else
      throw std::invalid_argument("unknown option " + key);
  }
  if (o.L < 8)
    throw std::invalid_argument("L must be at least 8");
  if (o.production <= o.sample_every)
    throw std::invalid_argument("production must exceed sample cadence");
  return o;
}

enum class Mode { Swap, SwapChain, SwapLifted, SwapLiftedSwap };

static ArmResult run_arm(const Options &o, const std::vector<int> &nn,
                         int num_type1, int num_type2, Mode mode,
                         std::mt19937 &gen) {
  const int lat_size = o.L * o.L * o.L;
  std::vector<uint8_t> lattice =
      fp::random_lattice(num_type1, num_type2, lat_size, gen);

  const long long lifted_events = static_cast<long long>(lat_size);
  const int lifted_refresh = 2 * o.L;
  for (int s = 0; s < o.warmup; ++s) {
    fp::nonlocal_swap_sweep(lattice, o.beta, 0.0L, nullptr, gen, nn.data());
    if (mode == Mode::SwapChain)
      collective::vacancy_chain_sweep(lattice, o.beta, o.k_max,
                                      o.chain_attempts, gen, nn.data());
    else if (mode == Mode::SwapLifted)
      lifted::lifted_vacancy_chain(lattice, o.beta, lifted_events,
                                   lifted_refresh, gen, nn.data());
    else if (mode == Mode::SwapLiftedSwap)
      axis_lift::lifted_swap_chain(lattice, o.beta, lifted_events, o.stride,
                                   lifted_refresh, o.L, gen, nn.data());
  }

  std::vector<long double> plain, rb;
  ArmResult result;
  const auto t0 = std::chrono::steady_clock::now();
  for (int s = 1; s <= o.production; ++s) {
    fp::nonlocal_swap_sweep(lattice, o.beta, 0.0L, nullptr, gen, nn.data());
    if (mode == Mode::SwapChain) {
      const collective::SweepStats cs = collective::vacancy_chain_sweep(
          lattice, o.beta, o.k_max, o.chain_attempts, gen, nn.data());
      result.chain_attempts += cs.attempts;
      result.chain_accepted += cs.accepted;
    } else if (mode == Mode::SwapLifted) {
      const lifted::EventStats ls = lifted::lifted_vacancy_chain(
          lattice, o.beta, lifted_events, lifted_refresh, gen, nn.data());
      result.chain_attempts += ls.events;
      result.chain_accepted += ls.accepted;
    } else if (mode == Mode::SwapLiftedSwap) {
      const axis_lift::EventStats ls = axis_lift::lifted_swap_chain(
          lattice, o.beta, lifted_events, o.stride, lifted_refresh, o.L, gen,
          nn.data());
      result.chain_attempts += ls.events;
      result.chain_accepted += ls.accepted;
    }
    if (s % o.sample_every == 0) {
      plain.push_back(fp::total_energy_int(lattice, nn.data()));
      rb.push_back(species::rao_blackwell_energy(lattice, nn.data(),
                                                 fp::kNumNeighbors, o.beta,
                                                 num_type1));
    }
  }
  const auto t1 = std::chrono::steady_clock::now();
  result.seconds = std::chrono::duration<double>(t1 - t0).count();
  result.samples = static_cast<int>(plain.size());
  result.plain = fp::integrated_autocorrelation_time(plain);
  result.rb = fp::integrated_autocorrelation_time(rb);
  return result;
}

static double ess_per_sec(const ArmResult &a, const fp::AutocorrResult &x) {
  const long double ess = fp::effective_samples(a.samples, x.tau_int);
  return a.seconds > 0.0 ? static_cast<double>(ess) / a.seconds : 0.0;
}

static void print_arm(const std::string &name, const ArmResult &a) {
  std::cout << name << ".seconds " << a.seconds << "\n";
  std::cout << name << ".samples " << a.samples << "\n";
  std::cout << name << ".plain_energy_tau " << static_cast<double>(a.plain.tau_int)
            << " resolved " << (a.plain.resolved ? 1 : 0) << "\n";
  std::cout << name << ".rb_energy_tau " << static_cast<double>(a.rb.tau_int)
            << " resolved " << (a.rb.resolved ? 1 : 0) << "\n";
  std::cout << name << ".plain_energy_ess_per_sec " << ess_per_sec(a, a.plain)
            << "\n";
  std::cout << name << ".rb_energy_ess_per_sec " << ess_per_sec(a, a.rb) << "\n";
  if (a.plain.variance > 0.0L)
    std::cout << name << ".rb_variance_reduction "
              << static_cast<double>(a.plain.variance / a.rb.variance) << "\n";
  if (a.chain_attempts > 0)
    std::cout << name << ".chain_accept_per_attempt "
              << static_cast<double>(a.chain_accepted) / a.chain_attempts
              << "\n";
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
    std::mt19937 swap_gen(o.seed ^ 0x9E3779B9u);
    std::mt19937 chain_gen(o.seed ^ 0xC2B2AE35u);

    std::cout << std::setprecision(8);
    std::cout << "occ.L " << o.L << "\n";
    std::cout << "occ.beta " << static_cast<double>(o.beta) << "\n";
    std::cout << "occ.production " << o.production << "\n";
    std::cout << "occ.k_max " << o.k_max << "\n";
    std::cout << "occ.chain_attempts " << o.chain_attempts << "\n";
    std::cout << "occ.stride " << o.stride << "\n";

    const ArmResult swap =
        run_arm(o, nn, num_type1, num_type2, Mode::Swap, swap_gen);
    print_arm("swap", swap);
    const ArmResult swapliftedswap = run_arm(
        o, nn, num_type1, num_type2, Mode::SwapLiftedSwap, chain_gen);
    print_arm("swapliftedswap", swapliftedswap);
    std::mt19937 lifted_gen(o.seed ^ 0x1B56C4E9u);
    const ArmResult swaplifted =
        run_arm(o, nn, num_type1, num_type2, Mode::SwapLifted, lifted_gen);
    print_arm("swaplifted", swaplifted);

    auto speedup = [&](const fp::AutocorrResult &b, const fp::AutocorrResult &c,
                       const ArmResult &ba, const ArmResult &ca) {
      const double base = ess_per_sec(ba, b);
      const double clu = ess_per_sec(ca, c);
      return base > 0.0 ? clu / base : 0.0;
    };
    std::cout << "speedup.liftedswap_plain_energy_ess_per_sec "
              << speedup(swap.plain, swapliftedswap.plain, swap, swapliftedswap)
              << "\n";
    std::cout << "speedup.liftedswap_rb_energy_ess_per_sec "
              << speedup(swap.rb, swapliftedswap.rb, swap, swapliftedswap)
              << "\n";
    std::cout << "speedup.lifted_plain_energy_ess_per_sec "
              << speedup(swap.plain, swaplifted.plain, swap, swaplifted) << "\n";
    std::cout << "speedup.lifted_rb_energy_ess_per_sec "
              << speedup(swap.rb, swaplifted.rb, swap, swaplifted) << "\n";
    // RB-vs-plain measurement gain on the baseline (provable, dynamics-free).
    std::cout << "rb_measurement_gain.ess_per_sec "
              << (ess_per_sec(swap, swap.plain) > 0.0
                      ? ess_per_sec(swap, swap.rb) / ess_per_sec(swap, swap.plain)
                      : 0.0)
              << "\n";
  } catch (const std::exception &error) {
    std::cerr << "occupancy_efficiency: " << error.what() << "\n";
    return 1;
  }
  return 0;
}
