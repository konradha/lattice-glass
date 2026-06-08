// Informed (locally-balanced) vs blind nonlocal occupancy swap efficiency.
//
// The experiment that asks whether informing the destination proposal fixes the
// nonlocal-swap acceptance collapse on the NH occupancy bottleneck (see
// informed_swap.h). Three detailed-balance samplers of the fixed-composition
// Gibbs measure, at matched wall-clock:
//   arm 'fullswap' : trusted all-pairs nonlocal swap (the NH baseline)
//   arm 'blind'    : occupancy-only swap, destination chosen UNIFORMLY
//   arm 'informed' : occupancy-only swap, destination ~ sqrt(exp(-beta dE))
// blind and informed share the exact same move set and sweep unit (one attempt
// per particle), so they differ ONLY in the destination proposal -- isolating
// the effect of informing it.
//
// For each arm it reports occupancy-move acceptance and, for two observables --
// total energy H and the pure-occupancy bond count B = sum_{occ} m_i / 2 -- the
// integrated autocorrelation time (with the Madras-Sokal resolved flag), ESS,
// and ESS-per-second. The headline metrics are informed-vs-blind on B: a true
// fix of the occupancy bottleneck shows up as both restored acceptance and a
// lower bond autocorrelation per sweep AND per second. B is the clean target
// because energy is also sensitive to species-label mixing, which both
// occupancy arms drive identically (by relocation).

#include "../fp_sampler.h"
#include "../informed_swap.h"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fp = lattice_glass::fp;
namespace informed = lattice_glass::informed;

struct Options {
  int L = 10;
  long double beta = 3.0L;
  long double rho = 0.75L;
  long double rho1 = 0.30L;
  int warmup = 20000;
  int production = 200000;
  int sample_every = 4;
  unsigned int seed = 20260608u;
};

struct ArmResult {
  double seconds = 0.0;
  int samples = 0;
  long long attempts = 0;
  long long accepted = 0;
  long double accept_prob_sum = 0.0L;
  fp::AutocorrResult energy;
  fp::AutocorrResult bond;
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

enum class Mode { FullSwap, BlindOcc, InformedOcc };

static ArmResult run_arm(const Options &o, const std::vector<int> &nn,
                         int num_type1, int num_type2, Mode mode,
                         std::mt19937 &gen) {
  const int lat_size = o.L * o.L * o.L;
  std::vector<uint8_t> lattice =
      fp::random_lattice(num_type1, num_type2, lat_size, gen);
  informed::SiteLists lists = informed::build_site_lists(lattice);
  std::vector<int> m = informed::build_neighbor_counts(lattice, nn.data());
  std::vector<double> wbuf(lists.vac.size(), 0.0);
  const int np = static_cast<int>(lists.occ.size());

  auto step = [&]() -> informed::SweepStats {
    if (mode == Mode::FullSwap) {
      fp::nonlocal_swap_sweep(lattice, o.beta, 0.0L, nullptr, gen, nn.data());
      return {};
    }
    if (mode == Mode::BlindOcc)
      return informed::blind_occupancy_sweep(lattice, o.beta, np, lists, m, gen,
                                             nn.data());
    return informed::informed_occupancy_sweep(lattice, o.beta, np, lists, m,
                                              wbuf, gen, nn.data());
  };

  for (int s = 0; s < o.warmup; ++s)
    step();

  std::vector<long double> energy, bond;
  ArmResult result;
  const auto t0 = std::chrono::steady_clock::now();
  for (int s = 1; s <= o.production; ++s) {
    const informed::SweepStats st = step();
    result.attempts += st.attempts;
    result.accepted += st.accepted;
    result.accept_prob_sum += st.accept_prob_sum;
    if (s % o.sample_every == 0) {
      energy.push_back(fp::total_energy_int(lattice, nn.data()));
      bond.push_back(
          static_cast<long double>(informed::occupied_bond_count(lattice, nn.data())));
    }
  }
  const auto t1 = std::chrono::steady_clock::now();
  result.seconds = std::chrono::duration<double>(t1 - t0).count();
  result.samples = static_cast<int>(energy.size());
  result.energy = fp::integrated_autocorrelation_time(energy);
  result.bond = fp::integrated_autocorrelation_time(bond);
  return result;
}

static double ess_per_sec(const ArmResult &a, const fp::AutocorrResult &x) {
  const long double ess = fp::effective_samples(a.samples, x.tau_int);
  return a.seconds > 0.0 ? static_cast<double>(ess) / a.seconds : 0.0;
}

static void print_arm(const std::string &name, const ArmResult &a) {
  std::cout << name << ".seconds " << a.seconds << "\n";
  std::cout << name << ".samples " << a.samples << "\n";
  if (a.attempts > 0) {
    std::cout << name << ".occ_acceptance "
              << static_cast<double>(a.accepted) / a.attempts << "\n";
    std::cout << name << ".occ_mean_accept_prob "
              << static_cast<double>(a.accept_prob_sum / a.attempts) << "\n";
  }
  std::cout << name << ".energy_tau " << static_cast<double>(a.energy.tau_int)
            << " resolved " << (a.energy.resolved ? 1 : 0) << "\n";
  std::cout << name << ".bond_tau " << static_cast<double>(a.bond.tau_int)
            << " resolved " << (a.bond.resolved ? 1 : 0) << "\n";
  std::cout << name << ".energy_ess_per_sec " << ess_per_sec(a, a.energy)
            << "\n";
  std::cout << name << ".bond_ess_per_sec " << ess_per_sec(a, a.bond) << "\n";
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

    std::cout << std::setprecision(8);
    std::cout << "inf.L " << o.L << "\n";
    std::cout << "inf.beta " << static_cast<double>(o.beta) << "\n";
    std::cout << "inf.temperature " << static_cast<double>(1.0L / o.beta) << "\n";
    std::cout << "inf.production " << o.production << "\n";
    std::cout << "inf.sample_every " << o.sample_every << "\n";

    std::mt19937 full_gen(o.seed ^ 0x9E3779B9u);
    std::mt19937 blind_gen(o.seed ^ 0xC2B2AE35u);
    std::mt19937 informed_gen(o.seed ^ 0x1B56C4E9u);

    const ArmResult full =
        run_arm(o, nn, num_type1, num_type2, Mode::FullSwap, full_gen);
    print_arm("fullswap", full);
    const ArmResult blind =
        run_arm(o, nn, num_type1, num_type2, Mode::BlindOcc, blind_gen);
    print_arm("blind", blind);
    const ArmResult informed_arm =
        run_arm(o, nn, num_type1, num_type2, Mode::InformedOcc, informed_gen);
    print_arm("informed", informed_arm);

    // Per-sweep mixing speedup (blind and informed share the np-attempt sweep
    // unit and sample cadence, so the ratio of taus is the per-sweep gain).
    auto tau_ratio = [](const fp::AutocorrResult &b, const fp::AutocorrResult &i) {
      return i.tau_int > 0.0L ? static_cast<double>(b.tau_int / i.tau_int) : 0.0;
    };
    std::cout << "speedup.informed_vs_blind.bond_per_sweep "
              << tau_ratio(blind.bond, informed_arm.bond) << "\n";
    std::cout << "speedup.informed_vs_blind.energy_per_sweep "
              << tau_ratio(blind.energy, informed_arm.energy) << "\n";

    auto sec_ratio = [&](const ArmResult &b, const fp::AutocorrResult &bx,
                         const ArmResult &i, const fp::AutocorrResult &ix) {
      const double base = ess_per_sec(b, bx);
      return base > 0.0 ? ess_per_sec(i, ix) / base : 0.0;
    };
    std::cout << "speedup.informed_vs_blind.bond_ess_per_sec "
              << sec_ratio(blind, blind.bond, informed_arm, informed_arm.bond)
              << "\n";
    std::cout << "speedup.informed_vs_blind.energy_ess_per_sec "
              << sec_ratio(blind, blind.energy, informed_arm, informed_arm.energy)
              << "\n";
    std::cout << "speedup.informed_vs_fullswap.bond_ess_per_sec "
              << sec_ratio(full, full.bond, informed_arm, informed_arm.bond)
              << "\n";
    std::cout << "speedup.informed_vs_fullswap.energy_ess_per_sec "
              << sec_ratio(full, full.energy, informed_arm, informed_arm.energy)
              << "\n";
  } catch (const std::exception &error) {
    std::cerr << "informed_swap_efficiency: " << error.what() << "\n";
    return 1;
  }
  return 0;
}
