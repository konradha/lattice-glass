// Checkpointed parallel-tempering production sampler for the NH lattice glass.
//
// One process runs ONE PT ladder across a beta-grid (so it emits equilibrium
// configurations at EVERY temperature, and the warm replicas equilibrate the
// cold one). Launch many independent ranks (job array) to multiply the config
// count -- the ensemble is embarrassingly parallel, which is what makes a
// fixed-wall HPC allocation usable.
//
// Restart-safe for a hard wall cap: state (every replica's lattice, every RNG,
// sweep counter, and all configs collected so far) is checkpointed to a single
// binary file every --checkpoint-secs and once more when --wall-secs is about
// to be hit. Re-launching with the same --ckpt resumes exactly where it left
// off. Per-temperature .npy files (configs x sites, uint8) are (re)written at
// each checkpoint so partial output is always usable by RSMI-NE.
//
// Moves: the trusted all-pairs non-local swap (fp::nonlocal_swap_sweep) per
// replica + Metropolis replica exchanges between adjacent betas. Composition is
// conserved by construction. Equilibration is the user's responsibility to
// verify (energy-vs-sweep is logged per checkpoint, and --warmup is discarded);
// the cold point must be checked before trusting its configs.

#include "../fp_sampler.h"
#include "../npy.hpp"
#include "../informed_swap.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fp = lattice_glass::fp;
namespace cluster = lattice_glass::cluster;
namespace informed = lattice_glass::informed;

struct Options {
  int L = 16;
  long double rho = 0.75L, rho1 = 0.30L;
  double beta_min = 0.3, beta_max = 3.0;
  int n_temps = 10;
  long long warmup = 200000;       // sweeps discarded per replica before sampling
  long long sample_every = 2000;   // sweeps between collected configs (~2 tau_a)
  int configs_per_temp = 10000;
  int exchange_every = 10;         // sweeps between PT exchange rounds
  double checkpoint_secs = 1800;   // wall between checkpoints
  double wall_secs = 6.8 * 86400;  // stop+checkpoint before this (7-day cap, margin)
  unsigned seed = 1;
  std::string out_prefix = "configs";
  std::string ckpt = "ckpt.bin";
};

static Options parse(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto v = [&]() { return std::string(argv[++i]); };
    if (i + 1 >= argc) throw std::invalid_argument("missing value for " + k);
    if (k == "--L") o.L = std::stoi(v());
    else if (k == "--rho") o.rho = std::stold(v());
    else if (k == "--rho1") o.rho1 = std::stold(v());
    else if (k == "--beta-min") o.beta_min = std::stod(v());
    else if (k == "--beta-max") o.beta_max = std::stod(v());
    else if (k == "--n-temps") o.n_temps = std::stoi(v());
    else if (k == "--warmup") o.warmup = std::stoll(v());
    else if (k == "--sample-every") o.sample_every = std::stoll(v());
    else if (k == "--configs-per-temp") o.configs_per_temp = std::stoi(v());
    else if (k == "--exchange-every") o.exchange_every = std::stoi(v());
    else if (k == "--checkpoint-secs") o.checkpoint_secs = std::stod(v());
    else if (k == "--wall-secs") o.wall_secs = std::stod(v());
    else if (k == "--seed") o.seed = static_cast<unsigned>(std::stoul(v()));
    else if (k == "--out-prefix") o.out_prefix = v();
    else if (k == "--ckpt") o.ckpt = v();
    else throw std::invalid_argument("unknown option " + k);
  }
  return o;
}

struct State {
  std::vector<double> betas;                       // per replica
  std::vector<std::vector<uint8_t>> lattice;       // per replica
  std::vector<std::mt19937> rng;                   // per replica
  std::mt19937 exch_rng;
  long long sweeps = 0;                            // sweeps done per replica
  std::vector<std::vector<uint8_t>> configs;       // [temp] flattened configs
  std::vector<int> collected;                      // per temp
  long long exch_attempts = 0, exch_accepts = 0;
};

static void write_pod(std::ofstream &f, const void *p, size_t n) {
  f.write(reinterpret_cast<const char *>(p), n);
}
static void read_pod(std::ifstream &f, void *p, size_t n) {
  f.read(reinterpret_cast<char *>(p), n);
}
static void write_rng(std::ofstream &f, std::mt19937 &g) {
  std::ostringstream ss; ss << g; std::string s = ss.str();
  uint32_t len = static_cast<uint32_t>(s.size());
  write_pod(f, &len, 4); f.write(s.data(), len);
}
static void read_rng(std::ifstream &f, std::mt19937 &g) {
  uint32_t len; read_pod(f, &len, 4);
  std::string s(len, '\0'); f.read(&s[0], len);
  std::istringstream ss(s); ss >> g;
}

static const uint32_t kMagic = 0x4E484C47; // "NHLG"

static void checkpoint(const Options &o, State &s, const std::string &tmp) {
  std::ofstream f(tmp, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open checkpoint " + tmp);
  uint32_t magic = kMagic; int L = o.L, nt = o.n_temps;
  write_pod(f, &magic, 4); write_pod(f, &L, 4); write_pod(f, &nt, 4);
  write_pod(f, &s.sweeps, 8);
  write_pod(f, &s.exch_attempts, 8); write_pod(f, &s.exch_accepts, 8);
  for (int t = 0; t < nt; ++t) {
    write_pod(f, &s.betas[t], 8);
    write_pod(f, s.lattice[t].data(), s.lattice[t].size());
    write_rng(f, s.rng[t]);
    int c = s.collected[t]; write_pod(f, &c, 4);
    write_pod(f, s.configs[t].data(), s.configs[t].size());
  }
  write_rng(f, s.exch_rng);
  f.flush(); f.close();
  std::rename(tmp.c_str(), o.ckpt.c_str()); // atomic replace
}

static bool restore(const Options &o, State &s) {
  std::ifstream f(o.ckpt, std::ios::binary);
  if (!f) return false;
  uint32_t magic; int L, nt;
  read_pod(f, &magic, 4); read_pod(f, &L, 4); read_pod(f, &nt, 4);
  if (magic != kMagic || L != o.L || nt != o.n_temps)
    throw std::runtime_error("checkpoint mismatch (L/n_temps/magic)");
  const int sites = o.L * o.L * o.L;
  read_pod(f, &s.sweeps, 8);
  read_pod(f, &s.exch_attempts, 8); read_pod(f, &s.exch_accepts, 8);
  s.betas.resize(nt); s.lattice.assign(nt, {}); s.rng.resize(nt);
  s.configs.assign(nt, {}); s.collected.assign(nt, 0);
  for (int t = 0; t < nt; ++t) {
    read_pod(f, &s.betas[t], 8);
    s.lattice[t].resize(sites); read_pod(f, s.lattice[t].data(), sites);
    read_rng(f, s.rng[t]);
    int c; read_pod(f, &c, 4); s.collected[t] = c;
    s.configs[t].resize(static_cast<size_t>(c) * sites);
    read_pod(f, s.configs[t].data(), s.configs[t].size());
  }
  read_rng(f, s.exch_rng);
  return static_cast<bool>(f);
}

static void dump_npy(const Options &o, const State &s) {
  const unsigned long sites = static_cast<unsigned long>(o.L) * o.L * o.L;
  for (int t = 0; t < o.n_temps; ++t) {
    if (s.collected[t] == 0) continue;
    char name[512];
    std::snprintf(name, sizeof(name), "%s_seed%u_T%.4f.npy", o.out_prefix.c_str(),
                  o.seed, 1.0 / s.betas[t]);
    const unsigned long shape[2] = {static_cast<unsigned long>(s.collected[t]), sites};
    npy::SaveArrayAsNumpy(std::string(name), false, 2, shape, s.configs[t]);
  }
}

int main(int argc, char **argv) {
  try {
    const Options o = parse(argc, argv);
    const int sites = o.L * o.L * o.L;
    const int np = static_cast<int>(o.rho * sites);
    const int n1 = static_cast<int>(o.rho1 * sites);
    const int n2 = np - n1;
    if (n1 < 0 || n2 < 0 || np > sites) throw std::invalid_argument("bad composition");
    const std::vector<int> nn = fp::cubic_neighbors(o.L);

    State s;
    const bool resumed = restore(o, s);
    if (!resumed) {
      s.betas.resize(o.n_temps);
      s.lattice.resize(o.n_temps);
      s.rng.resize(o.n_temps);
      s.configs.assign(o.n_temps, {});
      s.collected.assign(o.n_temps, 0);
      s.exch_rng.seed(o.seed ^ 0xABCDEF01u);
      for (int t = 0; t < o.n_temps; ++t) {
        s.betas[t] = o.beta_min + (o.beta_max - o.beta_min) * t / (o.n_temps - 1);
        s.rng[t].seed(o.seed * 2654435761u + 7919u * t + 1u);
        s.lattice[t] = fp::random_lattice(n1, n2, sites, s.rng[t]);
      }
    }
    std::printf("[rank seed=%u] %s at sweeps=%lld (L=%d, %d temps, T in [%.3f,%.3f])\n",
                o.seed, resumed ? "RESUMED" : "fresh", s.sweeps, o.L, o.n_temps,
                1.0 / s.betas.back(), 1.0 / s.betas.front());
    std::fflush(stdout);

    // Occupied-neighbour counts (rebuilt from lattices; derivable, so not part
    // of the checkpoint) and per-replica integer-dE exp tables for the fast swap.
    std::vector<std::vector<int>> m(o.n_temps);
    for (int t = 0; t < o.n_temps; ++t)
      m[t] = informed::build_neighbor_counts(s.lattice[t], nn.data());
    const int kOff = 256;
    std::vector<std::vector<double>> etab(o.n_temps, std::vector<double>(kOff + 1));
    for (int t = 0; t < o.n_temps; ++t)
      for (int d = 0; d <= kOff; ++d)
        etab[t][d] = std::exp(-s.betas[t] * d);
    auto t_start = std::chrono::steady_clock::now();
    auto last_ckpt = t_start;
    auto elapsed = [&]() {
      return std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    };

    const std::string tmp = o.ckpt + ".tmp";
    bool done = false;
    while (!done) {
      // Run exchange_every sweeps per replica, then one PT exchange round.
      for (int t = 0; t < o.n_temps; ++t)
        for (int k = 0; k < o.exchange_every; ++k)
          informed::fast_swap_sweep(s.lattice[t], m[t], (long double)s.betas[t],
                                    sites, etab[t].data(), kOff, s.rng[t],
                                    nn.data());
      s.sweeps += o.exchange_every;

      // Replica exchange on alternating adjacent pairs.
      std::uniform_real_distribution<double> uni(0.0, 1.0);
      const int parity = static_cast<int>((s.sweeps / o.exchange_every) & 1);
      for (int t = parity; t + 1 < o.n_temps; t += 2) {
        const double ea = (double)fp::total_energy_int(s.lattice[t], nn.data());
        const double eb = (double)fp::total_energy_int(s.lattice[t + 1], nn.data());
        const double arg = (s.betas[t] - s.betas[t + 1]) * (ea - eb);
        ++s.exch_attempts;
        if (arg >= 0.0 || uni(s.exch_rng) < std::exp(arg)) {
          std::swap(s.lattice[t], s.lattice[t + 1]); // exchange configs + counts
          std::swap(m[t], m[t + 1]);
          ++s.exch_accepts;
        }
      }

      // Sample once past warmup, on the sampling cadence.
      if (s.sweeps >= o.warmup && (s.sweeps % o.sample_every) < o.exchange_every) {
        for (int t = 0; t < o.n_temps; ++t)
          if (s.collected[t] < o.configs_per_temp) {
            s.configs[t].insert(s.configs[t].end(), s.lattice[t].begin(), s.lattice[t].end());
            ++s.collected[t];
          }
      }

      done = true;
      for (int t = 0; t < o.n_temps; ++t)
        if (s.collected[t] < o.configs_per_temp) done = false;

      const bool wall_hit = elapsed() > o.wall_secs;
      const bool ckpt_due =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - last_ckpt).count()
          > o.checkpoint_secs;
      if (done || wall_hit || ckpt_due) {
        checkpoint(o, s, tmp);
        dump_npy(o, s);
        last_ckpt = std::chrono::steady_clock::now();
        const double ecold = (double)fp::total_energy_int(s.lattice.front(), nn.data()) / sites;
        std::printf("[seed=%u] sweeps=%lld collected[cold]=%d/%d E_cold/site=%.5f "
                    "exch_acc=%.3f wall=%.0fs%s\n",
                    o.seed, s.sweeps, s.collected.front(), o.configs_per_temp, ecold,
                    s.exch_attempts ? (double)s.exch_accepts / s.exch_attempts : 0.0,
                    elapsed(), done ? " DONE" : (wall_hit ? " WALL-STOP" : ""));
        std::fflush(stdout);
        if (wall_hit && !done) break;
      }
    }
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "production_sampler: %s\n", e.what());
    return 1;
  }
}
