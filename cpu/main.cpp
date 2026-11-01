#include "lattice.hpp"

#include <chrono>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>

namespace {
volatile std::sig_atomic_t g_stop = 0;
void on_signal(int) { g_stop = 1; }

struct Options {
  std::string run_dir = "run";
  int ntemps = 8;
  double beta_min = 0.1;
  double beta_max = 3.5;
  uint64_t sweeps = 1000000;
  int exchange_every = 10;
  int sample_every = 10;
  int checkpoint_every = 1000;
  int local_pct = 50;
  uint64_t seed = 1;
  bool resume = false;
  bool validate_only = false;
  double max_wall_seconds = 0.0;
  uint64_t snapshot_first = 10000;
  double snapshot_growth = 1.15;
  int max_snapshots = 200;
};

uint64_t parse_u64(const std::string& s) { return std::stoull(s); }
double parse_d(const std::string& s) { return std::stod(s); }

Options parse(int argc, char** argv) {
  Options o;
  for (int i=1;i<argc;i++) {
    const std::string k = argv[i];
    auto next = [&]() -> std::string {
      if (i+1 >= argc) throw std::runtime_error("missing value for " + k);
      return argv[++i];
    };
    if (k == "--run-dir") o.run_dir = next();
    else if (k == "--ntemps") o.ntemps = std::stoi(next());
    else if (k == "--beta-min") o.beta_min = parse_d(next());
    else if (k == "--beta-max") o.beta_max = parse_d(next());
    else if (k == "--sweeps") o.sweeps = parse_u64(next());
    else if (k == "--exchange-every") o.exchange_every = std::stoi(next());
    else if (k == "--sample-every") o.sample_every = std::stoi(next());
    else if (k == "--checkpoint-every") o.checkpoint_every = std::stoi(next());
    else if (k == "--local") o.local_pct = std::stoi(next());
    else if (k == "--seed") o.seed = parse_u64(next());
    else if (k == "--resume") o.resume = true;
    else if (k == "--validate-only") o.validate_only = true;
    else if (k == "--max-wall-seconds") o.max_wall_seconds = parse_d(next());
    else if (k == "--snapshot-first") o.snapshot_first = parse_u64(next());
    else if (k == "--snapshot-growth") o.snapshot_growth = parse_d(next());
    else if (k == "--max-snapshots") o.max_snapshots = std::stoi(next());
    else throw std::runtime_error("unknown option " + k);
  }
  return o;
}

void write_manifest(const Options& o) {
  std::filesystem::create_directories(o.run_dir);
  std::ofstream out(o.run_dir + "/manifest.txt");
  out << "L=" << lg::L << "\n";
  out << "ntemps=" << o.ntemps << "\n";
  out << "beta_min=" << o.beta_min << "\n";
  out << "beta_max=" << o.beta_max << "\n";
  out << "sweeps=" << o.sweeps << "\n";
  out << "exchange_every=" << o.exchange_every << "\n";
  out << "sample_every=" << o.sample_every << "\n";
  out << "local_pct=" << o.local_pct << "\n";
  out << "seed=" << o.seed << "\n";
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Options o = parse(argc, argv);
    if (o.validate_only) {
      const bool ok = lg::verify_kernel(20000, o.seed);
      std::cout << "kernel_verify=" << (ok ? "PASS" : "FAIL") << "\n";
      return ok ? 0 : 1;
    }
    std::filesystem::create_directories(o.run_dir);
    std::filesystem::create_directories(o.run_dir + "/snapshots");
    std::filesystem::create_directories(o.run_dir + "/configs");
    write_manifest(o);

    lg::PtConfig cfg;
    cfg.ntemps = o.ntemps;
    cfg.beta_min = o.beta_min;
    cfg.beta_max = o.beta_max;
    cfg.local_pct = o.local_pct;
    cfg.exchange_every = o.exchange_every;
    cfg.sample_every = o.sample_every;
    cfg.seed = o.seed;
    cfg.total_sweeps = o.sweeps;

    lg::PtLadder ladder(cfg);
    const std::string ckpt = o.run_dir + "/checkpoint.bin";
    if (o.resume) {
      if (!lg::load_checkpoint(ckpt, ladder))
        if (!lg::load_checkpoint(ckpt + ".prev", ladder))
          throw std::runtime_error("resume requested but no valid checkpoint");
      ladder.cfg.total_sweeps = o.sweeps;
    }

    std::signal(SIGTERM, on_signal);
    std::signal(SIGINT, on_signal);

    uint64_t next_snapshot = o.snapshot_first;
    int snapshots = 0;
    const auto start = std::chrono::steady_clock::now();

    while (ladder.sweeps_done < o.sweeps && !g_stop) {
      uint64_t chunk = static_cast<uint64_t>(std::max(1, o.checkpoint_every));
      if (chunk > 1000) chunk = 1000; // prompt signal/preemption response
      chunk = std::min<uint64_t>(chunk, o.sweeps - ladder.sweeps_done);
      for (uint64_t i=0; i<chunk && !g_stop; ++i) ladder.sweep();

      lg::save_checkpoint(ckpt, ladder);

      if (ladder.sweeps_done >= next_snapshot && snapshots < o.max_snapshots) {
        const std::string sp = o.run_dir + "/snapshots/snapshot_" +
                               std::to_string(ladder.sweeps_done) + ".bin";
        lg::save_checkpoint(sp, ladder);
        ++snapshots;
        next_snapshot = static_cast<uint64_t>(
            std::max<double>(next_snapshot + 1, next_snapshot * o.snapshot_growth));
      }

      const double elapsed = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - start).count();
      std::cout << "sweeps=" << ladder.sweeps_done
                << " cold_beta=" << ladder.reps.back().beta
                << " cold_energy=" << ladder.reps.back().energy
                << " samples=" << ladder.samples_done
                << " elapsed=" << elapsed << "\n" << std::flush;

      if (o.max_wall_seconds > 0 && elapsed >= o.max_wall_seconds) break;
    }

    lg::save_checkpoint(ckpt, ladder);
    lg::save_configs(o.run_dir + "/configs", ladder);
    lg::save_histograms(o.run_dir + "/hist.csv", ladder);

    std::ofstream done(o.run_dir + "/done.txt");
    done << "sweeps_done=" << ladder.sweeps_done << "\n";
    done << "total_sweeps=" << o.sweeps << "\n";
    done << "samples_done=" << ladder.samples_done << "\n";
    done << "complete=" << (ladder.sweeps_done >= o.sweeps ? 1 : 0) << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "lattice_pt: " << e.what() << "\n";
    return 1;
  }
}
