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
  double beta = 2.0;
  uint64_t sweeps = 1000000;
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
  int history_every = 100;
};

Options parse(int argc, char** argv) {
  Options o;
  for (int i=1;i<argc;i++) {
    const std::string k = argv[i];
    auto next = [&]() -> std::string {
      if (i+1 >= argc) throw std::runtime_error("missing value for " + k);
      return argv[++i];
    };
    if (k == "--run-dir") o.run_dir = next();
    else if (k == "--beta") o.beta = std::stod(next());
    else if (k == "--sweeps") o.sweeps = std::stoull(next());
    else if (k == "--sample-every") o.sample_every = std::stoi(next());
    else if (k == "--checkpoint-every") o.checkpoint_every = std::stoi(next());
    else if (k == "--local") o.local_pct = std::stoi(next());
    else if (k == "--seed") o.seed = std::stoull(next());
    else if (k == "--resume") o.resume = true;
    else if (k == "--validate-only") o.validate_only = true;
    else if (k == "--max-wall-seconds") o.max_wall_seconds = std::stod(next());
    else if (k == "--snapshot-first") o.snapshot_first = std::stoull(next());
    else if (k == "--snapshot-growth") o.snapshot_growth = std::stod(next());
    else if (k == "--max-snapshots") o.max_snapshots = std::stoi(next());
    else if (k == "--history-every") o.history_every = std::stoi(next());
    else throw std::runtime_error("unknown option " + k);
  }
  if (o.beta < 0.0) throw std::runtime_error("beta must be non-negative");
  return o;
}

void write_manifest(const Options& o) {
  std::filesystem::create_directories(o.run_dir);
  std::ofstream out(o.run_dir + "/manifest.txt");
  out << "L=" << lg::L << "\n";
  out << "beta=" << o.beta << "\n";
  out << "sweeps=" << o.sweeps << "\n";
  out << "sample_every=" << o.sample_every << "\n";
  out << "local_pct=" << o.local_pct << "\n";
  out << "seed=" << o.seed << "\n";
  out << "history_every=" << o.history_every << "\n";
}

struct HistoryHeader {
  uint64_t magic = 0x4c474831ULL; // LGH1
  uint32_t version = 1;
  double beta = 0.0;
};
struct HistoryRecord {
  uint64_t sweep = 0;
  int32_t energy = 0;
  int32_t accepted_since = 0;
  int64_t accepted_total = 0;
};

void prepare_history(const std::string& path, double beta, uint64_t sweeps_done,
                     bool resume) {
  if (!resume || !std::filesystem::exists(path)) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    HistoryHeader h; h.beta = beta;
    out.write(reinterpret_cast<const char*>(&h), sizeof(h));
    return;
  }
  std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
  if (!f) throw std::runtime_error("cannot open history for resume");
  HistoryHeader h;
  f.read(reinterpret_cast<char*>(&h), sizeof(h));
  if (!f || h.magic != 0x4c474831ULL || h.version != 1 ||
      std::abs(h.beta - beta) > 1e-12) {
    throw std::runtime_error("history header mismatch");
  }
  uint64_t keep = 0;
  HistoryRecord rec;
  while (f.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
    if (rec.sweep > sweeps_done) break;
    ++keep;
  }
  const auto end = static_cast<std::streamoff>(sizeof(HistoryHeader) +
                                               keep * sizeof(HistoryRecord));
  f.close();
  std::filesystem::resize_file(path, static_cast<uintmax_t>(end));
}

void append_history(std::ofstream& out, uint64_t sweep, int energy,
                    int accepted_since, int accepted_total) {
  HistoryRecord r;
  r.sweep = sweep;
  r.energy = energy;
  r.accepted_since = accepted_since;
  r.accepted_total = accepted_total;
  out.write(reinterpret_cast<const char*>(&r), sizeof(r));
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

    lg::SamplerConfig cfg;
    cfg.beta = o.beta;
    cfg.local_pct = o.local_pct;
    cfg.sample_every = o.sample_every;
    cfg.seed = o.seed;
    cfg.total_sweeps = o.sweeps;

    lg::Sampler sampler(cfg);
    const std::string ckpt = o.run_dir + "/checkpoint.bin";
    if (o.resume) {
      if (!lg::load_checkpoint(ckpt, sampler))
        if (!lg::load_checkpoint(ckpt + ".prev", sampler))
          throw std::runtime_error("resume requested but no valid checkpoint");
      sampler.cfg.total_sweeps = o.sweeps;
    }

    const std::string history_path = o.run_dir + "/history.bin";
    prepare_history(history_path, sampler.cfg.beta, sampler.sweeps_done, o.resume);
    std::ofstream history(history_path, std::ios::binary | std::ios::app);
    if (!history) throw std::runtime_error("cannot open history");

    std::signal(SIGTERM, on_signal);
    std::signal(SIGINT, on_signal);

    uint64_t next_snapshot = o.snapshot_first;
    int snapshots = 0;
    int accepted_since = 0;
    const auto start = std::chrono::steady_clock::now();

    while (sampler.sweeps_done < o.sweeps && !g_stop) {
      uint64_t chunk = static_cast<uint64_t>(std::max(1, o.checkpoint_every));
      if (chunk > 1000) chunk = 1000;
      chunk = std::min<uint64_t>(chunk, o.sweeps - sampler.sweeps_done);
      for (uint64_t i=0; i<chunk && !g_stop; ++i) {
        const int accepted = sampler.sweep();
        accepted_since += accepted;
        if (o.history_every > 0 && sampler.sweeps_done % o.history_every == 0) {
          append_history(history, sampler.sweeps_done, sampler.energy,
                         accepted_since, sampler.accepted_total);
          accepted_since = 0;
          history.flush();
        }
      }

      lg::save_checkpoint(ckpt, sampler);

      if (sampler.sweeps_done >= next_snapshot && snapshots < o.max_snapshots) {
        const std::string sp = o.run_dir + "/snapshots/snapshot_" +
                               std::to_string(sampler.sweeps_done) + ".bin";
        lg::save_checkpoint(sp, sampler);
        ++snapshots;
        next_snapshot = static_cast<uint64_t>(
            std::max<double>(next_snapshot + 1, next_snapshot * o.snapshot_growth));
      }

      const double elapsed = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - start).count();
      std::cout << "sweeps=" << sampler.sweeps_done
                << " beta=" << sampler.cfg.beta
                << " energy=" << sampler.energy
                << " samples=" << sampler.samples_done
                << " elapsed=" << elapsed << "\n" << std::flush;

      if (o.max_wall_seconds > 0 && elapsed >= o.max_wall_seconds) break;
    }

    if (accepted_since != 0 ||
        (o.history_every > 0 && sampler.sweeps_done % o.history_every != 0)) {
      append_history(history, sampler.sweeps_done, sampler.energy,
                     accepted_since, sampler.accepted_total);
    }
    history.flush();
    history.close();

    lg::save_checkpoint(ckpt, sampler);
    lg::save_configs(o.run_dir + "/configs", sampler);
    lg::save_histograms(o.run_dir + "/hist.csv", sampler);

    std::ofstream done(o.run_dir + "/done.txt");
    done << "sweeps_done=" << sampler.sweeps_done << "\n";
    done << "total_sweeps=" << o.sweeps << "\n";
    done << "samples_done=" << sampler.samples_done << "\n";
    done << "energy=" << sampler.energy << "\n";
    done << "complete=" << (sampler.sweeps_done >= o.sweeps ? 1 : 0) << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "lattice_single: " << e.what() << "\n";
    return 1;
  }
}
