// Reusable, version- and endianness-guarded checkpoint machinery for the NH
// lattice-glass parallel-tempering production sampler.
//
// One process runs ONE PT ladder; its full state (every replica's lattice, every
// RNG, the sweep counter, and all configs collected so far) serializes to a single
// binary file. The on-disk layout is the historical sampler layout plus a small
// version + endianness header so a checkpoint can never be silently mis-read on a
// host with a different build (version bump) or byte order (the sentinel only
// validates on the little-endian hosts this raw-POD format supports).
//
// Header (little-endian POD): magic(4), kVersion(4), kEndianSentinel(4), L(4),
// n_temps(4), sweeps(8), exch_attempts(8), exch_accepts(8); then per temp t:
// beta(8), lattice bytes (L*L*L), rng (length-prefixed text), collected(4),
// configs bytes (collected*L*L*L); then exch_rng.

#ifndef LATTICE_GLASS_PRODUCTION_CHECKPOINT_H
#define LATTICE_GLASS_PRODUCTION_CHECKPOINT_H

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace lattice_glass {
namespace prod {

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

inline void write_pod(std::ofstream &f, const void *p, size_t n) {
  f.write(reinterpret_cast<const char *>(p), n);
}
inline void read_pod(std::ifstream &f, void *p, size_t n) {
  f.read(reinterpret_cast<char *>(p), n);
}
inline void write_rng(std::ofstream &f, const std::mt19937 &g) {
  std::ostringstream ss; ss << g; std::string s = ss.str();
  uint32_t len = static_cast<uint32_t>(s.size());
  write_pod(f, &len, 4); f.write(s.data(), len);
}
inline void read_rng(std::ifstream &f, std::mt19937 &g) {
  uint32_t len; read_pod(f, &len, 4);
  std::string s(len, '\0'); f.read(&s[0], len);
  std::istringstream ss(s); ss >> g;
}

constexpr uint32_t kMagic = 0x4E484C47;          // "NHLG"
constexpr uint32_t kVersion = 1;
constexpr uint32_t kEndianSentinel = 0x01020304u;

// Atomically write the full state to ckpt_path: build ckpt_path + ".tmp" then
// std::rename it into place (rename is atomic on POSIX).
inline void checkpoint(const State &s, int L, int n_temps, const std::string &ckpt_path) {
  const std::string tmp = ckpt_path + ".tmp";
  std::ofstream f(tmp, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open checkpoint " + tmp);
  uint32_t magic = kMagic, version = kVersion, endian = kEndianSentinel;
  int Lw = L, nt = n_temps;
  write_pod(f, &magic, 4); write_pod(f, &version, 4); write_pod(f, &endian, 4);
  write_pod(f, &Lw, 4); write_pod(f, &nt, 4);
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
  std::rename(tmp.c_str(), ckpt_path.c_str()); // atomic replace
}

// Restore state from ckpt_path. Returns false if the file cannot be opened
// (fresh start). Throws std::runtime_error on magic/version/endianness or
// L/n_temps mismatch. Otherwise reads every field, resizing State to match, and
// returns the stream's good() state.
inline bool restore(State &s, int L, int n_temps, const std::string &ckpt_path) {
  std::ifstream f(ckpt_path, std::ios::binary);
  if (!f) return false;
  uint32_t magic, version, endian;
  read_pod(f, &magic, 4);
  if (magic != kMagic) throw std::runtime_error("checkpoint magic mismatch");
  read_pod(f, &version, 4);
  if (version != kVersion) throw std::runtime_error("checkpoint version mismatch");
  read_pod(f, &endian, 4);
  if (endian != kEndianSentinel)
    throw std::runtime_error("checkpoint endianness mismatch (little-endian hosts only)");
  int Lr, nt;
  read_pod(f, &Lr, 4); read_pod(f, &nt, 4);
  if (Lr != L || nt != n_temps)
    throw std::runtime_error("checkpoint L/n_temps mismatch");
  const int sites = L * L * L;
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

} // namespace prod
} // namespace lattice_glass

#endif // LATTICE_GLASS_PRODUCTION_CHECKPOINT_H
