// Round-trip + corruption-rejection test for the portable production checkpoint.
//
// Verifies that prod::checkpoint -> prod::restore preserves every field byte-for-
// byte (lattices, configs, counters, betas, collected) AND restores each RNG to a
// state that produces the same next draw, and that restore() rejects a checkpoint
// whose magic, version, or endianness sentinel has been corrupted while returning
// false (not throwing) for a missing file.
//
// Compiled with -UNDEBUG so assert() is live.

#include "../experiments/production_checkpoint.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace prod = lattice_glass::prod;

// Copy src verbatim to dst, then overwrite the 4 bytes at byte offset `off` with
// `val` (little-endian POD) using an in/out binary stream + seekp.
static void copy_and_patch(const char *src, const char *dst, long off, uint32_t val) {
  {
    std::ifstream in(src, std::ios::binary);
    std::ofstream out(dst, std::ios::binary);
    out << in.rdbuf();
  }
  std::ofstream patch(dst, std::ios::binary | std::ios::in | std::ios::out);
  assert(patch);
  patch.seekp(off);
  patch.write(reinterpret_cast<const char *>(&val), 4);
}

static bool restore_throws(const char *path, int L, int nt) {
  prod::State c;
  try {
    prod::restore(c, L, nt, path);
  } catch (const std::exception &) {
    return true;
  }
  return false;
}

int main() {
  const int L = 6, nt = 4, sites = 216; // 6^3 = 216

  // --- Build a fully-populated source state -------------------------------
  prod::State a;
  a.betas = {0.3, 1.0, 2.0, 3.0};
  a.lattice.resize(nt);
  a.rng.resize(nt);
  a.configs.resize(nt);
  a.collected = {2, 1, 0, 3};
  for (int t = 0; t < nt; ++t) {
    a.rng[t].seed(1000u + static_cast<unsigned>(t));
    a.rng[t].discard(50); // advance to a non-initial state
    a.lattice[t].resize(sites);
    for (int i = 0; i < sites; ++i)
      a.lattice[t][i] = static_cast<uint8_t>((t * 31 + i) % 3);
    a.configs[t].resize(static_cast<size_t>(a.collected[t]) * sites);
    for (size_t i = 0; i < a.configs[t].size(); ++i)
      a.configs[t][i] = static_cast<uint8_t>((t * 17 + i * 7) % 5);
  }
  a.sweeps = 12345;
  a.exch_attempts = 777;
  a.exch_accepts = 321;
  a.exch_rng.seed(424242u);
  a.exch_rng.discard(13);

  // --- Round-trip ---------------------------------------------------------
  const char *path = "/tmp/nhlg_ckpt_test.bin";
  prod::checkpoint(a, L, nt, path);

  prod::State b;
  const bool ok = prod::restore(b, L, nt, path);
  assert(ok);

  assert(b.sweeps == a.sweeps);
  assert(b.exch_attempts == a.exch_attempts);
  assert(b.exch_accepts == a.exch_accepts);
  assert(b.betas == a.betas);
  assert(b.collected == a.collected);
  for (int t = 0; t < nt; ++t) {
    assert(b.lattice[t] == a.lattice[t]);
    assert(b.configs[t] == a.configs[t]);
  }

  // RNG equivalence: compare next draws on copies so originals are untouched.
  for (int t = 0; t < nt; ++t) {
    std::mt19937 ca = a.rng[t], cb = b.rng[t];
    assert(ca() == cb());
  }
  {
    std::mt19937 ca = a.exch_rng, cb = b.exch_rng;
    assert(ca() == cb());
  }

  // --- Corruption rejection ----------------------------------------------
  // Header byte offsets: magic@0, kVersion@4, kEndianSentinel@8.
  const char *bad_magic = "/tmp/nhlg_ckpt_badmagic.bin";
  const char *bad_ver = "/tmp/nhlg_ckpt_badver.bin";
  const char *bad_endian = "/tmp/nhlg_ckpt_badendian.bin";

  copy_and_patch(path, bad_magic, 0, 0x12345678u);   // != kMagic
  copy_and_patch(path, bad_ver, 4, 0xDEADBEEFu);     // != kVersion
  copy_and_patch(path, bad_endian, 8, 0x04030201u);  // != kEndianSentinel (byte-swapped)

  assert(restore_throws(bad_magic, L, nt));
  assert(restore_throws(bad_ver, L, nt));
  assert(restore_throws(bad_endian, L, nt));

  // Missing file: returns false, does not throw.
  {
    prod::State d;
    bool threw = false, ret = true;
    try {
      ret = prod::restore(d, L, nt, "/tmp/nhlg_ckpt_does_not_exist_xyz.bin");
    } catch (const std::exception &) {
      threw = true;
    }
    assert(!threw);
    assert(!ret);
  }

  // --- Cleanup ------------------------------------------------------------
  std::remove(path);
  std::remove(bad_magic);
  std::remove(bad_ver);
  std::remove(bad_endian);

  std::printf("checkpoint portability tests passed\n");
  return 0;
}
