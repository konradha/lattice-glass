#include "../lattice.hpp"

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>

using namespace lg;

static int failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::printf("FAIL: %s\n", msg); ++failures; } } while (0)

static void test_kernel() {
  CHECK(verify_kernel(20000, 12345), "fast kernel exact dE");
  State S(777);
  Rng r(888);
  for (int c=0;c<2000;c++) {
    int a = (int)r.bounded(N), b = (int)r.bounded(N);
    if (a == b) continue;
    int e0 = S.full_energy();
    Aff aff[14]; int na = 0;
    int dE = delta_generic(S, a, b, aff, na);
    commit_generic(S, aff, na);
    CHECK(S.full_energy() - e0 == dE, "generic kernel exact dE");
    delta_generic(S, a, b, aff, na); commit_generic(S, aff, na);
    CHECK(S.full_energy() == e0, "generic kernel restores state");
  }
  int c1=0,c2=0;
  for (auto v : S.st) { if ((v&3)==1) c1++; else if ((v&3)==2) c2++; }
  CHECK(c1 == N1 && c2 == N2, "composition preserved");
}

static void test_local_moves() {
  State S(901);
  Rng r(902);
  ExpTable table(2.5);
  int energy = S.full_energy();
  for (int i=0;i<2000;i++) {
    try_move(S, r, table, 100, energy);
    if (i % 100 == 0) CHECK(energy == S.full_energy(), "incremental energy matches full");
  }
  CHECK(energy == S.full_energy(), "local move energy");
}

static PtLadder make_ladder(uint64_t seed) {
  PtConfig c;
  c.ntemps = 3;
  c.beta_min = 0.5;
  c.beta_max = 2.0;
  c.exchange_every = 2;
  c.sample_every = 2;
  c.hist_bins = 2048;
  c.hist_width = 128;
  c.seed = seed;
  c.total_sweeps = 1000;
  c.local_pct = 50;
  return PtLadder(c);
}

static void test_checkpoint() {
  auto dir = std::filesystem::temp_directory_path() / "lg_cpu_test";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const std::string path = (dir / "checkpoint.bin").string();

  PtLadder a = make_ladder(4242);
  for (int i=0;i<40;i++) a.sweep();
  save_checkpoint(path, a);

  PtLadder b([]{ PtConfig c; c.ntemps=1; c.beta_min=0.1; c.beta_max=0.1; return PtLadder(c); }());
  CHECK(load_checkpoint(path, b), "checkpoint loads");
  CHECK(a.sweeps_done == b.sweeps_done, "checkpoint sweeps");
  CHECK(a.reps.size() == b.reps.size(), "checkpoint ntemps");
  for (size_t i=0;i<a.reps.size();i++) {
    CHECK(a.reps[i].state.st == b.reps[i].state.st, "checkpoint state");
    CHECK(a.reps[i].energy == b.reps[i].energy, "checkpoint energy");
    CHECK(std::memcmp(a.reps[i].rng.s, b.reps[i].rng.s, sizeof(a.reps[i].rng.s)) == 0,
          "checkpoint rng");
    CHECK(a.reps[i].hist == b.reps[i].hist, "checkpoint hist");
  }

  for (int i=0;i<40;i++) { a.sweep(); b.sweep(); }
  for (size_t i=0;i<a.reps.size();i++) {
    CHECK(a.reps[i].state.st == b.reps[i].state.st, "resume trajectory");
    CHECK(a.reps[i].energy == b.reps[i].energy, "resume energy");
  }

  // Corrupt one payload byte and ensure the checksum rejects it.
  {
    std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
    f.seekp(static_cast<std::streamoff>(sizeof(CheckpointHeader) + 100));
    char c; f.seekg(static_cast<std::streamoff>(sizeof(CheckpointHeader) + 100)); f.get(c);
    c ^= 0x5a;
    f.seekp(static_cast<std::streamoff>(sizeof(CheckpointHeader) + 100));
    f.put(c); f.close();
  }
  CHECK(!load_checkpoint(path, b), "corrupt checkpoint rejected");
  std::filesystem::remove_all(dir);
}

static void test_pt_and_reweight() {
  PtConfig c;
  c.ntemps = 4;
  c.beta_min = 0.5;
  c.beta_max = 2.0;
  c.exchange_every = 5;
  c.sample_every = 5;
  c.hist_bins = 4096;
  c.hist_width = 64;
  c.seed = 991;
  c.total_sweeps = 2000;
  c.local_pct = 50;
  PtLadder lad(c);
  for (int i=0;i<2000;i++) lad.sweep();
  CHECK(lad.sweeps_done == 2000, "pt sweeps");
  for (int i=0;i<c.ntemps;i++) {
    CHECK(lad.reps[i].energy == lad.reps[i].state.full_energy(), "pt incremental energy");
    auto w = lad.reweight(0, lad.reps[i].beta);
    double sum = 0; for (double v : w) sum += v;
    CHECK(std::abs(sum - 1.0) < 1e-9, "reweight normalization");
    for (double v : w) CHECK(std::isfinite(v), "reweight finite");
  }
}

int main() {
  test_kernel();
  test_local_moves();
  test_checkpoint();
  test_pt_and_reweight();
  if (failures == 0) std::printf("ALL TESTS PASS\n");
  else std::printf("%d FAILURES\n", failures);
  return failures == 0 ? 0 : 1;
}
