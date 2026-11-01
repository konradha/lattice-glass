#include "../lattice.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>

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
  ExpTable table(2.0);
  int energy = S.full_energy();
  for (int i=0;i<2000;i++) {
    try_move(S, r, table, 100, energy);
    if (i % 100 == 0) CHECK(energy == S.full_energy(), "incremental energy matches full");
  }
  CHECK(energy == S.full_energy(), "local move energy");
}

static Sampler make_sampler() {
  SamplerConfig c;
  c.beta = 2.0;
  c.local_pct = 50;
  c.sample_every = 2;
  c.hist_bins = 2048;
  c.hist_width = 128;
  c.seed = 4242;
  c.total_sweeps = 1000;
  return Sampler(c);
}

static void test_checkpoint() {
  auto dir = std::filesystem::temp_directory_path() / "lg_single_test";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const std::string path = (dir / "checkpoint.bin").string();

  Sampler a = make_sampler();
  for (int i=0;i<40;i++) a.sweep();
  save_checkpoint(path, a);

  Sampler b = make_sampler();
  CHECK(load_checkpoint(path, b), "checkpoint loads");
  CHECK(a.sweeps_done == b.sweeps_done, "checkpoint sweeps");
  CHECK(a.state.st == b.state.st, "checkpoint state");
  CHECK(a.energy == b.energy, "checkpoint energy");
  CHECK(std::memcmp(a.rng.s, b.rng.s, sizeof(a.rng.s)) == 0, "checkpoint rng");
  CHECK(a.hist == b.hist, "checkpoint hist");
  CHECK(a.cfg.beta == b.cfg.beta, "checkpoint beta");

  for (int i=0;i<40;i++) { a.sweep(); b.sweep(); }
  CHECK(a.state.st == b.state.st, "resume trajectory");
  CHECK(a.energy == b.energy, "resume energy");

  {
    std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
    f.seekg(static_cast<std::streamoff>(sizeof(CheckpointHeader) + 100));
    char c; f.get(c); c ^= 0x5a;
    f.seekp(static_cast<std::streamoff>(sizeof(CheckpointHeader) + 100));
    f.put(c); f.close();
  }
  CHECK(!load_checkpoint(path, b), "corrupt checkpoint rejected");
  std::filesystem::remove_all(dir);
}

static void test_histogram() {
  Sampler s = make_sampler();
  for (int i=0;i<100;i++) s.sweep();
  CHECK(s.samples_done == 50, "sample count");
  long long total = 0;
  for (auto v : s.hist) total += v;
  CHECK(total == 50, "histogram total");
  CHECK(s.energy == s.state.full_energy(), "sampler energy");
}

int main() {
  test_kernel();
  test_local_moves();
  test_checkpoint();
  test_histogram();
  if (failures == 0) std::printf("ALL TESTS PASS\n");
  else std::printf("%d FAILURES\n", failures);
  return failures == 0 ? 0 : 1;
}
