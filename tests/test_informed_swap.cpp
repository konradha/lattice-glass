#include "../fp_sampler.h"
#include "../informed_swap.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace fp = lattice_glass::fp;
namespace informed = lattice_glass::informed;
namespace cluster = lattice_glass::cluster;

static std::array<int, 3> counts(const std::vector<uint8_t> &lattice) {
  std::array<int, 3> c{};
  for (uint8_t v : lattice)
    c[v]++;
  return c;
}

// The occ/vac partition and the occupied-neighbour counts must stay exactly
// consistent with the lattice after any number of accepted moves.
static void assert_state_consistent(const std::vector<uint8_t> &lattice,
                                    const informed::SiteLists &lists,
                                    const std::vector<int> &m, const int *nn) {
  assert(lists.occ.size() + lists.vac.size() == lattice.size());
  for (int idx = 0; idx < static_cast<int>(lists.occ.size()); ++idx) {
    const int site = lists.occ[idx];
    assert(lattice[site] != cluster::kEmpty);
    assert(lists.slot[site] == idx);
  }
  for (int idx = 0; idx < static_cast<int>(lists.vac.size()); ++idx) {
    const int site = lists.vac[idx];
    assert(lattice[site] == cluster::kEmpty);
    assert(lists.slot[site] == idx);
  }
  const std::vector<int> fresh = informed::build_neighbor_counts(lattice, nn);
  assert(fresh == m);
}

// move_delta_energy (analytic, m-backed) must equal the region-recompute
// reference for every occupied->vacant relocation, across fresh and mid-chain
// configurations -- including i,j adjacency and shared-neighbour cases.
static void test_move_delta_matches_reference() {
  const int L = 6;
  const int lat_size = L * L * L;
  const std::vector<int> nn = fp::cubic_neighbors(L);
  std::mt19937 gen(4242);

  long long checks = 0;
  for (int trial = 0; trial < 400; ++trial) {
    std::vector<uint8_t> lattice = fp::random_lattice(60, 90, lat_size, gen);
    informed::SiteLists lists = informed::build_site_lists(lattice);
    std::vector<int> m = informed::build_neighbor_counts(lattice, nn.data());
    std::vector<double> wbuf(lists.vac.size(), 0.0);

    // Mix in a few accepted moves so configs are not all fresh-random.
    informed::blind_occupancy_sweep(lattice, 1.0L,
                                    static_cast<int>(lists.occ.size()), lists, m,
                                    gen, nn.data());

    std::uniform_int_distribution<int> occ_dist(
        0, static_cast<int>(lists.occ.size()) - 1);
    std::uniform_int_distribution<int> vac_dist(
        0, static_cast<int>(lists.vac.size()) - 1);
    for (int k = 0; k < 25; ++k) {
      const int i = lists.occ[occ_dist(gen)];
      const int j = lists.vac[vac_dist(gen)];
      const int analytic = informed::move_delta_energy(lattice, m, i, j, nn.data());
      const int reference = informed::swap_delta_energy(lattice, i, j, nn.data());
      assert(analytic == reference);

      // The informed sweep relies on dE(i->j) = remove(i) + insert(j in x').
      const uint8_t label = lattice[i];
      const int d_remove = informed::remove_particle_delta(lattice, m, i, nn.data());
      lattice[i] = cluster::kEmpty;
      const int *ni = nn.data() + lattice_glass::fp::kNumNeighbors * i;
      for (int kk = 0; kk < lattice_glass::fp::kNumNeighbors; ++kk)
        --m[ni[kk]];
      const int d_insert =
          informed::insert_particle_delta(lattice, m, j, label, nn.data());
      lattice[i] = label;
      for (int kk = 0; kk < lattice_glass::fp::kNumNeighbors; ++kk)
        ++m[ni[kk]];
      assert(d_remove + d_insert == analytic);
      ++checks;
    }
  }
  std::cout << "  move_delta matches reference on " << checks << " relocations\n";
}

static void test_conserves_composition() {
  const int L = 6;
  const int lat_size = L * L * L;
  const std::vector<int> nn = fp::cubic_neighbors(L);
  std::mt19937 gen(909);

  for (int informed_mode = 0; informed_mode < 2; ++informed_mode) {
    std::vector<uint8_t> lattice = fp::random_lattice(60, 90, lat_size, gen);
    const auto c0 = counts(lattice);
    informed::SiteLists lists = informed::build_site_lists(lattice);
    std::vector<int> m = informed::build_neighbor_counts(lattice, nn.data());
    std::vector<double> wbuf(lists.vac.size(), 0.0);
    const int np = static_cast<int>(lists.occ.size());

    long long accepted = 0;
    for (int round = 0; round < 40; ++round) {
      const informed::SweepStats s =
          informed_mode
              ? informed::informed_occupancy_sweep(lattice, 2.0L, np, lists, m,
                                                   wbuf, gen, nn.data())
              : informed::blind_occupancy_sweep(lattice, 2.0L, np, lists, m, gen,
                                                nn.data());
      accepted += s.accepted;
      assert(counts(lattice) == c0);
      assert_state_consistent(lattice, lists, m, nn.data());
    }
    assert(accepted > 0);
  }
}

struct ChainMoments {
  long double e_mean = 0.0L;
  long double e2_mean = 0.0L;
  long double bond_mean = 0.0L;
};

// Reference chain: the trusted all-pairs nonlocal swap (no site lists).
static ChainMoments run_reference(int L, long double beta, int burn, int measure,
                                  unsigned seed) {
  const int lat_size = L * L * L;
  const std::vector<int> nn = fp::cubic_neighbors(L);
  std::mt19937 gen(seed);
  std::vector<uint8_t> lattice = fp::random_lattice(60, 90, lat_size, gen);
  for (int s = 0; s < burn; ++s)
    fp::nonlocal_swap_sweep(lattice, beta, 0.0L, nullptr, gen, nn.data());

  ChainMoments m;
  for (int s = 0; s < measure; ++s) {
    fp::nonlocal_swap_sweep(lattice, beta, 0.0L, nullptr, gen, nn.data());
    const long double e = fp::total_energy_int(lattice, nn.data());
    m.e_mean += e;
    m.e2_mean += e * e;
    m.bond_mean += informed::occupied_bond_count(lattice, nn.data());
  }
  m.e_mean /= measure;
  m.e2_mean /= measure;
  m.bond_mean /= measure;
  return m;
}

// Occupancy-only chain under test: blind or informed destination proposal.
static ChainMoments run_occupancy_only(int L, long double beta, int burn,
                                       int measure, bool informed_mode,
                                       unsigned seed) {
  const int lat_size = L * L * L;
  const std::vector<int> nn = fp::cubic_neighbors(L);
  std::mt19937 gen(seed);
  std::vector<uint8_t> lattice = fp::random_lattice(60, 90, lat_size, gen);
  informed::SiteLists lists = informed::build_site_lists(lattice);
  std::vector<int> mcount = informed::build_neighbor_counts(lattice, nn.data());
  std::vector<double> wbuf(lists.vac.size(), 0.0);
  const int np = static_cast<int>(lists.occ.size());

  auto step = [&]() {
    if (informed_mode)
      informed::informed_occupancy_sweep(lattice, beta, np, lists, mcount, wbuf,
                                         gen, nn.data());
    else
      informed::blind_occupancy_sweep(lattice, beta, np, lists, mcount, gen,
                                      nn.data());
  };

  for (int s = 0; s < burn; ++s)
    step();

  ChainMoments m;
  for (int s = 0; s < measure; ++s) {
    step();
    const long double e = fp::total_energy_int(lattice, nn.data());
    m.e_mean += e;
    m.e2_mean += e * e;
    m.bond_mean += informed::occupied_bond_count(lattice, nn.data());
  }
  m.e_mean /= measure;
  m.e2_mean /= measure;
  m.bond_mean /= measure;
  return m;
}

// Stationarity gate: an occupancy-only chain (blind or informed) is ergodic for
// the fixed-composition Gibbs measure, so its equilibrium energy, energy^2, and
// occupied-bond moments must match the trusted nonlocal swap. The bond mean is
// a pure-occupancy observable, the sharpest check that the informed acceptance
// is exactly correct (a wrong Hastings ratio would bias the occupancy law).
static void test_occupancy_only_matches_reference() {
  const int L = 6;
  const long double beta = 1.5L;
  const int burn = 4000;
  const int measure = 15000;

  const ChainMoments ref = run_reference(L, beta, burn, measure, 1234);
  const ChainMoments blind =
      run_occupancy_only(L, beta, burn, measure, false, 5678);
  const ChainMoments inf =
      run_occupancy_only(L, beta, burn, measure, true, 9012);

  auto rel = [](long double a, long double b) {
    return std::fabsl(a - b) / std::fabsl(b);
  };

  std::cout << "  reference  <E>=" << static_cast<double>(ref.e_mean)
            << " <B>=" << static_cast<double>(ref.bond_mean) << "\n";
  std::cout << "  blind-occ  <E>=" << static_cast<double>(blind.e_mean)
            << " <B>=" << static_cast<double>(blind.bond_mean)
            << " relE=" << static_cast<double>(rel(blind.e_mean, ref.e_mean))
            << " relB=" << static_cast<double>(rel(blind.bond_mean, ref.bond_mean))
            << "\n";
  std::cout << "  informed   <E>=" << static_cast<double>(inf.e_mean)
            << " <B>=" << static_cast<double>(inf.bond_mean)
            << " relE=" << static_cast<double>(rel(inf.e_mean, ref.e_mean))
            << " relB=" << static_cast<double>(rel(inf.bond_mean, ref.bond_mean))
            << "\n";

  // Bond mean is the pure-occupancy DB check: tight. Energy carries extra MC
  // noise because the occupancy-only chain mixes species labels (by relocation)
  // slower than the all-pairs reference; allow a looser band there.
  assert(rel(blind.e_mean, ref.e_mean) < 0.03L);
  assert(rel(blind.e2_mean, ref.e2_mean) < 0.05L);
  assert(rel(blind.bond_mean, ref.bond_mean) < 0.012L);

  assert(rel(inf.e_mean, ref.e_mean) < 0.03L);
  assert(rel(inf.e2_mean, ref.e2_mean) < 0.05L);
  assert(rel(inf.bond_mean, ref.bond_mean) < 0.012L);
}

int main() {
  test_move_delta_matches_reference();
  test_conserves_composition();
  test_occupancy_only_matches_reference();
  std::cout << "informed swap tests passed\n";
  return 0;
}
