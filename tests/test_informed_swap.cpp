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

// MTM-only chain under test (capped informed with k candidates).
static ChainMoments run_mtm_only(int L, long double beta, int burn, int measure,
                                 int k, unsigned seed) {
  const int lat_size = L * L * L;
  const std::vector<int> nn = fp::cubic_neighbors(L);
  std::mt19937 gen(seed);
  std::vector<uint8_t> lattice = fp::random_lattice(60, 90, lat_size, gen);
  informed::SiteLists lists = informed::build_site_lists(lattice);
  std::vector<int> mcount = informed::build_neighbor_counts(lattice, nn.data());
  std::vector<double> cw(k, 0.0);
  std::vector<int> cv(k, 0);
  const int np = static_cast<int>(lists.occ.size());
  auto step = [&]() {
    informed::informed_mtm_swap_sweep(lattice, beta, np, k, lists, mcount, cw, cv,
                                      gen, nn.data());
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

// Validates the MTM factorization at the exact integer-dE level: the forward
// decomposition dE(i->v) = remove(i) + insert(v in x'), and the reverse-weight
// identity dE_y(j->v) = -insert(j in x') + insert(v in x') that lets the reverse
// normalizer be formed without re-deriving energies in y.
static void test_mtm_reverse_identity() {
  const int L = 6;
  const int lat_size = L * L * L;
  const std::vector<int> nn = fp::cubic_neighbors(L);
  std::mt19937 gen(7777);
  long long checks = 0;
  for (int trial = 0; trial < 300; ++trial) {
    std::vector<uint8_t> lat = fp::random_lattice(60, 90, lat_size, gen);
    std::vector<int> m = informed::build_neighbor_counts(lat, nn.data());
    informed::SiteLists lists = informed::build_site_lists(lat);
    std::uniform_int_distribution<int> od(0, static_cast<int>(lists.occ.size()) - 1);
    std::uniform_int_distribution<int> vd(0, static_cast<int>(lists.vac.size()) - 1);
    const int i = lists.occ[od(gen)];
    const int j = lists.vac[vd(gen)];
    const uint8_t label = lat[i];

    const int d_ij = informed::move_delta_energy(lat, m, i, j, nn.data());
    const int d_rem = informed::remove_particle_delta(lat, m, i, nn.data());

    std::vector<uint8_t> xp = lat; // x' = x with i removed
    std::vector<int> mp = m;
    xp[i] = cluster::kEmpty;
    const int *ni = nn.data() + lattice_glass::fp::kNumNeighbors * i;
    for (int t = 0; t < lattice_glass::fp::kNumNeighbors; ++t)
      --mp[ni[t]];
    const int ins_j = informed::insert_particle_delta(xp, mp, j, label, nn.data());
    assert(d_rem + ins_j == d_ij); // forward decomposition

    std::vector<uint8_t> y = xp; // y = x' + particle at j
    std::vector<int> my = mp;
    y[j] = label;
    const int *nj = nn.data() + lattice_glass::fp::kNumNeighbors * j;
    for (int t = 0; t < lattice_glass::fp::kNumNeighbors; ++t)
      ++my[nj[t]];

    int vtests[2] = {i, lists.vac[vd(gen)]};
    for (int q = 0; q < 2; ++q) {
      int v = vtests[q];
      if (v == j)
        v = i; // v must be a vacancy of y
      const int direct = informed::move_delta_energy(y, my, j, v, nn.data());
      const int ins_v = informed::insert_particle_delta(xp, mp, v, label, nn.data());
      assert(direct == -ins_j + ins_v); // reverse-weight identity
      ++checks;
    }
  }
  std::cout << "  mtm reverse identity holds on " << checks << " pairs\n";
}

static void test_mtm_conserves() {
  const int L = 6;
  const int lat_size = L * L * L;
  const std::vector<int> nn = fp::cubic_neighbors(L);
  std::mt19937 gen(2024);
  std::vector<uint8_t> lattice = fp::random_lattice(60, 90, lat_size, gen);
  const auto c0 = counts(lattice);
  informed::SiteLists lists = informed::build_site_lists(lattice);
  std::vector<int> m = informed::build_neighbor_counts(lattice, nn.data());
  std::vector<double> cw(8, 0.0);
  std::vector<int> cv(8, 0);
  const int np = static_cast<int>(lists.occ.size());
  long long accepted = 0;
  for (int round = 0; round < 60; ++round) {
    const informed::SweepStats s = informed::informed_mtm_swap_sweep(
        lattice, 2.0L, np, 8, lists, m, cw, cv, gen, nn.data());
    accepted += s.accepted;
    assert(counts(lattice) == c0);
    assert_state_consistent(lattice, lists, m, nn.data());
  }
  assert(accepted > 0);
}

// Stationarity gate for the capped MTM kernel: its equilibrium must match the
// trusted nonlocal swap (sharp on the pure-occupancy bond mean).
static void test_mtm_matches_reference() {
  const int L = 6;
  const long double beta = 1.5L;
  const int burn = 4000;
  const int measure = 15000;
  const int k = 8;
  const ChainMoments ref = run_reference(L, beta, burn, measure, 1234);
  const ChainMoments mtm = run_mtm_only(L, beta, burn, measure, k, 31337);
  auto rel = [](long double a, long double b) {
    return std::fabsl(a - b) / std::fabsl(b);
  };
  std::cout << "  mtm(k=" << k << ")  <E>=" << static_cast<double>(mtm.e_mean)
            << " <B>=" << static_cast<double>(mtm.bond_mean)
            << " relE=" << static_cast<double>(rel(mtm.e_mean, ref.e_mean))
            << " relB=" << static_cast<double>(rel(mtm.bond_mean, ref.bond_mean))
            << "\n";
  assert(rel(mtm.e_mean, ref.e_mean) < 0.03L);
  assert(rel(mtm.e2_mean, ref.e2_mean) < 0.05L);
  assert(rel(mtm.bond_mean, ref.bond_mean) < 0.012L);
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

// fast_swap_sweep is the trusted all-pairs swap with O(1)/O(6) energy; its
// equilibrium must match nonlocal_swap_sweep tightly (identical move set), and
// it must keep the occupied-neighbour-count array exactly consistent.
static void test_fast_swap_matches_reference() {
  const int L = 6;
  const long double beta = 1.5L;
  const int burn = 5000, measure = 40000, off = 160;
  const std::vector<int> nn = fp::cubic_neighbors(L);
  std::vector<double> etab(off + 1);
  for (int d = 0; d <= off; ++d)
    etab[d] = std::exp(-static_cast<double>(beta) * d);
  std::mt19937 gen(24680);
  std::vector<uint8_t> lat = fp::random_lattice(60, 90, L * L * L, gen);
  std::vector<int> m = informed::build_neighbor_counts(lat, nn.data());
  const auto c0 = counts(lat);
  for (int s = 0; s < burn; ++s)
    informed::fast_swap_sweep(lat, m, beta, L * L * L, etab.data(), off, gen, nn.data());
  ChainMoments fs;
  for (int s = 0; s < measure; ++s) {
    informed::fast_swap_sweep(lat, m, beta, L * L * L, etab.data(), off, gen, nn.data());
    const long double e = fp::total_energy_int(lat, nn.data());
    fs.e_mean += e;
    fs.e2_mean += e * e;
    fs.bond_mean += informed::occupied_bond_count(lat, nn.data());
  }
  fs.e_mean /= measure;
  fs.e2_mean /= measure;
  fs.bond_mean /= measure;
  assert(counts(lat) == c0);
  assert(informed::build_neighbor_counts(lat, nn.data()) == m); // m stayed exact
  const ChainMoments ref = run_reference(L, beta, burn, measure, 1234);
  auto rel = [](long double a, long double b) {
    return std::fabsl(a - b) / std::fabsl(b);
  };
  std::cout << "  fast_swap  <E>=" << static_cast<double>(fs.e_mean)
            << " <B>=" << static_cast<double>(fs.bond_mean)
            << " relE=" << static_cast<double>(rel(fs.e_mean, ref.e_mean))
            << " relB=" << static_cast<double>(rel(fs.bond_mean, ref.bond_mean)) << "\n";
  // Energy carries species-mixing MC noise at L=6/15k samples even between two
  // correct samplers; the deterministic species-delta check below is the exact
  // energy gate, the bond mean (~5e-5) is the exact occupancy gate.
  assert(rel(fs.e_mean, ref.e_mean) < 0.03L);
  assert(rel(fs.e2_mean, ref.e2_mean) < 0.04L);
  assert(rel(fs.bond_mean, ref.bond_mean) < 0.012L);
}

// Exact check of the O(1) species-swap energy used by fast_swap_sweep against
// the region-recompute reference, over many random type1<->type2 swaps.
static void test_species_delta_matches_reference() {
  const int L = 6;
  const std::vector<int> nn = fp::cubic_neighbors(L);
  std::mt19937 gen(13579);
  long long checks = 0;
  for (int trial = 0; trial < 300; ++trial) {
    std::vector<uint8_t> lat = fp::random_lattice(60, 90, L * L * L, gen);
    std::vector<int> m = informed::build_neighbor_counts(lat, nn.data());
    std::vector<int> t1, t2;
    for (int s = 0; s < static_cast<int>(lat.size()); ++s) {
      if (lat[s] == cluster::kType1) t1.push_back(s);
      else if (lat[s] == cluster::kType2) t2.push_back(s);
    }
    std::uniform_int_distribution<int> d1(0, static_cast<int>(t1.size()) - 1);
    std::uniform_int_distribution<int> d2(0, static_cast<int>(t2.size()) - 1);
    for (int k = 0; k < 20; ++k) {
      const int x = t1[d1(gen)], y = t2[d2(gen)];
      const int lxp = informed::preferred_coordination(lat[x]);
      const int lyp = informed::preferred_coordination(lat[y]);
      const int dx = m[x], dy = m[y];
      const int dfast = (dx - lyp) * (dx - lyp) - (dx - lxp) * (dx - lxp) +
                        (dy - lxp) * (dy - lxp) - (dy - lyp) * (dy - lyp);
      const int dref = informed::swap_delta_energy(lat, x, y, nn.data());
      assert(dfast == dref);
      ++checks;
    }
  }
  std::cout << "  species delta matches reference on " << checks << " swaps\n";
}
int main() {
  test_move_delta_matches_reference();
  test_mtm_reverse_identity();
  test_conserves_composition();
  test_mtm_conserves();
  test_occupancy_only_matches_reference();
  test_mtm_matches_reference();
  test_species_delta_matches_reference();
  test_fast_swap_matches_reference();
  std::cout << "informed swap tests passed\n";
  return 0;
}
