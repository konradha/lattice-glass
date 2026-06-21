#include "../collective_moves.h"
#include "../fp_sampler.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace fp = lattice_glass::fp;
namespace collective = lattice_glass::collective;

static std::array<int, 3> counts(const std::vector<uint8_t> &lattice) {
  std::array<int, 3> c{};
  for (uint8_t v : lattice)
    c[v]++;
  return c;
}

// The cooperative vacancy chain must conserve each species' particle count.
static void test_chain_conserves_composition() {
  const int L = 6;
  const int lat_size = L * L * L;
  const std::vector<int> nn = fp::cubic_neighbors(L);
  std::mt19937 gen(2024);
  std::vector<uint8_t> lattice = fp::random_lattice(60, 90, lat_size, gen);
  const auto c0 = counts(lattice);

  int accepted = 0;
  for (int i = 0; i < 200000; ++i) {
    const collective::ChainResult r =
        collective::vacancy_chain_move(lattice, 2.0L, 5, gen, nn.data());
    accepted += r.accepted;
    assert(counts(lattice) == c0);
  }
  assert(accepted > 0); // the move is not trivially inert
}

// Stationarity gate: adding the chain to the trusted non-local swap must not
// change the equilibrium energy distribution. If the chain violated detailed
// balance, the first two energy moments would shift.
static void test_chain_preserves_equilibrium() {
  const int L = 6;
  const int lat_size = L * L * L;
  const long double beta = 1.5L;
  const std::vector<int> nn = fp::cubic_neighbors(L);

  const int burn = 15000;
  const int measure = 90000;

  auto run = [&](bool with_chain, unsigned seed) {
    std::mt19937 gen(seed);
    std::vector<uint8_t> lattice = fp::random_lattice(60, 90, lat_size, gen);
    for (int s = 0; s < burn; ++s) {
      fp::nonlocal_swap_sweep(lattice, beta, 0.0L, nullptr, gen, nn.data());
      if (with_chain)
        collective::vacancy_chain_sweep(lattice, beta, 4, 30, gen, nn.data());
    }
    long double e_sum = 0.0L, e2_sum = 0.0L;
    for (int s = 0; s < measure; ++s) {
      fp::nonlocal_swap_sweep(lattice, beta, 0.0L, nullptr, gen, nn.data());
      if (with_chain)
        collective::vacancy_chain_sweep(lattice, beta, 4, 30, gen, nn.data());
      const long double e = fp::total_energy_int(lattice, nn.data());
      e_sum += e;
      e2_sum += e * e;
    }
    const long double mean = e_sum / measure;
    const long double mean2 = e2_sum / measure;
    return std::array<long double, 2>{mean, mean2};
  };

  const auto swap_only = run(false, 11);
  const auto swap_chain = run(true, 22);

  const long double mean_rel =
      std::fabsl(swap_only[0] - swap_chain[0]) / swap_only[0];
  const long double mean2_rel =
      std::fabsl(swap_only[1] - swap_chain[1]) / swap_only[1];

  std::cout << "  swap_only <E>=" << static_cast<double>(swap_only[0])
            << " swap+chain <E>=" << static_cast<double>(swap_chain[0])
            << " rel=" << static_cast<double>(mean_rel) << "\n";

  assert(mean_rel < 0.02L);
  assert(mean2_rel < 0.04L);
}

int main() {
  test_chain_conserves_composition();
  test_chain_preserves_equilibrium();
  std::cout << "collective move tests passed\n";
  return 0;
}
