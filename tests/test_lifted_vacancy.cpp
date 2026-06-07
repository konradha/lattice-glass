#include "../fp_sampler.h"
#include "../lifted_vacancy.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace fp = lattice_glass::fp;
namespace lifted = lattice_glass::lifted;

static std::array<int, 3> counts(const std::vector<uint8_t> &lattice) {
  std::array<int, 3> c{};
  for (uint8_t v : lattice)
    c[v]++;
  return c;
}

static void test_lifted_conserves_composition() {
  const int L = 6;
  const int lat_size = L * L * L;
  const std::vector<int> nn = fp::cubic_neighbors(L);
  std::mt19937 gen(4242);
  std::vector<uint8_t> lattice = fp::random_lattice(60, 90, lat_size, gen);
  const auto c0 = counts(lattice);

  const lifted::EventStats s =
      lifted::lifted_vacancy_chain(lattice, 2.0L, 500000, 12, gen, nn.data());
  assert(counts(lattice) == c0);
  assert(s.accepted > 0); // the chain actually moves
}

// Stationarity gate: composing the lifted vacancy chain with the trusted
// non-local swap must not shift the equilibrium energy distribution. A broken
// global-balance construction would bias the first two energy moments.
static void test_lifted_preserves_equilibrium() {
  const int L = 6;
  const int lat_size = L * L * L;
  const long double beta = 1.5L;
  const std::vector<int> nn = fp::cubic_neighbors(L);

  const int burn = 15000;
  const int measure = 90000;
  const long long events_per_sweep = lat_size; // matched-ish elementary work
  const int refresh = 2 * L;

  auto run = [&](bool with_lifted, unsigned seed) {
    std::mt19937 gen(seed);
    std::vector<uint8_t> lattice = fp::random_lattice(60, 90, lat_size, gen);
    for (int s = 0; s < burn; ++s) {
      fp::nonlocal_swap_sweep(lattice, beta, 0.0L, nullptr, gen, nn.data());
      if (with_lifted)
        lifted::lifted_vacancy_chain(lattice, beta, events_per_sweep, refresh,
                                     gen, nn.data());
    }
    long double e_sum = 0.0L, e2_sum = 0.0L;
    for (int s = 0; s < measure; ++s) {
      fp::nonlocal_swap_sweep(lattice, beta, 0.0L, nullptr, gen, nn.data());
      if (with_lifted)
        lifted::lifted_vacancy_chain(lattice, beta, events_per_sweep, refresh,
                                     gen, nn.data());
      const long double e = fp::total_energy_int(lattice, nn.data());
      e_sum += e;
      e2_sum += e * e;
    }
    return std::array<long double, 2>{e_sum / measure, e2_sum / measure};
  };

  const auto swap_only = run(false, 101);
  const auto swap_lifted = run(true, 202);

  const long double mean_rel =
      std::fabsl(swap_only[0] - swap_lifted[0]) / swap_only[0];
  const long double mean2_rel =
      std::fabsl(swap_only[1] - swap_lifted[1]) / swap_only[1];

  std::cout << "  swap_only <E>=" << static_cast<double>(swap_only[0])
            << " swap+lifted <E>=" << static_cast<double>(swap_lifted[0])
            << " rel=" << static_cast<double>(mean_rel) << "\n";

  assert(mean_rel < 0.02L);
  assert(mean2_rel < 0.04L);
}

int main() {
  test_lifted_conserves_composition();
  test_lifted_preserves_equilibrium();
  std::cout << "lifted vacancy tests passed\n";
  return 0;
}
