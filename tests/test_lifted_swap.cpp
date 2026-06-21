#include "../fp_sampler.h"
#include "../lifted_swap.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace fp = lattice_glass::fp;
namespace axis_lift = lattice_glass::axis_lift;

static std::array<int, 3> counts(const std::vector<uint8_t> &lattice) {
  std::array<int, 3> c{};
  for (uint8_t v : lattice)
    c[v]++;
  return c;
}

static void test_conserves_composition() {
  const int L = 6;
  const int lat_size = L * L * L;
  const std::vector<int> nn = fp::cubic_neighbors(L);
  std::mt19937 gen(909);
  std::vector<uint8_t> lattice = fp::random_lattice(60, 90, lat_size, gen);
  const auto c0 = counts(lattice);

  for (int stride = 1; stride <= 3; ++stride) {
    const axis_lift::EventStats s = axis_lift::lifted_swap_chain(
        lattice, 2.0L, 200000, stride, 12, L, gen, nn.data());
    assert(counts(lattice) == c0);
    assert(s.accepted > 0);
  }
}

// Stationarity gate: composing the lifted nonlocal swap with the trusted
// non-local swap must not move the equilibrium energy distribution.
static void test_preserves_equilibrium() {
  const int L = 6;
  const int lat_size = L * L * L;
  const long double beta = 1.5L;
  const int stride = 2;
  const std::vector<int> nn = fp::cubic_neighbors(L);

  const int burn = 15000;
  const int measure = 90000;
  const long long events_per_sweep = lat_size;
  const int refresh = 2 * L;

  auto run = [&](bool with_lift, unsigned seed) {
    std::mt19937 gen(seed);
    std::vector<uint8_t> lattice = fp::random_lattice(60, 90, lat_size, gen);
    for (int s = 0; s < burn; ++s) {
      fp::nonlocal_swap_sweep(lattice, beta, 0.0L, nullptr, gen, nn.data());
      if (with_lift)
        axis_lift::lifted_swap_chain(lattice, beta, events_per_sweep, stride,
                                     refresh, L, gen, nn.data());
    }
    long double e_sum = 0.0L, e2_sum = 0.0L;
    for (int s = 0; s < measure; ++s) {
      fp::nonlocal_swap_sweep(lattice, beta, 0.0L, nullptr, gen, nn.data());
      if (with_lift)
        axis_lift::lifted_swap_chain(lattice, beta, events_per_sweep, stride,
                                     refresh, L, gen, nn.data());
      const long double e = fp::total_energy_int(lattice, nn.data());
      e_sum += e;
      e2_sum += e * e;
    }
    return std::array<long double, 2>{e_sum / measure, e2_sum / measure};
  };

  const auto swap_only = run(false, 303);
  const auto swap_lift = run(true, 404);

  const long double mean_rel =
      std::fabsl(swap_only[0] - swap_lift[0]) / swap_only[0];
  const long double mean2_rel =
      std::fabsl(swap_only[1] - swap_lift[1]) / swap_only[1];

  std::cout << "  swap_only <E>=" << static_cast<double>(swap_only[0])
            << " swap+lifted_swap <E>=" << static_cast<double>(swap_lift[0])
            << " rel=" << static_cast<double>(mean_rel) << "\n";

  assert(mean_rel < 0.02L);
  assert(mean2_rel < 0.04L);
}

int main() {
  test_conserves_composition();
  test_preserves_equilibrium();
  std::cout << "lifted swap tests passed\n";
  return 0;
}
