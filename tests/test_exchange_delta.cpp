#define LATTICE_GLASS_NO_MAIN
#include "../sim_omp.cpp"

#include <cassert>
#include <cmath>
#include <iostream>

static void fill_test_lattice(const int tid) {
  for (int site = 0; site < lat_size; ++site) {
    const uint8_t value = (site % 5 == 0) ? 1 : ((site % 7 == 0) ? 2 : 0);
    set_value_lattice(site, value, tid);
  }
}

static void assert_exchange_delta_matches_full_energy(const int site,
                                                      const int mv,
                                                      const int tid) {
  int affected[2 * (NUM_NN + 1)];
  const int affected_count =
      collect_exchange_affected_sites(site, mv, thread_nn[tid], affected);
  assert(affected_count <= 2 * (NUM_NN + 1));
  for (int i = 0; i < affected_count; ++i)
    for (int j = i + 1; j < affected_count; ++j)
      assert(affected[i] != affected[j]);

  const int full_before = energy(thread_nn[tid], tid);
  const float local_before =
      affected_energy_packed(affected, affected_count, thread_nn[tid], tid);

  exchange(site, mv, tid);

  const int full_after = energy(thread_nn[tid], tid);
  const float local_after =
      affected_energy_packed(affected, affected_count, thread_nn[tid], tid);

  assert(std::fabs((local_after - local_before) -
                   static_cast<float>(full_after - full_before)) < 1e-5f);

  exchange(site, mv, tid);
  assert(energy(thread_nn[tid], tid) == full_before);
}

int main() {
  generate_tables();
  constexpr int tid = 0;
  fill_test_lattice(tid);

  const int pairs[][2] = {
      {0, 1},
      {0, L},
      {0, L * L},
      {1, L * L + L + 1},
      {5, 42},
      {17, 113},
      {lat_size / 2, lat_size - 1},
  };

  for (const auto &pair : pairs)
    assert_exchange_delta_matches_full_energy(pair[0], pair[1], tid);

  std::cout << "exchange delta tests passed\n";
  return 0;
}
