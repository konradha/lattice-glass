/*
 * BUILD AND RUN
 *
 * make
 * OMP_NUM_THREADS=4 ./to_omp 5.0 0.58 0.29 output.npy
 *
 */

#include "maps_omp.h"
#include "npy.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <tuple>
#include <utility>

#if defined(__x86_64__) || defined(_M_X64)
#include <x86intrin.h>
#endif

int logand(uint8_t *x1, uint8_t *x2) {
  int nred = 0;
  int nblue = 0;
  for (int i = 0; i < L * L * L; ++i) {
    if (get_value(x1, i) == 1 && get_value(x2, i) == 1)
      nred++;
    if (get_value(x1, i) == 2 && get_value(x2, i) == 2)
      nblue++;
  }
  return nred + nblue;
}

static inline bool is_occupied(const uint8_t value) { return value != 0; }

static inline unsigned int rng_seed(const int tid) {
#if defined(__x86_64__) || defined(_M_X64)
  const uint64_t ticks = __rdtsc();
#else
  const auto now =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  const uint64_t ticks = static_cast<uint64_t>(now);
#endif
  const uint64_t mixed =
      ticks ^ (0x9E3779B97F4A7C15ull * static_cast<uint64_t>(tid + 1));
  return static_cast<unsigned int>(mixed ^ (mixed >> 32));
}

float local_energy_packed(const int site, const int *nearest_neighbors,
                          const int tid) {
  const int ty = get_value_lattice(site, tid);
  if (ty == 0)
    return 0.;
  const float connection = ty == 1 ? 3. : 5.;
  const int *nne = &(nearest_neighbors[NUM_NN * site]);
  const int e = is_occupied(get_value_lattice(nne[0], tid)) +
                is_occupied(get_value_lattice(nne[1], tid)) +
                is_occupied(get_value_lattice(nne[2], tid)) +
                is_occupied(get_value_lattice(nne[3], tid)) +
                is_occupied(get_value_lattice(nne[4], tid)) +
                is_occupied(get_value_lattice(nne[5], tid));
  const float current = static_cast<float>(e) - connection;
  return current * current;
}

int energy(const int *nearest_neighbors, const int &tid) {
  int e = 0;
  for (int i = 0; i < L * L * L; ++i)
    e += local_energy_packed(i, nearest_neighbors, tid);
  return e;
}

void geom_series(float *betas, float low, float high, int n) {
  float r = pow(high / low, 1.0 / (n - 1));
  betas[0] = low;
  for (int i = 1; i < n; i++) {
    betas[i] = betas[i - 1] * r;
  }
  assert((betas[n - 1] - high) < 1.e-5);
}

static inline std::tuple<int, int, int> revert(int s) {
  const auto k = s % L;
  const auto j = ((s - k) / L) % L;
  const auto i = (s - k - j * L) / (L * L);
  return {i, j, k};
}

void build_lattice(const int num_red, const int num_blue,
                   std::mt19937 &generator,
                   std::uniform_int_distribution<> &indices, const int tid) {
  for (int i = 0; i < L * L * L; ++i)
    set_value_lattice(i, 0, tid);
  int curr_red, curr_blue;
  curr_red = curr_blue = 0;
  while (curr_red < num_red) {
    int site = indices(generator);
    while (get_value_lattice(site, tid) != 0)
      site = indices(generator);
    set_value_lattice(site, 1, tid);
    curr_red++;
  }

  while (curr_blue < num_blue) {
    int site = indices(generator);
    while (get_value_lattice(site, tid) != 0)
      site = indices(generator);
    set_value_lattice(site, 2, tid);
    curr_blue++;
  }

  int nred, nblue;
  nred = nblue = 0;
  for (int i = 0; i < L * L * L; ++i) {
    if (get_value_lattice(i, tid) == 1)
      nred++;
    else if (get_value_lattice(i, tid) == 2)
      nblue++;
  }
  assert(nred == num_red);
  assert(nblue == num_blue);
}

void build_lattice_diag(const int num_red, const int num_blue,
                        std::mt19937 &generator,
                        std::uniform_int_distribution<> &indices,
                        const int tid) {

  for (int i = 0; i < L * L * L; ++i)
    set_value_lattice(i, 0, tid);
  int curr_red, curr_blue;
  curr_red = curr_blue = 0;
  while (curr_red < num_red) {
    const int site = indices(generator);
    const auto [i, j, k] = revert(site);

    if (((i + j + k) & 1) == 0 &&
        static_cast<int>(get_value_lattice(site, tid)) == 0) {
      set_value_lattice(site, 1, tid);
      curr_red++;
      continue;
    }
  }
  while (curr_blue < num_blue) {
    const int site = indices(generator);
    const auto [i, j, k] = revert(site);
    if (((i + j + k) & 1) == 1 &&
        static_cast<int>(get_value_lattice(site, tid)) == 0) {
      set_value_lattice(site, 2, tid);
      curr_blue++;
    }
  }
  int nred, nblue;
  nred = nblue = 0;
  for (int i = 0; i < L * L * L; ++i) {
    if (get_value_lattice(i, tid) == 1)
      nred++;
    else if (get_value_lattice(i, tid) == 2)
      nblue++;
  }
  assert(nred == num_red);
  assert(nblue == num_blue);
}

void exchange(const int &site, const int &to, const int &tid) {
  const auto tmp = get_value_lattice(site, tid);
  set_value_lattice(site, get_value_lattice(to, tid), tid);
  set_value_lattice(to, tmp, tid);
}


static inline void append_unique_site(const int site, int *affected,
                                      int &count) {
  for (int i = 0; i < count; ++i)
    if (affected[i] == site)
      return;
  affected[count++] = site;
}

static inline int collect_exchange_affected_sites(
    const int site, const int mv, const int *nearest_neighbors, int *affected) {
  int count = 0;
  append_unique_site(site, affected, count);
  append_unique_site(mv, affected, count);

  const int *site_nn = &(nearest_neighbors[NUM_NN * site]);
  const int *mv_nn = &(nearest_neighbors[NUM_NN * mv]);
  for (int i = 0; i < NUM_NN; ++i) {
    append_unique_site(site_nn[i], affected, count);
    append_unique_site(mv_nn[i], affected, count);
  }
  return count;
}

static inline float affected_energy_packed(const int *affected,
                                           const int count,
                                           const int *nearest_neighbors,
                                           const int tid) {
  float energy = 0.;
  for (int i = 0; i < count; ++i)
    energy += local_energy_packed(affected[i], nearest_neighbors, tid);
  return energy;
}

static inline bool metropolis_exchange(
    const int site, const int mv, const float beta, std::mt19937 &generator,
    std::uniform_real_distribution<> &uni, const int *nearest_neighbors,
    const int tid) {
  if (get_value_lattice(site, tid) == get_value_lattice(mv, tid))
    return false;

  int affected[2 * (NUM_NN + 1)];
  const int affected_count =
      collect_exchange_affected_sites(site, mv, nearest_neighbors, affected);
  const float E1 =
      affected_energy_packed(affected, affected_count, nearest_neighbors, tid);
  exchange(site, mv, tid);
  const float E2 =
      affected_energy_packed(affected, affected_count, nearest_neighbors, tid);
  const float dE = E2 - E1;
  if (dE <= 0. ||
      uni(generator) < std::exp(-static_cast<double>(beta) *
                                static_cast<double>(dE)))
    return true;

  exchange(site, mv, tid);
  return false;
}

void nonlocal_sweep(const int &num_trials, const float &beta,
                    std::mt19937 &generator,
                    std::uniform_int_distribution<> &indices,
                    std::uniform_real_distribution<> &uni,
                    const int *nearest_neighbors, const int &tid) {
  for (int i = 0; i < num_trials; ++i) {
    const int site = indices(generator);
    const int mv = indices(generator);
    metropolis_exchange(site, mv, beta, generator, uni, nearest_neighbors, tid);
  }
}

void concentrated_sweep(const float th, const float &beta,
                        std::mt19937 &generator,
                        std::uniform_int_distribution<> &indices,
                        std::uniform_real_distribution<> &uni,
                        const int *nearest_neighbors, const int &tid) {

  // need to reach certain threshold of low energy region -- could be
  // parametrized, too
  if (energy(nearest_neighbors, tid) > (int)(.3333 * L * L * L)) {
    nonlocal_sweep(L * L * L, beta, generator, indices, uni, nearest_neighbors,
                   tid);
#ifdef DEBUG
#pragma omp critical
    std::cout << "lattice " << tid << " too hot, launching nonlocal sweep\n";
#endif
    return;
  }
  constexpr int max_idx = L * L * L / 3;
  int hottest_idx[max_idx];
  for (int i = 0; i < max_idx; ++i)
    hottest_idx[i] = -1;

  int current_idx = 0;
  for (int site = 0; site < L * L * L; ++site) {
    if (local_energy_packed(site, nearest_neighbors, tid) > th)
      hottest_idx[current_idx++] = site;
    if (current_idx > max_idx - 1)
      break;
  }

  const int hot_count = current_idx;
  if (hot_count == 0)
    return;

  constexpr int num_trials = (int)(.4 * L * L * L);

  if (indices(generator) % 2 == 0) {
    for (int i = 0; i < num_trials; ++i) {
      const int site = hottest_idx[indices(generator) % hot_count];
      const int mv = hottest_idx[indices(generator) % hot_count];
      metropolis_exchange(site, mv, beta, generator, uni, nearest_neighbors,
                          tid);
    }
  } else {
    for (int i = 0; i < num_trials; ++i) {
      const int site = hottest_idx[indices(generator) % hot_count];
      const int nb = indices(generator) % NUM_NN;
      const int mv = nearest_neighbors[NUM_NN * site + nb];
      metropolis_exchange(site, mv, beta, generator, uni, nearest_neighbors,
                          tid);
    }
  }
}

void nonlocal_sweep_partitioned(const float &beta, std::mt19937 &generator,
                                std::uniform_int_distribution<> &indices,
                                std::uniform_real_distribution<> &uni,
                                const int *nearest_neighbors, const int &tid) {
  constexpr int max_partitions = 16;
  const int num_partitions = 1 << (1 + (indices(generator) % 4));
  const int partition_size = lat_size / num_partitions;

  std::array<std::pair<int, float>, max_partitions> partition_energies{};
  for (int i = 0; i < num_partitions; ++i) {
    const int partition_start = i * partition_size;
    const int partition_end = partition_start + partition_size;
    float e = 0.;
    for (int idx = partition_start; idx < partition_end; ++idx)
      e += local_energy_packed(idx, nearest_neighbors, tid);
    partition_energies[i] = {partition_start, e};
  }

  std::sort(partition_energies.begin(),
            partition_energies.begin() + num_partitions,
            [](const std::pair<int, float> x,
               const std::pair<int, float> y) { return x.second > y.second; });

  const int coldest_region_start = partition_energies[num_partitions - 1].first;
  const int warmest_region_start = partition_energies[0].first;

  for (int i = 0; i < partition_size; ++i) {
    const int site =
        warmest_region_start + (indices(generator) % partition_size);
    const int mv =
        coldest_region_start + (indices(generator) % partition_size);
    metropolis_exchange(site, mv, beta, generator, uni, nearest_neighbors, tid);
  }
}

void local_sweep(const float &beta, std::mt19937 &generator,
                 std::uniform_int_distribution<> &indices,
                 std::uniform_real_distribution<> &uni,
                 const int *nearest_neighbors, const int &tid) {

  for (int i = 0; i < L * L * L; ++i) {
    const int site = indices(generator);
    const int nb = indices(generator) % NUM_NN;
    const int mv = nearest_neighbors[NUM_NN * site + nb];
    metropolis_exchange(site, mv, beta, generator, uni, nearest_neighbors, tid);
  }
}

#ifndef LATTICE_GLASS_NO_MAIN
int main(int argc, char **argv) {
  if (argc != 5) {
    std::cout << "run as: ./bin beta rho rho1 outfile-path\n";
    return 1;
  }
  const auto arg1 = argv[1];
  const auto arg2 = argv[2];
  const auto arg3 = argv[3];
  const auto arg4 = argv[4];
  const auto beta = atof(arg1);
  const auto fname = std::string(arg4);

#pragma omp parallel
  assert(omp_get_num_threads() == NUM_THREADS);

  const float rho = atof(arg2);
  
  const float rho1 = atof(arg3);
  
  const int N = (int)(lat_size * rho);
  const int N1 = (int)(rho1 * lat_size);
  const int N2 = N - N1;
  const auto max_idx = L * L * L - 1;
  generate_tables();

  std::array<double, NUM_THREADS> times = {0.0};

  for (int t = 0; t < NUM_THREADS; ++t)
    for (int i = 0; i < L * L * L; ++i)
      set_value_lattice(i, 0, t);

#pragma omp parallel
  {
    const auto tid = omp_get_thread_num();
    auto uni = std::uniform_real_distribution<>(0., 1.);
    auto indices = std::uniform_int_distribution<>(0, max_idx);
    auto generator = std::mt19937();
    generator.seed(rng_seed(tid));

    // TODO: analysis
    // build_lattice_diag(N1, N2, generator, indices, tid);
    build_lattice(N1, N2, generator, indices, tid);
  }
#pragma omp flush

  constexpr size_t max_collect =
      100; // to have sufficient distance when copying in parallel
  uint8_t *config_collection = (uint8_t *)malloc(sizeof(uint8_t) * max_collect *
                                                 NUM_THREADS * packed_size);

  for (int t = 0; t < NUM_THREADS; ++t) {
    const auto offset = (t * max_collect) * packed_size;
    uint8_t *copy_spot = config_collection + offset;
    std::copy(thread_lattice[t], thread_lattice[t] + packed_size, copy_spot);
  }

  int cpycounter = 1;

  // heuristic:
  // 0. set up geometric series for simulated annealing
  // 1. partitioned "diffusion" sweeps -- build rotations: only two axes, ie. up
  // and right
  // 2. check if e_avg <= \kappa \rho L³
  // 3. if reached: do concentrated sweeps with find max \kappa \rho L³ sites
  // with local_e > 1
  // 4. continue for k steps
  // 5. then find particles with local_e > 0
  // 6. stop after MAX_ITER
#pragma omp parallel
  {
    const auto tid = omp_get_thread_num();
    auto uni = std::uniform_real_distribution<>(0., 1.);
    auto indices = std::uniform_int_distribution<>(0, max_idx);
    auto generator = std::mt19937();
    generator.seed(rng_seed(tid));


    constexpr int TESTP = 10;
    const int numsweeps_power = (beta <= 6.) ? (2 * ceil(beta) + 13) : 23;
    const double delta = .05;
    const double low_beta = 1.1 - delta;
    // cooling rate
    const float cliff = .1 * beta;
    const double r =
        pow((double)(beta - cliff) / low_beta,
            low_beta / (double)((1 << (numsweeps_power - 1))));


#pragma omp master
    {
      std::cout << "running " << (1 << (numsweeps_power))
                << " diffusion cooling sweeps for L=" << L << "\n";
      std::cout << "and " << (1 << (2 * TESTP + 1))
                << " nonlocal sweeps\n";
    }

    int my_cpycounter;
#pragma omp critical
    my_cpycounter = cpycounter;
#pragma omp barrier

    uint8_t *my_lattice = thread_lattice[tid];
    int *my_nn = thread_nn[tid];
    float curr_beta = static_cast<float>(low_beta);

    for (int d = 1; d < numsweeps_power; ++d) {
      auto t = -omp_get_wtime();
      for (int i = 1 << (d - 1); i < 1 << d; ++i) {
        curr_beta *= static_cast<float>(r);
        // nonlocal_sweep(L * L * L, beta, generator, indices, uni, my_nn, tid);
        nonlocal_sweep_partitioned(curr_beta, generator, indices, uni, my_nn,
                                   tid);
      }

#pragma omp master
      {
        std::cout << std::fixed << std::setprecision(4) << curr_beta << ": ";
        for (int t = 0; t < NUM_THREADS; ++t)
          std::cout << energy(my_nn, t) << " ";
        std::cout << "\n";
      }

      t += omp_get_wtime();
      const auto offset = (tid * max_collect + my_cpycounter++) * packed_size;
      uint8_t *copy_spot = config_collection + offset;
      std::copy(my_lattice, my_lattice + packed_size, copy_spot);

#pragma omp master
      cpycounter++;
#pragma omp critical
      times[tid] += t;
    }
#pragma omp master
    std::cout << "\nstarting nonlocal sweeps\n\n";

#pragma omp barrier

    for (int d = 1; d < 2 * TESTP + 1; ++d) {
      auto t = -omp_get_wtime();
      for (int i = 1 << (d - 1); i < 1 << d; ++i) {
        nonlocal_sweep(L * L * L, beta, generator, indices, uni, my_nn, tid);
      }

#pragma omp master
      {
        std::cout << (1 << (d - 1)) << ": ";
        for (int t = 0; t < NUM_THREADS; ++t)
          std::cout << energy(my_nn, t) << " ";
        std::cout << "\n";
      }

      t += omp_get_wtime();
      const auto offset = (tid * max_collect + my_cpycounter++) * packed_size;
      uint8_t *copy_spot = config_collection + offset;
      std::copy(my_lattice, my_lattice + packed_size, copy_spot);

#pragma omp master
      cpycounter++;
#pragma omp critical
      times[tid] += t;
    }
#pragma omp barrier
  }
#pragma omp flush

  for (size_t t = 0; t < NUM_THREADS; ++t) {
    for (size_t d = 0; d < cpycounter; ++d) {
      const auto offset = (t * max_collect + d) * packed_size;
      uint8_t *current_config = config_collection + offset;
      int nred, nblue;
      nred = nblue = 0;
      for (int i = 0; i < L * L * L; ++i) {
        if (static_cast<short>(get_value(current_config, i)) == 1)
          nred++;
        if (static_cast<short>(get_value(current_config, i)) == 2)
          nblue++;
      }
      if (N1 != nred)
        std::cout << "it's red on " << t << " epoch=" << d << "\n";
      if (N2 != nblue)
        std::cout << "it's blue on " << t << " epoch=" << d << "\n";
      assert(N1 == nred);
      assert(N2 == nblue);
    }
  }

  short *serialized_configs =
      (short *)malloc(sizeof(short) * NUM_THREADS * cpycounter * L * L * L);
  for (size_t t = 0; t < NUM_THREADS; ++t) {
    for (size_t d = 0; d < cpycounter; ++d) {
      const auto offset = (t * max_collect + d) * packed_size;
      uint8_t *current_config = config_collection + offset;
      for (size_t s = 0; s < L * L * L; ++s)
        serialized_configs[(t * cpycounter * L * L * L) + (d * L * L * L) + s] =
            static_cast<short>(get_value(current_config, s));
    }
  }

  // could save some mem using uint8_t -> np.int8
  npy::npy_data_ptr<short> d;
  d.data_ptr = reinterpret_cast<const short *>(serialized_configs);

  d.shape = {NUM_THREADS, static_cast<unsigned long>(cpycounter), L, L, L};

  d.fortran_order = false;

#ifdef DEBUG
  std::cout << "SAVING " << sizeof(short) * NUM_THREADS * cpycounter * L * L * L
            << " bytes\n";
#endif

  npy::write_npy(fname, d);

#ifdef DEBUG
  std::cout << "RUNTIME\n";
  for (size_t t = 0; t < NUM_THREADS; ++t)
    std::cout << times[t] << " ";
  std::cout << "\n";
#endif

  free(serialized_configs);
  free(config_collection);
  return 0;
}
#endif
