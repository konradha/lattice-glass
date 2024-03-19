/*
 * COMPILE COMMAND
 *
 * clang++-17 -fno-strict-aliasing -ftree-vectorize -pedantic -ffast-math
 * -march=native -O3 -Wall -Wunknown-pragmas -fopenmp  -lm -lstdc++ -std=c++17
 * sim_omp.cpp -o to_omp && GOMP_CPU_AFFINITY="0,2,8,10" ./to_omp 7.
 *
 */

#include "maps_omp.h"
#include "npy.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <tuple>

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

#define ONETWO(x) ((x & 0x1) || (x & 0x2))
const float local_energy_packed(const int &site, const int *nearest_neighbors,
                                const int &tid) {
  const int ty = get_value_lattice(site, tid);
  if (ty == 0)
    return 0.;
  const float connection = ty == 1 ? 3. : 5.;
  const int *nne = &(nearest_neighbors[NUM_NN * site]);
  const int e = ONETWO(get_value_lattice(nne[0], tid)) +
                ONETWO(get_value_lattice(nne[1], tid)) +
                ONETWO(get_value_lattice(nne[2], tid)) +
                ONETWO(get_value_lattice(nne[3], tid)) +
                ONETWO(get_value_lattice(nne[4], tid)) +
                ONETWO(get_value_lattice(nne[5], tid));
  float current = static_cast<float>(e) - connection;
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

void build_lattice(const int &num_particles, const int &num_red,
                   const int &num_blue, std::mt19937 &generator,
                   std::uniform_int_distribution<> &indices, const int &tid) {
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

void build_lattice_diag(const int &num_particles, const int &num_red,
                        const int &num_blue, std::mt19937 &generator,
                        std::uniform_int_distribution<> &indices,
                        const int &tid) {

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

const float nn_energy_packed(const int &site, const int *nearest_neighbors,
                             const int &tid) {
  float res = local_energy_packed(site, nearest_neighbors, tid);
  const int *nne = &(nearest_neighbors[NUM_NN * site]);
  for (int i = 0; i < NUM_NN; ++i)
    res += local_energy_packed(nne[i], nearest_neighbors, tid);
  return res;
}

void nonlocal_sweep(const int &num_trials, const float &beta,
                    std::mt19937 &generator,
                    std::uniform_int_distribution<> &indices,
                    std::uniform_real_distribution<> &uni,
                    const int *nearest_neighbors, const int &tid) {
  for (int i = 0; i < num_trials; ++i) {
    const int site = indices(generator);
    const auto mv = indices(generator);
    if (get_value_lattice(site, tid) == get_value_lattice(mv, tid))
      continue;
    const float E1 = nn_energy_packed(site, nearest_neighbors, tid) +
                     nn_energy_packed(mv, nearest_neighbors, tid);
    exchange(site, mv, tid);
    const float E2 = nn_energy_packed(site, nearest_neighbors, tid) +
                     nn_energy_packed(mv, nearest_neighbors, tid);
    const float dE = E2 - E1;
    if (dE <= 0 || uni(generator) < std::exp(-beta * dE))
      continue;
    exchange(site, mv, tid);
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
    std::string s;
    s += "lattice " + std::to_string(tid) +
         " too hot, launching nonlocal sweep\n";
    nonlocal_sweep(L * L * L, beta, generator, indices, uni, nearest_neighbors,
                   tid);
#pragma omp critical
    std::cout << s;
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

  constexpr int num_trials = (int)(.4 * L * L * L);

  if (indices(generator) % 2 == 0) {
    for (int i = 0; i < num_trials; ++i) {
      const int site = hottest_idx[indices(generator) % max_idx];
      const int mv = hottest_idx[indices(generator) % max_idx];
      if (site == -1 || mv == -1)
        continue;
      if (get_value_lattice(site, tid) == get_value_lattice(mv, tid))
        continue;
      if (get_value_lattice(site, tid) == get_value_lattice(mv, tid))
        continue;
      const float E1 = nn_energy_packed(site, nearest_neighbors, tid) +
                       nn_energy_packed(mv, nearest_neighbors, tid);
      exchange(site, mv, tid);
      const float E2 = nn_energy_packed(site, nearest_neighbors, tid) +
                       nn_energy_packed(mv, nearest_neighbors, tid);
      const float dE = E2 - E1;
      if (dE <= 0 || uni(generator) < std::exp(-beta * dE))
        continue;
      exchange(site, mv, tid);
    }
  } else {
    for (int i = 0; i < num_trials; ++i) {
      const int site = hottest_idx[indices(generator) % max_idx];
      if (site == -1)
        continue;
      const auto nb = indices(generator) % NUM_NN;
      const auto mv = nearest_neighbors[NUM_NN * site + nb];
      if (get_value_lattice(site, tid) == get_value_lattice(mv, tid))
        continue;
      if (get_value_lattice(site, tid) == get_value_lattice(mv, tid))
        continue;
      const float E1 = nn_energy_packed(site, nearest_neighbors, tid) +
                       nn_energy_packed(mv, nearest_neighbors, tid);
      exchange(site, mv, tid);
      const float E2 = nn_energy_packed(site, nearest_neighbors, tid) +
                       nn_energy_packed(mv, nearest_neighbors, tid);
      const float dE = E2 - E1;
      if (dE <= 0 || uni(generator) < std::exp(-beta * dE))
        continue;
      exchange(site, mv, tid);
    }
  }
}

void nonlocal_sweep_partitioned(const float &beta, std::mt19937 &generator,
                                std::uniform_int_distribution<> &indices,
                                std::uniform_real_distribution<> &uni,
                                const int *nearest_neighbors, const int &tid) {
  // TODO: also just partition by slices

  // maximally 16 partitions
  const int num_partitions = 1 << (1 + (indices(generator) % 4));
  const int partition_size = L * L * L / num_partitions;

  // keep energy and index together to ease iteration later
  std::vector<std::tuple<int, float>> partition_energies;
  for (int i = 0; i < num_partitions; ++i) {
    const int partition_start = i * partition_size;
    const int partition_end = i * partition_size + partition_size - 1;
    float e = 0.;
    for (int idx = partition_start; idx < partition_end; ++idx)
      e += local_energy_packed(idx, nearest_neighbors, tid);
    partition_energies.push_back(std::make_tuple(partition_start, e));
  }

  // sort from highest energy to lowest energy
  // can then get region with (std::get<0>(a), std::get<0>(a) + partition_size -
  // 1)
  std::sort(partition_energies.begin(), partition_energies.end(),
            [&](std::tuple<int, float> x, std::tuple<int, float> y) {
              return std::get<1>(x) > std::get<1>(y);
            });

  // a) highest <-> lowest
  const int coldest_region_start =
      std::get<0>(partition_energies[num_partitions - 1]);
  const int coldest_region_end = coldest_region_start + partition_size - 1;
  const int warmest_region_start = std::get<0>(partition_energies[0]);
  const int warmest_region_end = warmest_region_start + partition_size - 1;

  // b) highest 2 -- could be done too
  /*
  const int region1_start = std::get<1>(partition_energies[0]);
  const int region1_end   = region1_start + partition_size - 1;
  const int region2_start = std::get<1>(partition_energies[1]);
  const int region2_end   = region1_start + partition_size - 1;
  */

  if (indices(generator) % 2 == 0) {
    for (int i = 0; i < partition_size; ++i) {
      const int site =
          warmest_region_start + (indices(generator) % (partition_size));
      const int mv =
          coldest_region_start + (indices(generator) % (partition_size));
      if (get_value_lattice(site, tid) == get_value_lattice(mv, tid))
        continue;
      const float E1 = nn_energy_packed(site, nearest_neighbors, tid) +
                       nn_energy_packed(mv, nearest_neighbors, tid);
      exchange(site, mv, tid);
      const float E2 = nn_energy_packed(site, nearest_neighbors, tid) +
                       nn_energy_packed(mv, nearest_neighbors, tid);
      const float dE = E2 - E1;
      if (dE <= 0 || uni(generator) < std::exp(-beta * dE))
        continue;
      exchange(site, mv, tid);
    }
  } else {
    for (int i = 0; i < partition_size; ++i) {
      const int site =
          warmest_region_start + (indices(generator) % (partition_size));
      const auto nb = indices(generator) % NUM_NN;
      const auto mv = nearest_neighbors[NUM_NN * site + nb];
      if (get_value_lattice(site, tid) == get_value_lattice(mv, tid))
        continue;
      const float E1 = nn_energy_packed(site, nearest_neighbors, tid) +
                       nn_energy_packed(mv, nearest_neighbors, tid);
      exchange(site, mv, tid);
      const float E2 = nn_energy_packed(site, nearest_neighbors, tid) +
                       nn_energy_packed(mv, nearest_neighbors, tid);
      const float dE = E2 - E1;
      if (dE <= 0 || uni(generator) < std::exp(-beta * dE))
        continue;
      exchange(site, mv, tid);
    }
  }
}

void local_sweep(const float &beta, std::mt19937 &generator,
                 std::uniform_int_distribution<> &indices,
                 std::uniform_real_distribution<> &uni,
                 const int *nearest_neighbors, const int &tid) {

  for (int i = 0; i < L * L * L; ++i) {
    const int site = indices(generator);
    const auto nb = indices(generator) % NUM_NN;
    const auto mv = nearest_neighbors[NUM_NN * site + nb];
    if (get_value_lattice(site, tid) == get_value_lattice(mv, tid))
      continue;
    const float E1 = nn_energy_packed(site, nearest_neighbors, tid) +
                     nn_energy_packed(mv, nearest_neighbors, tid);
    exchange(site, mv, tid);
    const float E2 = nn_energy_packed(site, nearest_neighbors, tid) +
                     nn_energy_packed(mv, nearest_neighbors, tid);
    const float dE = E2 - E1;
    if (dE <= 0 || uni(generator) < std::exp(-beta * dE))
      continue;
    exchange(site, mv, tid);
  }
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cout << "run as: ./bin beta outfile-path\n";
    return 1;
  }
  const auto arg1 = argv[1];
  const auto arg2 = argv[2];
  const auto beta = atof(arg1);
  const auto fname = std::string(arg2);

#pragma omp parallel
  assert(omp_get_num_threads() == NUM_THREADS);

  const float rho = .75;
  // convenience defs, it's actually rho1 = .6 and rho2 = .4
  // ie. here rho1_used = rho1 * rho
  const float rho1 = .45;
  // const float rho2 = .3;
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
    generator.seed(__rdtsc() + tid * tid);

    build_lattice_diag(N, N1, N2, generator, indices, tid);
    // build_lattice(N, N1, N2, generator, indices, tid);
    uint8_t *my_lattice = thread_lattice[tid];
    int *my_nn = thread_nn[tid];
  }
#pragma omp flush

  constexpr size_t max_collect =
      100; // to have sufficient distance when copying in parallel
  constexpr size_t power2 = 12;
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
    generator.seed(__rdtsc() + tid * tid);

    uint8_t *my_lattice = thread_lattice[tid];
    int *my_nn = thread_nn[tid];
    constexpr int TESTP = 7;
    // TODO: infrastructure to set power2 to something nice (cmd args??)
    for (int d = 1; d < 2 * TESTP + 1; ++d) {
      auto t = -omp_get_wtime();
      for (int i = 1 << (d - 1); i < 1 << d; ++i) {
        // nonlocal_sweep(L * L * L, beta, generator, indices, uni, my_nn, tid);
        nonlocal_sweep_partitioned(beta, generator, indices, uni, my_nn, tid);
      }

#pragma omp master
      std::cout << energy(my_nn, tid) << "\n";

      t += omp_get_wtime();
      const auto offset = (tid * max_collect + d) * packed_size;
      uint8_t *copy_spot = config_collection + offset;
      std::copy(my_lattice, my_lattice + packed_size, copy_spot);
#pragma omp master
      cpycounter++;
#pragma omp critical
      times[tid] += t;
    }

#pragma omp master
    std::cout << "\nstarting concentrated sweeps\n\n";

#pragma omp barrier

    for (int d = 1; d < 2 * TESTP + 1; ++d) {
      auto t = -omp_get_wtime();
      float threshold = 1;
      for (int i = 1 << (d - 1); i < 1 << d; ++i) {
        nonlocal_sweep(L * L * L, beta, generator, indices, uni, my_nn, tid);
        concentrated_sweep(threshold, beta, generator, indices, uni, my_nn,
                           tid);
      }

#pragma omp master
      std::cout << energy(my_nn, tid) << "\n";

      t += omp_get_wtime();
      const auto offset = (tid * max_collect + d + 2 * TESTP) * packed_size;
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

  /*
  // this here: _working_ nonlocal sweeps on lattice
  #pragma omp parallel
    {
      const auto tid = omp_get_thread_num();
      auto uni = std::uniform_real_distribution<>(0., 1.);
      auto indices = std::uniform_int_distribution<>(0, max_idx);
      auto generator = std::mt19937();
      generator.seed(__rdtsc() + tid * tid);

      uint8_t * my_lattice = thread_lattice[tid];
      int * my_nn          = thread_nn[tid];
      for (int d = 1; d < power2 + 1; ++d){
        auto t = -omp_get_wtime();
        for (int i = 1 << (d - 1); i < 1 << d; ++i)
        {
          nonlocal_sweep(L * L * L, beta, generator, indices, uni, my_nn, tid);
        }
        t += omp_get_wtime();
        const auto offset = (tid * max_collect + d) * packed_size;
        uint8_t * copy_spot  = config_collection + offset;
        std::copy(my_lattice, my_lattice + packed_size, copy_spot);
  #pragma omp critical
        times[tid] += t;
      }
  #pragma omp barrier
    }
  */

  /* working offsets
  for(size_t t = 0; t < NUM_THREADS; ++t)
   {
       for(size_t d = 0; d < power2 + 1; ++d)
       {
         const auto offset = (t * max_collect + d) * packed_size;
         uint8_t * current_config = config_collection + offset;
         int nred, nblue;
         nred = nblue = 0;
         for(int i=0;i<L*L*L;++i)
         {
             if (static_cast<short>(get_value(current_config, i)) == 1) nred++;
             if (static_cast<short>(get_value(current_config, i)) == 2) nblue++;
         }
         if (N1 != nred)  std::cout << "it's red on "  << t << " epoch=" << d <<
  "\n"; if (N2 != nblue) std::cout << "it's blue on " << t << " epoch=" << d <<
  "\n"; assert(N1 == nred); assert(N2 == nblue);
       }
   }

   short serialized_configs[NUM_THREADS][power2 + 1][L*L*L];
   for(size_t t = 0; t < NUM_THREADS; ++t)
   {

       for(size_t d = 0; d < power2 + 1; ++d)
       {
         const auto offset = (t * max_collect + d) * packed_size;
         uint8_t * current_config = config_collection + offset;
         for(size_t s = 0; s < L*L*L; ++s)
             serialized_configs[t][d][s] =
  static_cast<short>(get_value(current_config, s));
       }
   }
   */

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

  // d.shape = {NUM_THREADS, (power2 + 1),  L, L, L};
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
