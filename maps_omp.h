#include <omp.h>

#include <array>
#include <iostream>
#include <stdint.h>
#include <stdlib.h>

#define L 8
#define NUM_NN 6
#define NUM_THREADS 4

constexpr int lat_size = L * L * L;
constexpr int pad_size = 128;
constexpr int packed_size = (L * L * L / 4 + pad_size - 1) & ~(pad_size - 1);
constexpr int packed_nb_size =
    (lat_size * NUM_NN + (pad_size - 1)) & ~(pad_size - 1);
constexpr int lat_size_packed = NUM_THREADS * packed_size;
uint8_t packed_lattice[lat_size_packed];
int nearest_neighbors[NUM_THREADS * packed_nb_size];

uint8_t *thread_lattice[NUM_THREADS];
int *thread_nn[NUM_THREADS];

uint8_t get_value(uint8_t *lattice, const int &s) {
  const int idx = s >> 2;
  const int offset = (s & 3) * 2;
  return (lattice[idx] >> offset) & 3;
}

void set_value(uint8_t *lattice, const int &s, const uint8_t &value) {
  const int idx = s >> 2;
  const int offset = (s & 3) * 2;
  lattice[idx] &= ~(3 << offset);
  lattice[idx] |= ((value & 3) << offset);
}

uint8_t get_value_lattice(const int &idx_in, const int &tid) {
  // explainer for the operations below:
  // const int idx = idx_in / 4;
  // const int offset = (idx_in % 4) * 2;
  const int idx = idx_in >> 2;
  const int offset = (idx_in & 3) * 2;
  return (thread_lattice[tid][idx] >> offset) & 3;
}

void set_value_lattice(const int &idx_in, const uint8_t &value,
                       const int &tid) {
  const int idx = idx_in >> 2;
  const int offset = (idx_in & 3) * 2;
  // unset
  thread_lattice[tid][idx] &= ~(3 << offset);
  // set
  thread_lattice[tid][idx] |= ((value & 3) << offset);
}

void build_nn_table() {
  for (int t = 0; t < NUM_THREADS; ++t) {
    for (int i = 0; i < L; ++i)
      for (int j = 0; j < L; ++j)
        for (int k = 0; k < L; ++k) {
          const auto site = k + L * (j + i * L);
          int ii, kk, jj;
          ii = (i == 0 ? L - 1 : i - 1);
          jj = (j == 0 ? L - 1 : j - 1);
          kk = (k == 0 ? L - 1 : k - 1);
          const int f = k + L * (j + (ii % L) * L);
          const int b = k + L * (j + ((i + 1) % L) * L);
          const int u = k + L * ((jj % L) + i * L);
          const int d = k + L * (((j + 1) % L) + i * L);
          const int l = (kk % L) + L * (j + i * L);
          const int r = ((k + 1) % L) + L * (j + i * L);
          nearest_neighbors[t * packed_nb_size + NUM_NN * site + 0] = f;
          nearest_neighbors[t * packed_nb_size + NUM_NN * site + 1] = b;
          nearest_neighbors[t * packed_nb_size + NUM_NN * site + 2] = u;
          nearest_neighbors[t * packed_nb_size + NUM_NN * site + 3] = d;
          nearest_neighbors[t * packed_nb_size + NUM_NN * site + 4] = l;
          nearest_neighbors[t * packed_nb_size + NUM_NN * site + 5] = r;
        }
  }
}

void generate_tables() {
  for (int t = 0; t < NUM_THREADS; ++t) {
    const auto lattice_offset = t * packed_size;
    const auto nn_offset = t * packed_nb_size;
    thread_lattice[t] = &packed_lattice[lattice_offset];
    thread_nn[t] = &nearest_neighbors[nn_offset];
  }
  build_nn_table();
}
