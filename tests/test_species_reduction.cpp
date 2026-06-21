#include "../species_reduction.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace species = lattice_glass::species;

static std::vector<int> ring_neighbors(const int sites) {
  std::vector<int> nn(sites * 2);
  for (int site = 0; site < sites; ++site) {
    nn[2 * site] = (site + sites - 1) % sites;
    nn[2 * site + 1] = (site + 1) % sites;
  }
  return nn;
}

static long double direct_coefficient(const std::vector<int> &counts,
                                      const long double beta,
                                      const int num_type1) {
  const auto weights = species::species_weights(beta);
  long double total = 0.0L;
  const int n = static_cast<int>(counts.size());
  for (int mask = 0; mask < (1 << n); ++mask) {
    if (__builtin_popcount(static_cast<unsigned int>(mask)) != num_type1)
      continue;
    long double weight = 1.0L;
    for (int i = 0; i < n; ++i)
      if (mask & (1 << i))
        weight *= weights[counts[i]];
    total += weight;
  }
  return total;
}

static void assert_close(const long double lhs, const long double rhs,
                         const long double rel_tol = 1e-10L) {
  const long double scale = 1.0L + std::max(std::fabsl(lhs), std::fabsl(rhs));
  assert(std::fabsl(lhs - rhs) <= rel_tol * scale);
}

static void test_partition_and_marginals_match_enumeration() {
  const long double beta = 0.73L;
  const int num_type1 = 3;
  const std::vector<int> counts = {0, 1, 2, 3, 4, 5, 6, 2};

  std::array<int, species::kNumNeighborCounts> histogram{};
  for (const int count : counts)
    histogram[count]++;

  const long double dp =
      species::species_partition_coefficient(histogram, beta, num_type1);
  const long double enumerated = direct_coefficient(counts, beta, num_type1);
  assert_close(dp, enumerated);

  const auto class_probabilities =
      species::class_type1_probabilities(histogram, beta, num_type1);
  const auto weights = species::species_weights(beta);

  for (int site = 0; site < static_cast<int>(counts.size()); ++site) {
    long double numerator = 0.0L;
    const int n = static_cast<int>(counts.size());
    for (int mask = 0; mask < (1 << n); ++mask) {
      if (!(mask & (1 << site)))
        continue;
      if (__builtin_popcount(static_cast<unsigned int>(mask)) != num_type1)
        continue;
      long double weight = 1.0L;
      for (int i = 0; i < n; ++i)
        if (mask & (1 << i))
          weight *= weights[counts[i]];
      numerator += weight;
    }
    assert_close(class_probabilities[counts[site]], numerator / enumerated);
  }
}

static void test_occupancy_stats_and_sampler_preserve_sector() {
  const std::vector<int> nn = ring_neighbors(8);
  const std::vector<uint8_t> occupancy = {1, 0, 1, 1, 0, 1, 0, 1};
  const auto stats = species::compute_occupancy_stats(occupancy, nn.data(), 2);

  assert(stats.occupied_sites.size() == 5);
  assert(species::occupied_count(stats.histogram) == 5);
  for (const int site : stats.occupied_sites)
    assert(stats.neighbor_counts[site] <= 2);

  std::mt19937 generator(1234);
  for (int trial = 0; trial < 64; ++trial) {
    const std::vector<uint8_t> sample = species::sample_species_given_occupancy(
        occupancy, nn.data(), 2, 0.9L, 2, generator);
    int type1 = 0;
    int type2 = 0;
    for (int site = 0; site < static_cast<int>(sample.size()); ++site) {
      if (occupancy[site] == 0) {
        assert(sample[site] == species::kEmpty);
      } else if (sample[site] == species::kType1) {
        ++type1;
      } else if (sample[site] == species::kType2) {
        ++type2;
      } else {
        assert(false);
      }
    }
    assert(type1 == 2);
    assert(type2 == 3);
  }
}

struct AssignmentWeight {
  std::vector<uint8_t> assignment;
  long double weight;
};

static std::vector<AssignmentWeight> enumerate_assignments(
    const std::vector<uint8_t> &occupancy, const std::vector<long double> &site_weights,
    const int num_type1) {
  std::vector<int> occupied_sites;
  for (int site = 0; site < static_cast<int>(occupancy.size()); ++site)
    if (occupancy[site] != 0)
      occupied_sites.push_back(site);

  std::vector<AssignmentWeight> result;
  const int n = static_cast<int>(occupied_sites.size());
  for (int mask = 0; mask < (1 << n); ++mask) {
    if (__builtin_popcount(static_cast<unsigned int>(mask)) != num_type1)
      continue;
    AssignmentWeight current;
    current.assignment.assign(occupancy.size(), 0);
    current.weight = 1.0L;
    for (int i = 0; i < n; ++i) {
      const int site = occupied_sites[i];
      if (mask & (1 << i)) {
        current.assignment[site] = species::kType1;
        current.weight *= site_weights[site];
      } else {
        current.assignment[site] = species::kType2;
      }
    }
    result.push_back(current);
  }
  return result;
}

static void test_rao_blackwell_overlap_matches_enumeration() {
  const std::vector<int> nn = ring_neighbors(6);
  const long double beta = 0.41L;
  const std::vector<uint8_t> occ1 = {1, 1, 0, 1, 0, 1};
  const std::vector<uint8_t> occ2 = {1, 0, 1, 1, 0, 1};

  const auto stats1 = species::compute_occupancy_stats(occ1, nn.data(), 2);
  const auto stats2 = species::compute_occupancy_stats(occ2, nn.data(), 2);
  const auto marginals1 = species::site_type1_marginals(stats1, beta, 2);
  const auto marginals2 = species::site_type1_marginals(stats2, beta, 1);

  const long double rb = species::rao_blackwell_overlap_sum(
      occ1, marginals1, occ2, marginals2);

  const auto weights = species::species_weights(beta);
  std::vector<long double> site_weights1(occ1.size(), 0.0L);
  std::vector<long double> site_weights2(occ2.size(), 0.0L);
  for (int site = 0; site < static_cast<int>(occ1.size()); ++site) {
    site_weights1[site] = weights[stats1.neighbor_counts[site]];
    site_weights2[site] = weights[stats2.neighbor_counts[site]];
  }

  const auto assignments1 = enumerate_assignments(occ1, site_weights1, 2);
  const auto assignments2 = enumerate_assignments(occ2, site_weights2, 1);

  long double numerator = 0.0L;
  long double denominator = 0.0L;
  for (const auto &a1 : assignments1) {
    for (const auto &a2 : assignments2) {
      long double overlap = 0.0L;
      for (int site = 0; site < static_cast<int>(occ1.size()); ++site)
        overlap += (a1.assignment[site] == a2.assignment[site]);
      numerator += a1.weight * a2.weight * overlap;
      denominator += a1.weight * a2.weight;
    }
  }

  assert_close(rb, numerator / denominator);
}

static void test_rao_blackwell_energy_matches_enumeration() {
  const std::vector<int> nn = ring_neighbors(8);
  const long double beta = 0.6L;
  const std::vector<uint8_t> occ = {1, 1, 0, 1, 1, 0, 1, 1};
  const int num_type1 = 3;

  const auto stats = species::compute_occupancy_stats(occ, nn.data(), 2);
  const long double rb = species::rao_blackwell_energy(stats, beta, num_type1);

  const auto weights = species::species_weights(beta);
  std::vector<long double> site_weights(occ.size(), 0.0L);
  for (int site = 0; site < static_cast<int>(occ.size()); ++site)
    site_weights[site] = weights[stats.neighbor_counts[site]];

  const auto assignments = enumerate_assignments(occ, site_weights, num_type1);
  long double numerator = 0.0L;
  long double denominator = 0.0L;
  for (const auto &a : assignments) {
    long double h = 0.0L;
    for (int site = 0; site < static_cast<int>(occ.size()); ++site) {
      if (a.assignment[site] == species::kEmpty)
        continue;
      const int m = stats.neighbor_counts[site];
      const int l = a.assignment[site] == species::kType1 ? 3 : 5;
      h += static_cast<long double>((m - l) * (m - l));
    }
    numerator += a.weight * h;
    denominator += a.weight;
  }

  assert_close(rb, numerator / denominator);
}

int main() {
  test_partition_and_marginals_match_enumeration();
  test_occupancy_stats_and_sampler_preserve_sector();
  test_rao_blackwell_overlap_matches_enumeration();
  test_rao_blackwell_energy_matches_enumeration();
  std::cout << "species reduction tests passed\n";
  return 0;
}
