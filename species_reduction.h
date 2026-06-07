#ifndef LATTICE_GLASS_SPECIES_REDUCTION_H
#define LATTICE_GLASS_SPECIES_REDUCTION_H

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace lattice_glass {
namespace species {

constexpr int kNumNeighborCounts = 7;
constexpr uint8_t kEmpty = 0;
constexpr uint8_t kType1 = 1;
constexpr uint8_t kType2 = 2;
constexpr long double kLogZero = -1.0e300L;

struct OccupancyStats {
  std::array<int, kNumNeighborCounts> histogram{};
  std::vector<int> occupied_sites;
  std::vector<uint8_t> neighbor_counts;
};

inline bool is_occupied(const uint8_t value) { return value != kEmpty; }

inline int occupied_neighbor_count(const std::vector<uint8_t> &lattice,
                                   const int site,
                                   const int *nearest_neighbors,
                                   const int num_neighbors) {
  int count = 0;
  const int *nn = nearest_neighbors + num_neighbors * site;
  for (int i = 0; i < num_neighbors; ++i)
    count += is_occupied(lattice[nn[i]]);
  return count;
}

inline OccupancyStats compute_occupancy_stats(
    const std::vector<uint8_t> &lattice, const int *nearest_neighbors,
    const int num_neighbors) {
  if (num_neighbors + 1 > kNumNeighborCounts)
    throw std::invalid_argument("neighbor count exceeds species histogram size");

  OccupancyStats stats;
  stats.neighbor_counts.assign(lattice.size(), 0);

  for (int site = 0; site < static_cast<int>(lattice.size()); ++site) {
    if (!is_occupied(lattice[site]))
      continue;

    const int count =
        occupied_neighbor_count(lattice, site, nearest_neighbors, num_neighbors);
    stats.histogram[count]++;
    stats.occupied_sites.push_back(site);
    stats.neighbor_counts[site] = static_cast<uint8_t>(count);
  }
  return stats;
}

inline long double type1_weight(const int occupied_neighbor_count,
                                const long double beta) {
  return std::expl(4.0L * beta * (4 - occupied_neighbor_count));
}

inline long double log_type1_weight(const int occupied_neighbor_count,
                                    const long double beta) {
  return 4.0L * beta * (4 - occupied_neighbor_count);
}

inline std::array<long double, kNumNeighborCounts>
species_weights(const long double beta) {
  std::array<long double, kNumNeighborCounts> weights{};
  for (int count = 0; count < kNumNeighborCounts; ++count)
    weights[count] = type1_weight(count, beta);
  return weights;
}

inline int occupied_count(const std::array<int, kNumNeighborCounts> &histogram) {
  int count = 0;
  for (const int class_count : histogram)
    count += class_count;
  return count;
}

inline void validate_type1_count(
    const std::array<int, kNumNeighborCounts> &histogram,
    const int num_type1) {
  const int num_occupied = occupied_count(histogram);
  if (num_type1 < 0 || num_type1 > num_occupied)
    throw std::invalid_argument("type-1 count is outside occupancy sector");
}

inline long double logsumexp(const long double lhs, const long double rhs) {
  if (lhs <= kLogZero / 2.0L)
    return rhs;
  if (rhs <= kLogZero / 2.0L)
    return lhs;
  const long double pivot = std::max(lhs, rhs);
  return pivot + std::log1pl(std::expl(std::min(lhs, rhs) - pivot));
}

inline std::vector<long double> log_elementary_coefficients_from_histogram(
    const std::array<int, kNumNeighborCounts> &histogram,
    const long double beta, const int max_degree) {
  if (max_degree < 0)
    throw std::invalid_argument("negative polynomial degree cap");

  const long double neg_inf = kLogZero;
  std::vector<long double> log_coeff(max_degree + 1, neg_inf);
  log_coeff[0] = 0.0L;

  int processed = 0;
  for (int count = 0; count < kNumNeighborCounts; ++count) {
    const long double log_weight = log_type1_weight(count, beta);
    for (int repeat = 0; repeat < histogram[count]; ++repeat) {
      const int upper = std::min(max_degree, processed + 1);
      for (int degree = upper; degree >= 1; --degree) {
        log_coeff[degree] =
            logsumexp(log_coeff[degree], log_weight + log_coeff[degree - 1]);
      }
      ++processed;
    }
  }
  return log_coeff;
}

inline std::vector<long double> elementary_coefficients_from_histogram(
    const std::array<int, kNumNeighborCounts> &histogram,
    const long double beta, const int max_degree) {
  const auto log_coeff =
      log_elementary_coefficients_from_histogram(histogram, beta, max_degree);
  std::vector<long double> coeff(log_coeff.size(), 0.0L);
  for (int degree = 0; degree < static_cast<int>(log_coeff.size()); ++degree)
    coeff[degree] = log_coeff[degree] <= kLogZero / 2.0L
                        ? 0.0L
                        : std::expl(log_coeff[degree]);
  return coeff;
}

inline long double log_species_partition_coefficient(
    const std::array<int, kNumNeighborCounts> &histogram,
    const long double beta, const int num_type1) {
  validate_type1_count(histogram, num_type1);
  return log_elementary_coefficients_from_histogram(histogram, beta,
                                                    num_type1)[num_type1];
}

inline long double species_partition_coefficient(
    const std::array<int, kNumNeighborCounts> &histogram,
    const long double beta, const int num_type1) {
  return std::expl(log_species_partition_coefficient(histogram, beta, num_type1));
}

inline long double type2_reference_energy(
    const std::array<int, kNumNeighborCounts> &histogram) {
  long double energy = 0.0L;
  for (int count = 0; count < kNumNeighborCounts; ++count) {
    const int delta = count - 5;
    energy += static_cast<long double>(histogram[count] * delta * delta);
  }
  return energy;
}

inline long double effective_hamiltonian(
    const std::array<int, kNumNeighborCounts> &histogram,
    const long double beta, const int num_type1) {
  if (beta <= 0.0L)
    throw std::invalid_argument("effective Hamiltonian requires beta > 0");

  const long double log_coefficient =
      log_species_partition_coefficient(histogram, beta, num_type1);
  if (!std::isfinite(log_coefficient))
    throw std::overflow_error("species partition coefficient is not finite");

  return type2_reference_energy(histogram) - log_coefficient / beta;
}

inline std::array<long double, kNumNeighborCounts>
class_type1_probabilities(const std::array<int, kNumNeighborCounts> &histogram,
                          const long double beta, const int num_type1) {
  validate_type1_count(histogram, num_type1);

  std::array<long double, kNumNeighborCounts> probabilities{};
  if (num_type1 == 0)
    return probabilities;

  const long double log_denominator =
      log_species_partition_coefficient(histogram, beta, num_type1);
  if (!std::isfinite(log_denominator))
    throw std::runtime_error("invalid species partition coefficient");

  for (int count = 0; count < kNumNeighborCounts; ++count) {
    if (histogram[count] == 0)
      continue;

    auto reduced_histogram = histogram;
    reduced_histogram[count]--;
    const long double log_other =
        log_species_partition_coefficient(reduced_histogram, beta, num_type1 - 1);
    const long double log_probability =
        log_type1_weight(count, beta) + log_other - log_denominator;
    probabilities[count] = std::expl(log_probability);
    if (probabilities[count] < 0.0L && probabilities[count] > -1e-12L)
      probabilities[count] = 0.0L;
    if (probabilities[count] > 1.0L && probabilities[count] < 1.0L + 1e-12L)
      probabilities[count] = 1.0L;
  }
  return probabilities;
}

inline std::vector<long double> site_type1_marginals(
    const OccupancyStats &stats, const long double beta, const int num_type1) {
  const auto class_probabilities =
      class_type1_probabilities(stats.histogram, beta, num_type1);

  std::vector<long double> marginals(stats.neighbor_counts.size(), 0.0L);
  for (const int site : stats.occupied_sites)
    marginals[site] = class_probabilities[stats.neighbor_counts[site]];
  return marginals;
}

inline std::vector<uint8_t> sample_species_given_occupancy(
    const std::vector<uint8_t> &occupancy, const int *nearest_neighbors,
    const int num_neighbors, const long double beta, const int num_type1,
    std::mt19937 &generator) {
  const OccupancyStats stats =
      compute_occupancy_stats(occupancy, nearest_neighbors, num_neighbors);
  validate_type1_count(stats.histogram, num_type1);

  std::vector<uint8_t> sample(occupancy.size(), kEmpty);
  if (stats.occupied_sites.empty())
    return sample;

  const int num_occupied = static_cast<int>(stats.occupied_sites.size());
  const long double neg_inf = kLogZero;
  std::vector<std::vector<long double>> suffix(
      num_occupied + 1, std::vector<long double>(num_type1 + 1, neg_inf));
  suffix[num_occupied][0] = 0.0L;
  for (int pos = num_occupied - 1; pos >= 0; --pos) {
    const int site = stats.occupied_sites[pos];
    const long double log_weight =
        log_type1_weight(stats.neighbor_counts[site], beta);
    suffix[pos][0] = 0.0L;
    const int remaining = num_occupied - pos;
    const int upper = std::min(num_type1, remaining);
    for (int chosen = 1; chosen <= upper; ++chosen) {
      suffix[pos][chosen] = logsumexp(
          suffix[pos + 1][chosen], log_weight + suffix[pos + 1][chosen - 1]);
    }
  }

  std::uniform_real_distribution<long double> uniform(0.0L, 1.0L);
  int need_type1 = num_type1;
  for (int pos = 0; pos < num_occupied; ++pos) {
    const int site = stats.occupied_sites[pos];
    const int remaining_after = num_occupied - pos - 1;

    if (need_type1 == 0) {
      sample[site] = kType2;
      continue;
    }
    if (need_type1 > remaining_after) {
      sample[site] = kType1;
      --need_type1;
      continue;
    }

    const long double log_weight =
        log_type1_weight(stats.neighbor_counts[site], beta);
    const long double log_choose_type1 =
        log_weight + suffix[pos + 1][need_type1 - 1];
    const long double log_total = suffix[pos][need_type1];
    const long double probability = std::expl(log_choose_type1 - log_total);
    if (uniform(generator) < probability) {
      sample[site] = kType1;
      --need_type1;
    } else {
      sample[site] = kType2;
    }
  }

  return sample;
}

inline long double rao_blackwell_site_overlap(
    const uint8_t occ1, const long double type1_probability1,
    const uint8_t occ2, const long double type1_probability2) {
  const bool occupied1 = is_occupied(occ1);
  const bool occupied2 = is_occupied(occ2);
  if (!occupied1 && !occupied2)
    return 1.0L;
  if (occupied1 != occupied2)
    return 0.0L;

  return type1_probability1 * type1_probability2 +
         (1.0L - type1_probability1) * (1.0L - type1_probability2);
}

inline long double rao_blackwell_overlap_sum(
    const std::vector<uint8_t> &occupancy1,
    const std::vector<long double> &type1_probabilities1,
    const std::vector<uint8_t> &occupancy2,
    const std::vector<long double> &type1_probabilities2) {
  if (occupancy1.size() != occupancy2.size() ||
      occupancy1.size() != type1_probabilities1.size() ||
      occupancy2.size() != type1_probabilities2.size())
    throw std::invalid_argument("overlap inputs must have equal lengths");

  long double overlap = 0.0L;
  for (int site = 0; site < static_cast<int>(occupancy1.size()); ++site) {
    overlap += rao_blackwell_site_overlap(
        occupancy1[site], type1_probabilities1[site], occupancy2[site],
        type1_probabilities2[site]);
  }
  return overlap;
}

} // namespace species
} // namespace lattice_glass

#endif // LATTICE_GLASS_SPECIES_REDUCTION_H
