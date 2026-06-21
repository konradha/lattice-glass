#ifndef LATTICE_GLASS_BALANCED_CLUSTER_H
#define LATTICE_GLASS_BALANCED_CLUSTER_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lattice_glass {
namespace cluster {

constexpr int kAlphabetSize = 3;
constexpr uint8_t kEmpty = 0;
constexpr uint8_t kType1 = 1;
constexpr uint8_t kType2 = 2;

struct ReplicaPair {
  std::vector<uint8_t> first;
  std::vector<uint8_t> second;
};

struct GrowthRule {
  long double kappa = 0.0L;
  // Invalid for production kernels. This exists only as a positive-control
  // switch for reversibility tests of the biased rule excluded in the proof.
  bool use_signed_weight_for_bias_test = false;
  bool restrict_unordered_type = false;
  uint8_t restricted_a = kEmpty;
  uint8_t restricted_b = kType1;
};

struct GrowthAttempt {
  bool proposed = false;
  bool abandoned = false;
  std::vector<int> cluster_sites;
  long double log_probability = 0.0L;
};

struct PathProbability {
  bool valid = false;
  long double log_probability = 0.0L;
};

struct MetropolisResult {
  bool proposed = false;
  bool accepted = false;
  long double delta_energy = 0.0L;
  int cluster_size = 0;
};

inline bool is_occupied(const uint8_t value) { return value != kEmpty; }

inline void validate_pair(const ReplicaPair &pair) {
  if (pair.first.size() != pair.second.size())
    throw std::invalid_argument("replicas must have equal sizes");
}

inline std::pair<uint8_t, uint8_t> unordered_type(const uint8_t a,
                                                  const uint8_t b) {
  return std::minmax(a, b);
}

inline std::array<int, kAlphabetSize> symbol_increment(const uint8_t before,
                                                       const uint8_t after) {
  std::array<int, kAlphabetSize> increment{};
  if (before == after)
    return increment;
  increment[before]--;
  increment[after]++;
  return increment;
}

inline std::array<int, kAlphabetSize> site_increment(const ReplicaPair &pair,
                                                     const int site) {
  return symbol_increment(pair.first[site], pair.second[site]);
}

inline void add_increment(std::array<int, kAlphabetSize> &accumulator,
                          const std::array<int, kAlphabetSize> &increment) {
  for (int symbol = 0; symbol < kAlphabetSize; ++symbol)
    accumulator[symbol] += increment[symbol];
}

inline std::array<int, kAlphabetSize> cluster_increment(
    const ReplicaPair &pair, const std::vector<int> &cluster_sites) {
  std::array<int, kAlphabetSize> increment{};
  for (const int site : cluster_sites)
    add_increment(increment, site_increment(pair, site));
  return increment;
}

inline bool is_zero_increment(
    const std::array<int, kAlphabetSize> &increment) {
  for (const int value : increment)
    if (value != 0)
      return false;
  return true;
}

inline bool is_balanced_cluster(const ReplicaPair &pair,
                                const std::vector<int> &cluster_sites) {
  return is_zero_increment(cluster_increment(pair, cluster_sites));
}

inline int increment_dot(const std::array<int, kAlphabetSize> &lhs,
                         const std::array<int, kAlphabetSize> &rhs) {
  int result = 0;
  for (int symbol = 0; symbol < kAlphabetSize; ++symbol)
    result += lhs[symbol] * rhs[symbol];
  return result;
}

inline std::array<int, kAlphabetSize>
composition_counts(const std::vector<uint8_t> &replica) {
  std::array<int, kAlphabetSize> counts{};
  for (const uint8_t value : replica)
    counts[value]++;
  return counts;
}

inline bool compositions_equal(const std::vector<uint8_t> &lhs,
                               const std::vector<uint8_t> &rhs) {
  return composition_counts(lhs) == composition_counts(rhs);
}

inline void exchange_cluster(ReplicaPair &pair,
                             const std::vector<int> &cluster_sites) {
  for (const int site : cluster_sites)
    std::swap(pair.first[site], pair.second[site]);
}

inline bool site_in_growth_domain(const ReplicaPair &pair, const int site,
                                  const GrowthRule &rule) {
  if (pair.first[site] == pair.second[site])
    return false;

  if (!rule.restrict_unordered_type)
    return true;

  const auto actual = unordered_type(pair.first[site], pair.second[site]);
  const auto required = unordered_type(rule.restricted_a, rule.restricted_b);
  return actual == required;
}

inline std::vector<int> disagreement_sites(const ReplicaPair &pair,
                                           const GrowthRule &rule) {
  validate_pair(pair);
  std::vector<int> sites;
  for (int site = 0; site < static_cast<int>(pair.first.size()); ++site)
    if (site_in_growth_domain(pair, site, rule))
      sites.push_back(site);
  return sites;
}

inline void append_unique(std::vector<int> &values, const int value) {
  if (std::find(values.begin(), values.end(), value) == values.end())
    values.push_back(value);
}

inline std::vector<int> frontier_sites(const ReplicaPair &pair,
                                       const std::vector<int> &cluster_sites,
                                       const std::vector<uint8_t> &in_cluster,
                                       const int *nearest_neighbors,
                                       const int num_neighbors,
                                       const GrowthRule &rule) {
  std::vector<int> frontier;
  for (const int site : cluster_sites) {
    const int *nn = nearest_neighbors + num_neighbors * site;
    for (int offset = 0; offset < num_neighbors; ++offset) {
      const int neighbor = nn[offset];
      if (in_cluster[neighbor])
        continue;
      if (site_in_growth_domain(pair, neighbor, rule))
        append_unique(frontier, neighbor);
    }
  }
  return frontier;
}

inline long double growth_weight(
    const ReplicaPair &pair, const int site,
    const std::array<int, kAlphabetSize> &imbalance, const GrowthRule &rule) {
  const int signed_alignment =
      increment_dot(site_increment(pair, site), imbalance);
  const int feature = rule.use_signed_weight_for_bias_test
                          ? signed_alignment
                          : std::abs(signed_alignment);
  return std::expl(rule.kappa * static_cast<long double>(feature));
}

inline PathProbability proposal_path_log_probability(
    const ReplicaPair &pair, const std::vector<int> &path,
    const int *nearest_neighbors, const int num_neighbors,
    const GrowthRule &rule) {
  validate_pair(pair);
  if (path.empty())
    return {};

  const std::vector<int> seeds = disagreement_sites(pair, rule);
  if (seeds.empty() ||
      std::find(seeds.begin(), seeds.end(), path[0]) == seeds.end())
    return {};

  std::vector<uint8_t> in_cluster(pair.first.size(), 0);
  std::vector<int> cluster_sites;
  cluster_sites.push_back(path[0]);
  in_cluster[path[0]] = 1;

  std::array<int, kAlphabetSize> imbalance = site_increment(pair, path[0]);
  long double log_probability =
      -std::log(static_cast<long double>(seeds.size()));

  for (int step = 1; step < static_cast<int>(path.size()); ++step) {
    if (is_zero_increment(imbalance))
      return {};

    const std::vector<int> frontier = frontier_sites(
        pair, cluster_sites, in_cluster, nearest_neighbors, num_neighbors, rule);
    if (frontier.empty())
      return {};

    long double total_weight = 0.0L;
    long double selected_weight = 0.0L;
    for (const int site : frontier) {
      const long double weight = growth_weight(pair, site, imbalance, rule);
      total_weight += weight;
      if (site == path[step])
        selected_weight = weight;
    }
    if (!(selected_weight > 0.0L) || !(total_weight > 0.0L))
      return {};

    log_probability += std::log(selected_weight / total_weight);
    cluster_sites.push_back(path[step]);
    in_cluster[path[step]] = 1;
    add_increment(imbalance, site_increment(pair, path[step]));
  }

  if (!is_zero_increment(imbalance))
    return {};

  return {true, log_probability};
}

inline GrowthAttempt grow_balanced_cluster(
    const ReplicaPair &pair, const int *nearest_neighbors, const int num_neighbors,
    const int max_cluster_size, const GrowthRule &rule,
    std::mt19937 &generator) {
  validate_pair(pair);
  if (max_cluster_size <= 0)
    throw std::invalid_argument("max cluster size must be positive");

  const std::vector<int> seeds = disagreement_sites(pair, rule);
  if (seeds.empty())
    return {};

  std::uniform_int_distribution<int> seed_distribution(
      0, static_cast<int>(seeds.size()) - 1);
  const int seed = seeds[seed_distribution(generator)];

  GrowthAttempt attempt;
  attempt.log_probability = -std::log(static_cast<long double>(seeds.size()));
  attempt.cluster_sites.push_back(seed);

  std::vector<uint8_t> in_cluster(pair.first.size(), 0);
  in_cluster[seed] = 1;
  std::array<int, kAlphabetSize> imbalance = site_increment(pair, seed);

  std::uniform_real_distribution<long double> uniform(0.0L, 1.0L);
  while (!is_zero_increment(imbalance)) {
    if (static_cast<int>(attempt.cluster_sites.size()) >= max_cluster_size) {
      attempt.abandoned = true;
      return attempt;
    }

    const std::vector<int> frontier = frontier_sites(
        pair, attempt.cluster_sites, in_cluster, nearest_neighbors, num_neighbors,
        rule);
    if (frontier.empty()) {
      attempt.abandoned = true;
      return attempt;
    }

    std::vector<long double> weights;
    weights.reserve(frontier.size());
    long double total_weight = 0.0L;
    for (const int site : frontier) {
      const long double weight = growth_weight(pair, site, imbalance, rule);
      weights.push_back(weight);
      total_weight += weight;
    }

    long double threshold = uniform(generator) * total_weight;
    int chosen_index = static_cast<int>(frontier.size()) - 1;
    for (int i = 0; i < static_cast<int>(frontier.size()); ++i) {
      threshold -= weights[i];
      if (threshold <= 0.0L) {
        chosen_index = i;
        break;
      }
    }

    attempt.log_probability += std::log(weights[chosen_index] / total_weight);
    const int chosen_site = frontier[chosen_index];
    attempt.cluster_sites.push_back(chosen_site);
    in_cluster[chosen_site] = 1;
    add_increment(imbalance, site_increment(pair, chosen_site));
  }

  attempt.proposed = true;
  return attempt;
}

inline long double local_energy(const std::vector<uint8_t> &replica,
                                const int site, const int *nearest_neighbors,
                                const int num_neighbors) {
  const uint8_t type = replica[site];
  if (!is_occupied(type))
    return 0.0L;

  const int preferred = type == kType1 ? 3 : 5;
  int occupied_neighbors = 0;
  const int *nn = nearest_neighbors + num_neighbors * site;
  for (int i = 0; i < num_neighbors; ++i)
    occupied_neighbors += is_occupied(replica[nn[i]]);

  const int delta = occupied_neighbors - preferred;
  return static_cast<long double>(delta * delta);
}

inline long double total_energy(const std::vector<uint8_t> &replica,
                                const int *nearest_neighbors,
                                const int num_neighbors) {
  long double energy = 0.0L;
  for (int site = 0; site < static_cast<int>(replica.size()); ++site)
    energy += local_energy(replica, site, nearest_neighbors, num_neighbors);
  return energy;
}

inline std::vector<int> cluster_affected_sites(
    const std::vector<int> &cluster_sites, const int *nearest_neighbors,
    const int num_neighbors) {
  std::vector<int> affected;
  for (const int site : cluster_sites) {
    append_unique(affected, site);
    const int *nn = nearest_neighbors + num_neighbors * site;
    for (int offset = 0; offset < num_neighbors; ++offset)
      append_unique(affected, nn[offset]);
  }
  return affected;
}

inline long double affected_energy(const std::vector<uint8_t> &replica,
                                   const std::vector<int> &affected_sites,
                                   const int *nearest_neighbors,
                                   const int num_neighbors) {
  long double energy = 0.0L;
  for (const int site : affected_sites)
    energy += local_energy(replica, site, nearest_neighbors, num_neighbors);
  return energy;
}

inline long double cluster_exchange_delta(ReplicaPair &pair,
                                          const std::vector<int> &cluster_sites,
                                          const int *nearest_neighbors,
                                          const int num_neighbors) {
  validate_pair(pair);
  const std::vector<int> affected =
      cluster_affected_sites(cluster_sites, nearest_neighbors, num_neighbors);

  const long double before =
      affected_energy(pair.first, affected, nearest_neighbors, num_neighbors) +
      affected_energy(pair.second, affected, nearest_neighbors, num_neighbors);
  exchange_cluster(pair, cluster_sites);
  const long double after =
      affected_energy(pair.first, affected, nearest_neighbors, num_neighbors) +
      affected_energy(pair.second, affected, nearest_neighbors, num_neighbors);
  exchange_cluster(pair, cluster_sites);

  return after - before;
}

inline MetropolisResult metropolis_cluster_exchange(
    ReplicaPair &pair, const long double beta, const int *nearest_neighbors,
    const int num_neighbors, const int max_cluster_size, const GrowthRule &rule,
    std::mt19937 &generator) {
  GrowthAttempt attempt = grow_balanced_cluster(
      pair, nearest_neighbors, num_neighbors, max_cluster_size, rule, generator);

  MetropolisResult result;
  result.proposed = attempt.proposed;
  result.cluster_size = static_cast<int>(attempt.cluster_sites.size());
  if (!attempt.proposed)
    return result;

  result.delta_energy = cluster_exchange_delta(pair, attempt.cluster_sites,
                                               nearest_neighbors, num_neighbors);
  std::uniform_real_distribution<long double> uniform(0.0L, 1.0L);
  if (result.delta_energy <= 0.0L ||
      uniform(generator) < std::expl(-beta * result.delta_energy)) {
    exchange_cluster(pair, attempt.cluster_sites);
    result.accepted = true;
  }
  return result;
}

} // namespace cluster
} // namespace lattice_glass

#endif // LATTICE_GLASS_BALANCED_CLUSTER_H
