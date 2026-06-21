#include "../balanced_cluster.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace cluster = lattice_glass::cluster;

static std::vector<int> ring_neighbors(const int sites) {
  std::vector<int> nn(sites * 2);
  for (int site = 0; site < sites; ++site) {
    nn[2 * site] = (site + sites - 1) % sites;
    nn[2 * site + 1] = (site + 1) % sites;
  }
  return nn;
}

static void assert_close(const long double lhs, const long double rhs,
                         const long double tol = 1e-12L) {
  const long double scale = 1.0L + std::max(std::fabsl(lhs), std::fabsl(rhs));
  assert(std::fabsl(lhs - rhs) <= tol * scale);
}

static cluster::ReplicaPair proposal_fixture() {
  cluster::ReplicaPair pair;
  pair.first = {0, 1, 2, 1, 0};
  pair.second = {1, 2, 0, 0, 1};
  return pair;
}

static void test_balance_and_composition_conservation() {
  cluster::ReplicaPair pair = proposal_fixture();
  const std::vector<int> balanced = {0, 1, 2};
  const std::vector<int> unbalanced = {0, 1};

  assert(cluster::is_balanced_cluster(pair, balanced));
  assert(!cluster::is_balanced_cluster(pair, unbalanced));

  const auto first_counts = cluster::composition_counts(pair.first);
  const auto second_counts = cluster::composition_counts(pair.second);
  cluster::exchange_cluster(pair, balanced);
  assert(cluster::composition_counts(pair.first) == first_counts);
  assert(cluster::composition_counts(pair.second) == second_counts);

  cluster::exchange_cluster(pair, balanced);
  assert(pair.first == proposal_fixture().first);
  assert(pair.second == proposal_fixture().second);
}

static void test_cluster_delta_matches_full_energy() {
  cluster::ReplicaPair pair = proposal_fixture();
  const std::vector<int> nn = ring_neighbors(5);
  const std::vector<int> sites = {0, 1, 2};

  const long double before = cluster::total_energy(pair.first, nn.data(), 2) +
                             cluster::total_energy(pair.second, nn.data(), 2);
  const long double delta =
      cluster::cluster_exchange_delta(pair, sites, nn.data(), 2);
  cluster::exchange_cluster(pair, sites);
  const long double after = cluster::total_energy(pair.first, nn.data(), 2) +
                            cluster::total_energy(pair.second, nn.data(), 2);

  assert_close(delta, after - before);
}

static void test_proposal_symmetry_and_signed_positive_control() {
  const std::vector<int> nn = ring_neighbors(5);
  const std::vector<int> path = {0, 1, 2};
  const std::vector<int> final_cluster = path;

  cluster::GrowthRule invariant_rule;
  invariant_rule.kappa = 0.7L;

  cluster::ReplicaPair forward = proposal_fixture();
  cluster::ReplicaPair reverse = proposal_fixture();
  cluster::exchange_cluster(reverse, final_cluster);

  const auto forward_probability = cluster::proposal_path_log_probability(
      forward, path, nn.data(), 2, invariant_rule);
  const auto reverse_probability = cluster::proposal_path_log_probability(
      reverse, path, nn.data(), 2, invariant_rule);
  assert(forward_probability.valid);
  assert(reverse_probability.valid);
  assert_close(forward_probability.log_probability,
               reverse_probability.log_probability);

  cluster::GrowthRule signed_rule = invariant_rule;
  signed_rule.use_signed_weight_for_bias_test = true;
  const auto signed_forward = cluster::proposal_path_log_probability(
      forward, path, nn.data(), 2, signed_rule);
  const auto signed_reverse = cluster::proposal_path_log_probability(
      reverse, path, nn.data(), 2, signed_rule);
  assert(signed_forward.valid);
  assert(signed_reverse.valid);
  assert(std::fabsl(signed_forward.log_probability -
                    signed_reverse.log_probability) > 1e-3L);
}

static void test_restricted_binary_domain() {
  const cluster::ReplicaPair pair = proposal_fixture();

  cluster::GrowthRule unrestricted;
  const auto all = cluster::disagreement_sites(pair, unrestricted);
  assert(all.size() == 5);

  cluster::GrowthRule zero_one;
  zero_one.restrict_unordered_type = true;
  zero_one.restricted_a = 0;
  zero_one.restricted_b = 1;
  const auto binary = cluster::disagreement_sites(pair, zero_one);
  assert(binary.size() == 3);
  for (const int site : binary)
    assert(cluster::unordered_type(pair.first[site], pair.second[site]) ==
           cluster::unordered_type(0, 1));
}

static void test_metropolis_step_preserves_counts_when_accepted() {
  const std::vector<int> nn = ring_neighbors(5);
  cluster::ReplicaPair pair = proposal_fixture();
  const auto first_counts = cluster::composition_counts(pair.first);
  const auto second_counts = cluster::composition_counts(pair.second);

  cluster::GrowthRule rule;
  rule.kappa = 0.0L;
  std::mt19937 generator(99);
  for (int trial = 0; trial < 32; ++trial) {
    const auto result = cluster::metropolis_cluster_exchange(
        pair, 0.0L, nn.data(), 2, 5, rule, generator);
    if (result.accepted) {
      assert(cluster::composition_counts(pair.first) == first_counts);
      assert(cluster::composition_counts(pair.second) == second_counts);
    }
  }
}

int main() {
  test_balance_and_composition_conservation();
  test_cluster_delta_matches_full_energy();
  test_proposal_symmetry_and_signed_positive_control();
  test_restricted_binary_domain();
  test_metropolis_step_preserves_counts_when_accepted();
  std::cout << "balanced cluster tests passed\n";
  return 0;
}
