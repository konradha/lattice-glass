#include "../balanced_cluster.h"
#include "../fp_sampler.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace cluster = lattice_glass::cluster;
namespace fp = lattice_glass::fp;

// The reference coupling sums agreement-with-s0 over BOTH replicas. Exchanging
// the two replicas on any cluster C leaves that sum invariant, so the field
// cancels in the cluster acceptance ratio. This is the structural claim that
// keeps detailed balance intact; verify it directly.
static void test_reference_term_cancels_under_exchange() {
  const int L = 6;
  const int lat_size = L * L * L;
  const std::vector<int> nn = fp::cubic_neighbors(L);

  std::mt19937 gen(12345);
  std::vector<uint8_t> reference = fp::random_lattice(60, 90, lat_size, gen);

  cluster::ReplicaPair pair;
  pair.first = fp::random_lattice(60, 90, lat_size, gen);
  pair.second = fp::random_lattice(60, 90, lat_size, gen);

  // Bias both replicas toward the reference so a disagreement set exists but is
  // concentrated, then test many grown clusters plus random clusters.
  fp::equilibrate(pair.first, 3.0L, 0.8L, &reference, 1500, gen, nn.data());
  fp::equilibrate(pair.second, 3.0L, 0.8L, &reference, 1500, gen, nn.data());

  cluster::GrowthRule rule;
  rule.kappa = 0.5L;

  bool saw_per_replica_change = false;
  std::uniform_int_distribution<int> site_dist(0, lat_size - 1);

  for (int trial = 0; trial < 400; ++trial) {
    std::vector<int> sites;
    if (trial % 2 == 0) {
      const cluster::GrowthAttempt attempt = cluster::grow_balanced_cluster(
          pair, nn.data(), fp::kNumNeighbors, 32, rule, gen);
      if (!attempt.proposed)
        continue;
      sites = attempt.cluster_sites;
    } else {
      const int count = 1 + (trial % 5);
      for (int i = 0; i < count; ++i)
        sites.push_back(site_dist(gen));
    }

    const int agree_first_before = fp::agreement_count(pair.first, reference);
    const int agree_second_before = fp::agreement_count(pair.second, reference);
    const int sum_before = agree_first_before + agree_second_before;

    cluster::exchange_cluster(pair, sites);

    const int agree_first_after = fp::agreement_count(pair.first, reference);
    const int agree_second_after = fp::agreement_count(pair.second, reference);
    const int sum_after = agree_first_after + agree_second_after;

    // Summed agreement is invariant under the exchange (the field cancels).
    assert(sum_before == sum_after);
    if (agree_first_before != agree_first_after)
      saw_per_replica_change = true;

    cluster::exchange_cluster(pair, sites); // restore
  }

  // The test would be vacuous if no per-replica agreement ever changed.
  assert(saw_per_replica_change);
}

// Sanity for the windowed integrated-autocorrelation-time estimator.
static void test_autocorrelation_estimator() {
  std::mt19937 gen(777);
  std::normal_distribution<double> noise(0.0, 1.0);

  // i.i.d. series -> tau_int near 0.5.
  std::vector<long double> iid(200000);
  for (auto &x : iid)
    x = noise(gen);
  const fp::AutocorrResult iid_result =
      fp::integrated_autocorrelation_time(iid);
  assert(iid_result.tau_int > 0.35L && iid_result.tau_int < 1.5L);
  assert(iid_result.resolved);

  // AR(1) with phi=0.8 -> tau_int = (1+phi)/(2(1-phi)) = 4.5.
  const long double phi = 0.8L;
  std::vector<long double> ar(200000);
  long double prev = 0.0L;
  for (auto &x : ar) {
    prev = phi * prev + noise(gen);
    x = prev;
  }
  const fp::AutocorrResult ar_result = fp::integrated_autocorrelation_time(ar);
  assert(ar_result.tau_int > 2.5L && ar_result.tau_int < 7.0L);
  assert(ar_result.resolved);
}

int main() {
  test_reference_term_cancels_under_exchange();
  test_autocorrelation_estimator();
  std::cout << "reference cancellation tests passed\n";
  return 0;
}
