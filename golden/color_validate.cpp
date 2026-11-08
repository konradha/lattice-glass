// Exhaustive validation of the candidate L=21 local-move edge colour schedule.
//
// Local move (cpu/lattice.hpp, local_pct=100):
//   pick a random site a, pick one of its six nearest neighbours b, swap types.
//   The read/write dependency set is N[a] union N[b], which for adjacent a,b
//   is exactly 12 distinct sites on a cubic lattice.
//   Two moves commute iff their 12-site affected sets are disjoint, which is
//   equivalent to every cross-endpoint torus Manhattan distance being >= 3.
//
// Candidate schedule (LOCAL_DYNAMICS_PLAN.md):
//   +x edge anchored at (x,y,z): c = (x + 3y + 5z) mod 21
//   +y edge anchored at (x,y,z): c = (y + 3z + 5x) mod 21
//   +z edge anchored at (x,y,z): c = (z + 3x + 5y) mod 21
//
// The 27,783 undirected nearest-neighbour bonds are enumerated once (all +x,
// +y and +z edges from every site). Every pair inside every one of the 63
// (axis, colour) classes is checked for disjoint 12-site affected sets and for
// the >= 3 endpoint-distance condition. The exit status is nonzero on any
// structural failure or conflict.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int L = 21;
constexpr int N = L * L * L;              // 9,261
constexpr int kAxes = 3;
constexpr int kColors = 21;
constexpr int kClasses = kAxes * kColors; // 63
constexpr int kBonds = kAxes * N;         // 27,783
constexpr int kAffected = 12;
constexpr int kClassSize = 441;
constexpr int kMaxWitness = 8;
constexpr long long kPairsPerClass =
    static_cast<long long>(kClassSize) * (kClassSize - 1) / 2;
constexpr long long kPairsPerAxis = kColors * kPairsPerClass;
constexpr long long kPairsTotal = kClasses * kPairsPerClass;

struct Coord {
  int x;
  int y;
  int z;
};

struct Axis {
  int dx;
  int dy;
  int dz;
  const char* name;
};

constexpr Axis kAxis[kAxes] = {
    {1, 0, 0, "+x"},
    {0, 1, 0, "+y"},
    {0, 0, 1, "+z"},
};

inline int wrap(int v) {
  v %= L;
  return v < 0 ? v + L : v;
}

// Site numbering matches cpu/lattice.hpp: (x,y,z) -> x*L^2 + y*L + z.
inline int site_of(int x, int y, int z) {
  return (wrap(x) * L + wrap(y)) * L + wrap(z);
}

inline Coord coord_of(int s) {
  return {s / (L * L), (s / L) % L, s % L};
}

inline int axis_distance(int a, int b) {
  int d = a - b;
  if (d < 0) d = -d;
  return d < L - d ? d : L - d;
}

inline int torus_manhattan(int a, int b) {
  const Coord p = coord_of(a);
  const Coord q = coord_of(b);
  return axis_distance(p.x, q.x) + axis_distance(p.y, q.y) +
         axis_distance(p.z, q.z);
}

inline void neighbours(int s, int out[6]) {
  const Coord p = coord_of(s);
  out[0] = site_of(p.x + 1, p.y, p.z);
  out[1] = site_of(p.x - 1, p.y, p.z);
  out[2] = site_of(p.x, p.y + 1, p.z);
  out[3] = site_of(p.x, p.y - 1, p.z);
  out[4] = site_of(p.x, p.y, p.z + 1);
  out[5] = site_of(p.x, p.y, p.z - 1);
}

// Local-move oracle: the union of the closed neighbourhoods of both endpoints
// of an adjacent bond. The result is sorted and deduplicated; for a valid
// nearest-neighbour bond it contains exactly 12 distinct sites.
int build_affected(int a, int b, std::array<int, kAffected>& out) {
  int tmp[2 + 2 * 6];
  int n = 0;
  tmp[n++] = a;
  tmp[n++] = b;
  int nn[6];
  neighbours(a, nn);
  for (int i = 0; i < 6; ++i) tmp[n++] = nn[i];
  neighbours(b, nn);
  for (int i = 0; i < 6; ++i) tmp[n++] = nn[i];

  std::sort(tmp, tmp + n);
  n = static_cast<int>(std::unique(tmp, tmp + n) - tmp);

  out.fill(-1);
  for (int i = 0; i < n && i < kAffected; ++i) out[i] = tmp[i];
  return n;
}

// Cyclic rotations of the +x colour formula.
int edge_color(int axis, int x, int y, int z) {
  int c = 0;
  switch (axis) {
    case 0:
      c = x + 3 * y + 5 * z;
      break;
    case 1:
      c = y + 3 * z + 5 * x;
      break;
    default:
      c = z + 3 * x + 5 * y;
      break;
  }
  c %= L;
  return c < 0 ? c + L : c;
}

struct Edge {
  int a;
  int b;
  int axis;
  int class_id;
  std::array<int, kAffected> affected;
};

inline bool disjoint(const std::array<int, kAffected>& a,
                     const std::array<int, kAffected>& b) {
  int i = 0;
  int j = 0;
  while (i < kAffected && j < kAffected) {
    if (a[i] == b[j]) return false;
    if (a[i] < b[j]) {
      ++i;
    } else {
      ++j;
    }
  }
  return true;
}

inline int shared_count(const std::array<int, kAffected>& a,
                        const std::array<int, kAffected>& b) {
  int i = 0;
  int j = 0;
  int n = 0;
  while (i < kAffected && j < kAffected) {
    if (a[i] == b[j]) {
      ++n;
      ++i;
      ++j;
    } else if (a[i] < b[j]) {
      ++i;
    } else {
      ++j;
    }
  }
  return n;
}

inline int min_endpoint_distance(const Edge& e, const Edge& f) {
  int d = torus_manhattan(e.a, f.a);
  d = std::min(d, torus_manhattan(e.a, f.b));
  d = std::min(d, torus_manhattan(e.b, f.a));
  d = std::min(d, torus_manhattan(e.b, f.b));
  return d;
}

struct Witness {
  int i = 0;
  int j = 0;
  int shared = 0;
  int min_distance = 0;
};

struct ClassTally {
  long long pairs = 0;
  long long conflicts = 0;
  long long distance_violations = 0;
  long long oracle_mismatches = 0;
};

} // namespace

int main() {
  std::vector<Edge> edges;
  edges.reserve(kBonds);

  int bad_affected = 0;
  int bad_endpoints = 0;

  // Every undirected nearest-neighbour bond exactly once: all +x, +y and +z
  // edges anchored at each of the N sites.
  for (int x = 0; x < L; ++x) {
    for (int y = 0; y < L; ++y) {
      for (int z = 0; z < L; ++z) {
        for (int axis = 0; axis < kAxes; ++axis) {
          const Axis& ax = kAxis[axis];
          Edge e{};
          e.axis = axis;
          e.a = site_of(x, y, z);
          e.b = site_of(x + ax.dx, y + ax.dy, z + ax.dz);
          e.class_id = axis * kColors + edge_color(axis, x, y, z);

          if (build_affected(e.a, e.b, e.affected) != kAffected) {
            ++bad_affected;
          }
          if (e.a == e.b || torus_manhattan(e.a, e.b) != 1) {
            ++bad_endpoints;
          }
          edges.push_back(e);
        }
      }
    }
  }

  // Check that the enumeration did not duplicate an undirected bond.
  std::vector<uint64_t> bond_keys;
  bond_keys.reserve(edges.size());
  for (const Edge& e : edges) {
    const int lo = std::min(e.a, e.b);
    const int hi = std::max(e.a, e.b);
    bond_keys.push_back(static_cast<uint64_t>(lo) * N +
                        static_cast<uint64_t>(hi));
  }
  std::sort(bond_keys.begin(), bond_keys.end());
  int duplicate_bonds = 0;
  for (std::size_t i = 1; i < bond_keys.size(); ++i) {
    if (bond_keys[i] == bond_keys[i - 1]) ++duplicate_bonds;
  }

  std::array<int, kClasses> class_size{};
  for (const Edge& e : edges) ++class_size[e.class_id];

  int bad_class_sizes = 0;
  for (int c = 0; c < kClasses; ++c) {
    if (class_size[c] != kClassSize) ++bad_class_sizes;
  }

  // A single move's 12 affected sites must have distinct colour residues.
  int residue_violations = 0;
  if (bad_affected == 0) {
    for (const Edge& e : edges) {
      bool seen[kColors] = {};
      for (int i = 0; i < kAffected; ++i) {
        const Coord p = coord_of(e.affected[i]);
        const int c = edge_color(e.axis, p.x, p.y, p.z);
        if (seen[c]) {
          ++residue_violations;
          break;
        }
        seen[c] = true;
      }
    }
  }

  // Group edges by (axis, colour) into contiguous class ranges.
  std::array<int, kClasses> class_begin{};
  int running = 0;
  for (int c = 0; c < kClasses; ++c) {
    class_begin[c] = running;
    running += class_size[c];
  }
  std::vector<int> members(static_cast<std::size_t>(running));
  std::array<int, kClasses> cursor = class_begin;
  for (std::size_t i = 0; i < edges.size(); ++i) {
    const int c = edges[i].class_id;
    members[static_cast<std::size_t>(cursor[c]++)] = static_cast<int>(i);
  }

  std::array<ClassTally, kClasses> tally{};
  long long total_pairs = 0;
  long long total_conflicts = 0;
  long long total_distance_violations = 0;
  long long total_oracle_mismatches = 0;

  // Exhaustive pair scan inside each of the 63 classes.
  for (int c = 0; c < kClasses; ++c) {
    const int axis = c / kColors;
    const int color = c % kColors;
    const int size = class_size[c];
    const int begin = class_begin[c];

    std::array<Witness, kMaxWitness> witnesses{};
    int witness_count = 0;

    for (int ii = 0; ii < size; ++ii) {
      const int ei = members[begin + ii];
      const Edge& e = edges[static_cast<std::size_t>(ei)];
      for (int jj = ii + 1; jj < size; ++jj) {
        const int fi = members[begin + jj];
        const Edge& f = edges[static_cast<std::size_t>(fi)];

        const bool sets_disjoint = disjoint(e.affected, f.affected);
        const int min_distance = min_endpoint_distance(e, f);

        ++tally[c].pairs;
        if (!sets_disjoint) {
          ++tally[c].conflicts;
          if (witness_count < kMaxWitness) {
            Witness& w = witnesses[witness_count++];
            w.i = ei;
            w.j = fi;
            w.shared = shared_count(e.affected, f.affected);
            w.min_distance = min_distance;
          }
        }
        if (min_distance < 3) ++tally[c].distance_violations;
        if ((!sets_disjoint) != (min_distance < 3)) {
          ++tally[c].oracle_mismatches;
        }
      }
    }

    total_pairs += tally[c].pairs;
    total_conflicts += tally[c].conflicts;
    total_distance_violations += tally[c].distance_violations;
    total_oracle_mismatches += tally[c].oracle_mismatches;

    const bool class_ok = size == kClassSize &&
                          tally[c].pairs == kPairsPerClass &&
                          tally[c].conflicts == 0 &&
                          tally[c].distance_violations == 0 &&
                          tally[c].oracle_mismatches == 0;
    std::printf(
        "axis %s color %2d: size %3d pairs %6lld conflicts %lld "
        "distance<3 %lld %s\n",
        kAxis[axis].name, color, size, tally[c].pairs, tally[c].conflicts,
        tally[c].distance_violations, class_ok ? "PASS" : "FAIL");

    for (int w = 0; w < witness_count; ++w) {
      const Edge& e = edges[static_cast<std::size_t>(witnesses[w].i)];
      const Edge& f = edges[static_cast<std::size_t>(witnesses[w].j)];
      const Coord pe = coord_of(e.a);
      const Coord pf = coord_of(f.a);
      std::printf(
          "  witness %s anchors (%d,%d,%d) vs (%d,%d,%d): shared=%d "
          "min_endpoint_dist=%d\n",
          kAxis[axis].name, pe.x, pe.y, pe.z, pf.x, pf.y, pf.z,
          witnesses[w].shared, witnesses[w].min_distance);
    }
  }

  std::printf("\nSummary\n");
  std::printf("local-move oracle: L=%d N=%d, undirected NN bonds=%zu "
              "(expected %d)\n",
              L, N, edges.size(), kBonds);
  std::printf("12-site affected sets: %s (%d invalid)\n",
              bad_affected == 0 ? "valid" : "INVALID", bad_affected);
  std::printf("duplicate bonds=%d, non-neighbour endpoints=%d\n",
              duplicate_bonds, bad_endpoints);
  std::printf("class size expected=%d, bad classes=%d, "
              "affected-residue collisions=%d edge(s)\n",
              kClassSize, bad_class_sizes, residue_violations);

  for (int axis = 0; axis < kAxes; ++axis) {
    long long pairs = 0;
    long long conflicts = 0;
    long long distance_violations = 0;
    long long oracle_mismatches = 0;
    int bad_sizes = 0;
    int axis_edges = 0;
    for (int color = 0; color < kColors; ++color) {
      const int c = axis * kColors + color;
      pairs += tally[c].pairs;
      conflicts += tally[c].conflicts;
      distance_violations += tally[c].distance_violations;
      oracle_mismatches += tally[c].oracle_mismatches;
      axis_edges += class_size[c];
      if (class_size[c] != kClassSize) ++bad_sizes;
    }
    const bool axis_ok = axis_edges == N && bad_sizes == 0 &&
                         pairs == kPairsPerAxis && conflicts == 0 &&
                         distance_violations == 0 && oracle_mismatches == 0;
    std::printf("axis %s: classes %d edges %d pairs %lld conflicts %lld "
                "distance<3 %lld %s\n",
                kAxis[axis].name, kColors, axis_edges, pairs, conflicts,
                distance_violations, axis_ok ? "PASS" : "FAIL");
  }

  const bool ok = edges.size() == static_cast<std::size_t>(kBonds) &&
                  bad_affected == 0 && bad_endpoints == 0 &&
                  duplicate_bonds == 0 && bad_class_sizes == 0 &&
                  residue_violations == 0 && total_pairs == kPairsTotal &&
                  total_conflicts == 0 &&
                  total_distance_violations == 0 &&
                  total_oracle_mismatches == 0;

  std::printf("63-colour union (axis,colour): classes %d edges %zu "
              "pairs %lld (expected %lld) conflicts %lld distance<3 %lld "
              "%s\n",
              kClasses, edges.size(), total_pairs, kPairsTotal,
              total_conflicts, total_distance_violations,
              ok ? "PASS" : "FAIL");

  if (!ok) {
    std::printf("RESULT: FAIL\n");
    return EXIT_FAILURE;
  }
  std::printf("RESULT: PASS\n");
  return EXIT_SUCCESS;
}
