import unittest

from analysis_utils import energy, generate_nn_list, get_neighbors, get_nn, local_e


class AnalysisUtilsTest(unittest.TestCase):
    def test_neighbor_tables_are_periodic_and_reciprocal(self):
        L = 4
        nn = generate_nn_list(L)
        self.assertEqual(len(nn), L**3)

        for i in range(L):
            for j in range(L):
                for k in range(L):
                    idx = k + L * (j + i * L)
                    neighbors = nn[idx]
                    self.assertEqual(neighbors, get_nn(i, j, k, L))
                    self.assertEqual(len(neighbors), 6)
                    self.assertEqual(len(set(neighbors)), 6)
                    for neighbor in neighbors:
                        self.assertIn(idx, nn[neighbor])

        tuple_neighbors = get_neighbors(0, 0, 0, L)
        self.assertIn((L - 1, 0, 0), tuple_neighbors)
        self.assertIn((0, L - 1, 0), tuple_neighbors)
        self.assertIn((0, 0, L - 1), tuple_neighbors)

    def test_energy_matches_known_configurations(self):
        L = 4
        nn = generate_nn_list(L)
        lattice = [0] * (L**3)
        self.assertEqual(energy(lattice, nn, L), 0)

        lattice[0] = 1
        self.assertEqual(local_e(lattice, 0, nn), 9.0)
        self.assertEqual(energy(lattice, nn, L), 9)

        lattice[0] = 2
        self.assertEqual(local_e(lattice, 0, nn), 25.0)
        self.assertEqual(energy(lattice, nn, L), 25)

        lattice = [1] * (L**3)
        self.assertEqual(energy(lattice, nn, L), 9 * L**3)

        lattice = [2] * (L**3)
        self.assertEqual(energy(lattice, nn, L), L**3)

    def test_bad_particle_type_raises(self):
        L = 4
        nn = generate_nn_list(L)
        lattice = [0] * (L**3)
        lattice[0] = 3
        with self.assertRaises(ValueError):
            local_e(lattice, 0, nn)


if __name__ == "__main__":
    unittest.main()
