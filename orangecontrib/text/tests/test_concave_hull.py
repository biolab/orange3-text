import unittest

import numpy as np
from Orange.data import Table

from orangecontrib.text.concave_hull import compute_concave_hulls


class TestConcaveHull(unittest.TestCase):
    def test_compute_concave_hulls(self):
        data = Table.from_file("iris")[:, 2:4]
        clusters = np.array([0] * 50 + [1] * 50 + [2] * 50)

        hulls = compute_concave_hulls(data.X, clusters, epsilon=0.5)
        self.assertEqual(3, len(hulls))
        self.assertEqual(2, hulls[0].shape[1])  # hull have x and y
        self.assertEqual(2, hulls[1].shape[1])  # hull have x and y
        self.assertEqual(2, hulls[2].shape[1])  # hull have x and y

        hulls = compute_concave_hulls(data.X, clusters)
        self.assertEqual(3, len(hulls))
        self.assertEqual(2, hulls[0].shape[1])  # hull have x and y
        self.assertEqual(2, hulls[1].shape[1])  # hull have x and y
        self.assertEqual(2, hulls[2].shape[1])  # hull have x and y

    def test_compute_concave_hulls_subsampling(self):
        """
        When more than 1000 points passed they are sub-sampled in order to
        compute a concave hull
        """
        iris = Table.from_file("iris")
        data = np.repeat(iris.X[:, 2:4], 10, axis=0)  # more than 1000 points
        clusters = np.array([0] * 50 * 10 + [1] * 50 * 10 + [2] * 50 * 10)

        hulls = compute_concave_hulls(data, clusters, epsilon=0.5)

        self.assertEqual(3, len(hulls))
        self.assertEqual(2, hulls[0].shape[1])  # hull have x and y
        self.assertEqual(2, hulls[1].shape[1])  # hull have x and y
        self.assertEqual(2, hulls[2].shape[1])  # hull have x and y

    def test_compute_concave_hulls_3_or_less_points(self):
        """
        Concave hull must also work for tree points - it is a special case
        """
        data = np.array([[1, 1], [1, 2], [2, 1]])
        clusters = np.array([0] * 3)
        hulls = compute_concave_hulls(data, clusters, epsilon=0.5)

        self.assertEqual(1, len(hulls))
        self.assertEqual(2, hulls[0].shape[1])  # hull have x and y

        hulls = compute_concave_hulls(data[:2], clusters[:2], epsilon=0.5)
        self.assertEqual(1, len(hulls))
        self.assertEqual(2, hulls[0].shape[1])  # hull have x and y

        hulls = compute_concave_hulls(data[:1], clusters[:1], epsilon=0.5)
        self.assertEqual(1, len(hulls))
        self.assertEqual(2, hulls[0].shape[1])  # hull have x and y

    def test_all_points_same_value(self):
        # same value along y axis
        data = np.array([[1, 1], [1, 1], [2, 1]])
        clusters = np.array([0] * 3)
        hulls = compute_concave_hulls(data, clusters, epsilon=0.5)
        self.assertEqual(1, len(hulls))
        self.assertEqual(2, hulls[0].shape[1])  # hull have x and y

        # same value along x axis
        data = np.array([[1, 2], [1, 1], [1, 1]])
        clusters = np.array([0] * 3)
        hulls = compute_concave_hulls(data, clusters, epsilon=0.5)
        self.assertEqual(1, len(hulls))
        self.assertEqual(2, hulls[0].shape[1])  # hull have x and y

    def test_non_float_data(self):
        data = np.array([[1, 1], [1, 1], [2, 1]], dtype="object")
        clusters = np.array([0] * 3)
        hulls = compute_concave_hulls(data, clusters, epsilon=0.5)
        self.assertEqual(1, len(hulls))
        self.assertEqual(2, hulls[0].shape[1])  # hull have x and y


if __name__ == "__main__":
    unittest.main()
