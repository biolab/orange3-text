import unittest

import numpy as np
from numpy.testing import assert_array_equal
from Orange.data import Table
from shapely import Polygon

from orangecontrib.text.hull import compute_hulls
from orangecontrib.text.hull.cluster_hull import (
    _denormalize,
    _normalize,
    _xy_range,
    polygon_to_array,
)


class TestClusterHull(unittest.TestCase):
    def test_polygon_to_array(self):
        res = polygon_to_array(Polygon([[1, 1], [1, 2], [2, 1]]))
        assert_array_equal(res, [[1, 1], [1, 2], [2, 1], [1, 1]])

    def test_xy_range(self):
        res = _xy_range(np.array([[1, 2], [2, 2], [3, 3], [4, 2]]))
        self.assertTupleEqual((1, 4, 2, 3), res)

        # examples collinear along x axis
        res = _xy_range(np.array([[1, 2], [2, 2], [3, 2]]))
        self.assertTupleEqual((1, 3, 1.5, 2.5), res)

        # examples collinear along y axis
        res = _xy_range(np.array([[2, 1], [2, 2], [2, 3]]))
        self.assertTupleEqual((1.5, 2.5, 1, 3), res)

        # one point
        res = _xy_range(np.array([[2, 1]]))
        self.assertTupleEqual((1.5, 2.5, 0.5, 1.5), res)

    def test_normalize(self):
        example = np.array([[1, 2], [2, 2], [3, 3], [4, 2]])
        res, x_min, x_max, y_min, y_max = _normalize(example)
        assert_array_equal(res, [[0, 0], [1 / 3, 0], [2 / 3, 1], [1, 0]])
        self.assertEqual(1, x_min)
        self.assertEqual(4, x_max)
        self.assertEqual(2, y_min)
        self.assertEqual(3, y_max)

    def test_denormalize(self):
        example = np.array([[1, 2], [2, 2], [3, 3], [4, 2]])
        res, x_min, x_max, y_min, y_max = _normalize(example)
        res = _denormalize(res, x_min, x_max, y_min, y_max)
        assert_array_equal(res, example)

    def test_compute_concave_hulls(self):
        data = Table.from_file("iris")[:, 2:4]
        clusters = np.array([0] * 50 + [1] * 50 + [2] * 50)

        hulls = compute_hulls(data.X, clusters)
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

        hulls = compute_hulls(data, clusters)
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
        hulls = compute_hulls(data, clusters)

        self.assertEqual(1, len(hulls))
        self.assertEqual(2, hulls[0].shape[1])  # hull have x and y

        hulls = compute_hulls(data[:2], clusters[:2])
        self.assertEqual(1, len(hulls))
        self.assertEqual(2, hulls[0].shape[1])  # hull have x and y

        hulls = compute_hulls(data[:1], clusters[:1])
        self.assertEqual(1, len(hulls))
        self.assertEqual(2, hulls[0].shape[1])  # hull have x and y

    def test_points_same_value(self):
        # same value along y axis
        data = np.array([[1, 1], [1, 1], [2, 1]])
        clusters = np.array([0] * 3)
        hulls = compute_hulls(data, clusters)
        self.assertEqual(1, len(hulls))
        self.assertEqual(2, hulls[0].shape[1])  # hull have x and y

        # same value along x axis
        data = np.array([[1, 2], [1, 1], [1, 1]])
        clusters = np.array([0] * 3)
        hulls = compute_hulls(data, clusters)
        self.assertEqual(1, len(hulls))
        self.assertEqual(2, hulls[0].shape[1])  # hull have x and y

    def test_non_float_data(self):
        data = np.array([[1, 1], [1, 1], [2, 1]], dtype="object")
        clusters = np.array([0] * 3)
        hulls = compute_hulls(data, clusters)
        self.assertEqual(1, len(hulls))
        self.assertEqual(2, hulls[0].shape[1])  # hull have x and y


if __name__ == "__main__":
    unittest.main()
