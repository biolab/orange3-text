import unittest
from unittest import TestCase

import numpy as np
from shapely import LineString, Point, Polygon

from orangecontrib.text.hull import compute_concave_polygon
from orangecontrib.text.hull.cluster_hull import polygon_to_array


class TestConcaveHullKnn(TestCase):
    def test_low_number_points(self):
        # one point resulting in Point
        data = np.array([[1, 1]])
        res = compute_concave_polygon(data)
        self.assertIsInstance(res, Point)
        np.testing.assert_array_equal([res.x, res.y], data[0])

        # two (and collinear) resulting in LineString
        data = np.array([[1, 1], [2, 2]])
        res = compute_concave_polygon(data)
        self.assertIsInstance(res, LineString)
        np.testing.assert_array_equal(res.coords.xy, [[1, 2], [1, 2]])

        # collinear points resulting in LineString
        data = np.array([[1, 1], [2, 2], [3, 3]])
        res = compute_concave_polygon(data)
        self.assertIsInstance(res, LineString)
        np.testing.assert_array_equal(res.coords.xy, [[1, 3], [1, 3]])

        # collinear points resulting in LineString
        data = np.array([[1, 1], [2, 2], [3, 3], [1, 1]])
        res = compute_concave_polygon(data)
        self.assertIsInstance(res, LineString)
        np.testing.assert_array_equal(res.coords.xy, [[1, 3], [1, 3]])

        # triangle with point in the middle
        data = np.array([[1, 1], [1, 2], [2, 2], [1.2, 1.7]])
        res = compute_concave_polygon(data)
        self.assertIsInstance(res, Polygon)
        np.testing.assert_array_equal(
            polygon_to_array(res), [[1, 2], [2, 2], [1.2, 1.7], [1, 1], [1, 2]]
        )

        data = np.array([[1, 1], [1, 2], [2, 2], [8, 8]])
        res = compute_concave_polygon(data)
        self.assertIsInstance(res, Polygon)
        np.testing.assert_array_equal(
            polygon_to_array(res), [[2, 2], [1, 1], [1, 2], [8, 8], [2, 2]]
        )

    def test_collinear(self):
        data = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        res = compute_concave_polygon(data)
        self.assertIsInstance(res, LineString)
        np.testing.assert_array_equal(res.coords.xy, [[1, 4], [1, 4]])

        data = np.array([[1, 1], [2, 2], [-1, -1], [3, 3], [4, 4], [5, 5], [7, 7]])
        res = compute_concave_polygon(data)
        self.assertIsInstance(res, LineString)
        np.testing.assert_array_equal(res.coords.xy, [[-1, 7], [-1, 7]])

    def test_convex_shape(self):
        ls = np.linspace(0, 7 / 4 * np.pi, 8).reshape(-1, 1)
        d = np.vstack(([[1, 1]], np.hstack((1 + np.cos(ls), 1 + np.sin(ls)))))
        res = polygon_to_array(compute_concave_polygon(d))

        # result starts in point with angle 3/4*pi and goes in reverse direction
        ls = np.linspace(-5 / 4 * np.pi, 3 / 4 * np.pi, 9)[::-1].reshape(-1, 1)
        expected = np.hstack((1 + np.cos(ls), 1 + np.sin(ls)))
        np.testing.assert_almost_equal(res, expected)

    def test_intersected_points(self):
        # fmt: off
        data = np.array([
            [1, 1], [10, 3], [11, 8], [9, 14], [15, 21], [-5, 15], [-3, 10],
            [2, 5], [9, 10], [8, 9], [8, 11], [8, 12], [9, 11], [9, 12]
        ])
        # fmt: on
        res = polygon_to_array(compute_concave_polygon(data))
        exp = [[9, 14], [15, 21], [11, 8], [10, 3], [1, 1], [-3, 10], [-5, 15], [9, 14]]
        np.testing.assert_array_equal(res, exp)

    def test_simple_solution(self):
        # fmt: off
        data = np.array([
            [10, 9], [9, 18], [16, 13], [11, 15], [12, 14], [18, 12], [2, 14],
            [6, 18], [9, 9], [10, 8], [6, 17], [5, 3], [13, 19], [3, 18],
            [8, 17], [9, 7], [3, 0], [13, 18], [15, 4], [13, 16]
        ])
        exp = np.array([
            [9, 18], [13, 19], [18, 12], [15, 4], [3, 0], [5, 3], [2, 14],
            [3, 18], [6, 18], [9, 18]
        ])
        # fmt: on
        res = polygon_to_array(compute_concave_polygon(data))
        np.testing.assert_array_equal(res, exp)


if __name__ == "__main__":
    unittest.main()
