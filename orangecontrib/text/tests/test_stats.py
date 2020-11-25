import unittest
import numpy as np
import scipy.sparse as sp

from orangecontrib.text.stats import hypergeom_p_values, is_sorted

class StatsTests(unittest.TestCase):
    x = np.array([[0, 0, 9, 0, 1],
                  [0, 1, 3, 0, 2],
                  [2, 5, 0, 0, 2],
                  [4, 1, 1, 2, 0]])

    def test_hypergeom_p_values(self):
        results = [0.16666666666666669, 0.49999999999999989, 1.0, 0.49999999999999989, 1.0]

        # calculating on counts
        pvals = hypergeom_p_values(self.x, self.x[-2:, :])
        np.testing.assert_almost_equal(pvals, results)

        # calculating on sparse counts
        pvals = hypergeom_p_values(sp.csr_matrix(self.x), self.x[-2:, :])
        np.testing.assert_almost_equal(pvals, results)

        # calculating on 0,1s
        clipped = self.x.clip(min=0, max=1)
        pvals = hypergeom_p_values(clipped, clipped[-2:, :])
        np.testing.assert_almost_equal(pvals, results)

        with self.assertRaises(ValueError):
            hypergeom_p_values(self.x, self.x[-2:, :-1])

    def test_is_sorted(self):
        self.assertTrue(is_sorted(range(10)))
        self.assertFalse(is_sorted(range(10)[::-1]))
