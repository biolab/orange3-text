import unittest
import numpy as np
import scipy.sparse as sp

from orangecontrib.text.stats import hypergeom_p_values, false_discovery_rate, is_sorted

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


    def test_false_discovery_rate(self):
        p_values = np.array(
            [0.727, 0.281, 0.791, 0.034, 0.628, 0.743, 0.958, 0.552, 0.867, 0.606,
             0.611, 0.594, 0.071, 0.517, 0.526, 0.526, 0.635, 0.932, 0.210, 0.636])
        # calculated with http://www.sdmproject.com/utilities/?show=FDR
        fdr_fixed = np.array(
            [0.92875, 0.9085714, 0.9305882, 0.68, 0.9085714, 0.92875, 0.958, 0.9085714,
             0.958, 0.9085714, 0.9085714, 0.9085714, 0.71, 0.9085714, 0.9085714, 0.9085714,
             0.9085714, 0.958, 0.9085714, 0.9085714]
        )
        corrected = false_discovery_rate(p_values)
        np.testing.assert_allclose(corrected, fdr_fixed)

        corrected = false_discovery_rate(p_values, m=len(p_values))
        np.testing.assert_allclose(corrected, fdr_fixed)

        corrected = false_discovery_rate(sorted(p_values), ordered=True)
        np.testing.assert_allclose(sorted(corrected), sorted(fdr_fixed))

        np.testing.assert_equal(false_discovery_rate([]), [])
        np.testing.assert_equal(false_discovery_rate(p_values, m=-1), [])

        dependant = [3.3414007065721947, 3.2688034599191167, 3.3480141985890031, 2.446462966857704,
                     3.2688034599191167, 3.3414007065721947, 3.4466345915436469, 3.2688034599191167,
                     3.4466345915436469, 3.2688034599191167, 3.2688034599191167, 3.2688034599191167,
                     2.554395156572014, 3.2688034599191167, 3.2688034599191167, 3.2688034599191167,
                     3.2688034599191167, 3.4466345915436469, 3.2688034599191167, 3.2688034599191167]
        np.testing.assert_equal(false_discovery_rate(p_values, dependent=True), dependant)

    def test_is_sorted(self):
        self.assertTrue(is_sorted(range(10)))
        self.assertFalse(is_sorted(range(10)[::-1]))
