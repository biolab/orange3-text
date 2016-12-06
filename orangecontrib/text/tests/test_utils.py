import unittest

import numpy as np
import scipy.sparse as sp

from orangecontrib.text.util import chunks, np_sp_sum


class ChunksTest(unittest.TestCase):

    def test_results(self):
        self.assertEqual(list(chunks([], 10)), [])
        self.assertEqual(list(chunks([1, 2], 3)), [[1, 2]])
        self.assertEqual(list(chunks([1, 2], 1)), [[1], [2]])

    def test_size(self):
        for chunk in chunks(range(10), 2):
            self.assertEqual(len(chunk), 2)

        for chunk in chunks(range(10), 3):
            pass
        self.assertEqual(len(chunk), 1)


class TestNpSpSum(unittest.TestCase):
    def test_np_sp_sum(self):
        for data in [np.eye(10), sp.csr_matrix(np.eye(10))]:
            self.assertEqual(np_sp_sum(data), 10)
            np.testing.assert_equal(np_sp_sum(data, axis=1), np.ones(10))
            np.testing.assert_equal(np_sp_sum(data, axis=0), np.ones(10))
