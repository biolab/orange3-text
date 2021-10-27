import unittest

import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_array_equal
from scipy.sparse import csc_matrix

from orangecontrib.text.util import chunks, np_sp_sum, Sparse2CorpusSliceable


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


class TestSparse2CorpusSliceable(unittest.TestCase):
    def setUp(self) -> None:
        self.orig_array = np.array([[1, 2, 3], [4, 5, 6]])
        self.s2c = Sparse2CorpusSliceable(csc_matrix(self.orig_array))

    def test_slice(self):
        assert_array_equal(self.s2c[:2].sparse.toarray(), self.orig_array[:, :2])
        assert_array_equal(self.s2c[1:3].sparse.toarray(), self.orig_array[:, 1:3])

    def test_index(self):
        assert_array_equal(self.s2c[1].sparse.toarray(), self.orig_array[:, [1]])

    def test_list_of_indices(self):
        assert_array_equal(
            self.s2c[[1, 2]].sparse.toarray(), self.orig_array[:, [1, 2]]
        )
        assert_array_equal(self.s2c[[1]].sparse.toarray(), self.orig_array[:, [1]])

    def test_ndarray(self):
        assert_array_equal(
            self.s2c[np.array([1, 2])].sparse.toarray(), self.orig_array[:, [1, 2]]
        )
        assert_array_equal(
            self.s2c[np.array([1])].sparse.toarray(), self.orig_array[:, [1]]
        )

    def test_range(self):
        assert_array_equal(
            self.s2c[range(1, 3)].sparse.toarray(), self.orig_array[:, [1, 2]]
        )
        assert_array_equal(
            self.s2c[range(1, 2)].sparse.toarray(), self.orig_array[:, [1]]
        )

    def test_elipsis(self):
        assert_array_equal(self.s2c[...].sparse.toarray(), self.orig_array)


if __name__ == "__main__":
    unittest.main()
