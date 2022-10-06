import os
import unittest

import numpy as np
import scipy.sparse as sp

from orangecontrib.text.corpus import Corpus
from orangecontrib.text.vectorization.base import (
    BaseVectorizer,
    VectorizationComputeValue,
)


class BaseVectorizationTest(unittest.TestCase):
    def test_variable_attributes(self):
        c1 = Corpus.from_file('deerwester')
        c2 = Corpus.from_file('deerwester')
        X = np.array([list(range(4)) for _ in range(len(c1))])
        X = sp.csr_matrix(X)

        dictionary = {
            0: 'd',
            1: 'c',
            2: 'b',
            3: 'a',
        }

        c1 = BaseVectorizer.add_features(c1, X, dictionary,
                                    compute_values=None, var_attrs=None)
        c2 = BaseVectorizer.add_features(c2, X, dictionary,
                                    compute_values=None, var_attrs={'foo': 1})

        n_attrs_before = len(c1.domain.attributes[0].attributes)
        n_attrs_after = len(c2.domain.attributes[0].attributes)
        self.assertTrue(n_attrs_after - n_attrs_before, 1)

        for a in c2.domain.attributes:
            self.assertIn('foo', a.attributes)


class TestVectorizationComputeValue(unittest.TestCase):
    def test_unpickling_old_pickle(self):
        """
        Before orange3-text version 1.12.0 variable was wrongly set to current
        variable (variable that has this compute value attached) instead of
        original variable which caused fails after latest changes in core
        Orange. Since variable from VectorizationComputeValue is never used in
        practice we do not set it anymore (it is always None for
        VectorizationComputeValue).
        Anyway it is still set in pickles create before 1.12.0. With this test
        we test that old pickle with variables that have VectorizationComputeValue
        are un-pickled correctly.
        """
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "old-bow-pickle.pkl"
        )
        data = Corpus.from_file(path)
        self.assertEqual(len(data), 3)
        self.assertIsInstance(data.domain["!"].compute_value, VectorizationComputeValue)
        self.assertIsInstance(
            data.domain["aboard"].compute_value, VectorizationComputeValue
        )
        self.assertIsNone(data.domain["!"].compute_value.variable)
        self.assertIsNone(data.domain["aboard"].compute_value.variable)


if __name__ == "__main__":
    unittest.main()
