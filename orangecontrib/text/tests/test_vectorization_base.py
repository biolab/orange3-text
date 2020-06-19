import unittest

import numpy as np
import scipy.sparse as sp

from orangecontrib.text.corpus import Corpus
from orangecontrib.text.vectorization.base import BaseVectorizer


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
