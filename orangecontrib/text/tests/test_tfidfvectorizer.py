import unittest

import numpy as np

from orangecontrib.text.corpus import Corpus
from orangecontrib.text.vectorization import TfidfVectorizer, CountVectorizer


class TfidfVectorizationTest(unittest.TestCase):
    def test_transform(self):
        vect = TfidfVectorizer()
        corpus = Corpus.from_file('deerwester')

        result = vect.transform(corpus)
        self.assertIsInstance(result, Corpus)
        self.assertEqual(len(result.domain), 43)

    def test_domain(self):
        vect = TfidfVectorizer()
        corpus = Corpus.from_file('deerwester')

        result = vect.transform(corpus)
        attrs = [attr.name for attr in result.domain.attributes]
        self.assertEqual(attrs, sorted(attrs))

        X = result.X.toarray()
        for i in range(len(corpus)):
            for contains, attr in zip(X[i], attrs):
                if contains > .001:
                    self.assertIn(attr, corpus.tokens[i])

    def test_ngrams(self):
        vect = TfidfVectorizer()
        corpus = Corpus.from_file('deerwester')
        corpus.ngram_range = (1, 3)
        result = vect.transform(corpus)
        attrs = [attr.name for attr in result.domain.attributes]
        self.assertIn(corpus.tokens[0][1], attrs)
        self.assertIn(' '.join(corpus.tokens[0][:2]), attrs)
        self.assertIn(' '.join(corpus.tokens[0][:3]), attrs)

    def test_report(self):
        vect = TfidfVectorizer()
        self.assertGreater(len(vect.report()), 0)

    def test_args(self):
        corpus = Corpus.from_file('deerwester')

        TfidfVectorizer.wglobals['const'] = lambda idf, D: 1

        vect = TfidfVectorizer(norm=TfidfVectorizer.NONE,
                               wlocal=TfidfVectorizer.COUNT,
                               wglobal='const')

        self.assertEqualCorpus(vect.transform(corpus),
                               CountVectorizer(binary=False).transform(corpus))

        vect = TfidfVectorizer(norm=TfidfVectorizer.NONE,
                               wlocal=TfidfVectorizer.BINARY,
                               wglobal='const')
        self.assertEqualCorpus(vect.transform(corpus),
                               CountVectorizer(binary=True).transform(corpus))

        vect = TfidfVectorizer(norm=TfidfVectorizer.L1,
                               wlocal=TfidfVectorizer.COUNT,
                               wglobal='const')
        x = vect.transform(corpus).X
        self.assertAlmostEqual(abs(x.sum(axis=1) - 1).sum(), 0)

    def assertEqualCorpus(self, first, second, msg=None):
        np.testing.assert_allclose(first.X.todense(), second.X.todense(), err_msg=msg)
