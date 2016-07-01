import unittest

import numpy as np

from orangecontrib.text.bagofowords import CountVectorizer, TfidfVectorizer, SimhashVectorizer
from orangecontrib.text.corpus import Corpus


class VectorizerTest:
    Vectorizer = NotImplemented

    def test_report(self):
        vect = self.Vectorizer()
        self.assertGreater(len(vect.report()), 0)


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
                               wlocal=TfidfVectorizer.IDENTITY,
                               wglobal='const')

        self.assertEqualCorpus(vect.transform(corpus),
                               CountVectorizer(binary=False).transform(corpus))

        vect = TfidfVectorizer(norm=TfidfVectorizer.NONE,
                               wlocal=TfidfVectorizer.BINARY,
                               wglobal='const')
        self.assertEqualCorpus(vect.transform(corpus),
                               CountVectorizer(binary=True).transform(corpus))

        vect = TfidfVectorizer(norm=TfidfVectorizer.L1,
                               wlocal=TfidfVectorizer.IDENTITY,
                               wglobal='const')
        x = vect.transform(corpus).X
        self.assertAlmostEqual(abs(x.sum(axis=1)-1).sum(), 0)

    def assertEqualCorpus(self, first, second, msg=None):
        np.testing.assert_allclose(first.X.todense(), second.X.todense(), err_msg=msg)


class CountVectorizerTests(unittest.TestCase):
    def test_ngrams(self):
        vect = CountVectorizer()
        corpus = Corpus.from_file('deerwester')
        corpus.ngram_range = (1, 3)
        result = vect.transform(corpus)
        attrs = [attr.name for attr in result.domain.attributes]
        self.assertIn(corpus.tokens[0][1], attrs)
        self.assertIn(' '.join(corpus.tokens[0][:2]), attrs)
        self.assertIn(' '.join(corpus.tokens[0][:3]), attrs)

    def test_transform(self):
        vect = CountVectorizer()
        corpus = Corpus.from_file('deerwester')
        result = vect.transform(corpus)
        self.assertIsInstance(result, Corpus)
        self.assertEqual(len(result.domain), 43)

    def test_binary(self):
        vect = CountVectorizer(binary=True)
        corpus = Corpus.from_file('deerwester')
        result = vect.transform(corpus)
        self.assertEqual(result.X.max(), 1.)

    def test_report(self):
        vect = CountVectorizer()
        self.assertGreater(len(vect.report()), 0)


class TestSimhash(VectorizerTest, unittest.TestCase):
    Vectorizer = SimhashVectorizer

    def setUp(self):
        self.corpus = Corpus.from_file('deerwester')

    def test_transform(self):
        vect = SimhashVectorizer(shingle_len=10, f=64)
        result = vect.transform(self.corpus)
        self.assertIsInstance(result, Corpus)
        self.assertEqual(len(result), len(self.corpus))
        self.assertEqual(result.X.shape, (len(self.corpus), 64))
