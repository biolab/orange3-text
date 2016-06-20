import unittest

from orangecontrib.text.bagofowords import CountVectorizer, TfidfVectorizer, SimhashVectorizer
from orangecontrib.text.corpus import Corpus


class VectorizerTest:
    Vectorizer = NotImplemented

    def test_report(self):
        vect = self.Vectorizer()
        self.assertGreater(len(vect.report()), 1)


class SklearnVectorizerTests(VectorizerTest):
    def test_fit_transform(self):
        vect = self.Vectorizer()
        corpus = Corpus.from_file('deerwester')

        result = vect.fit_transform(corpus)
        self.assertIsInstance(result, Corpus)
        self.assertEqual(len(result.domain), 43)

    def test_transform(self):
        vect = self.Vectorizer()
        corpus = Corpus.from_file('deerwester')

        vect.fit(corpus)
        result = vect.transform(corpus)
        self.assertIsInstance(result, Corpus)
        self.assertEqual(len(result.domain), 43)

    def test_domain(self):
        vect = self.Vectorizer()
        corpus = Corpus.from_file('deerwester')

        vect.fit(corpus)
        result = vect.transform(corpus)
        attrs = [attr.name for attr in result.domain.attributes]
        self.assertEqual(attrs, sorted(attrs))

        X = result.X.toarray()
        for i in range(len(corpus)):
            for contains, attr in zip(X[i], attrs):
                if contains:
                    self.assertIn(attr, corpus.tokens[i])

    def test_ngrams(self):
        vect = self.Vectorizer(ngram_range=(1, 3))
        corpus = Corpus.from_file('deerwester')
        result = vect.fit_transform(corpus)
        attrs = [attr.name for attr in result.domain.attributes]
        self.assertIn(corpus.tokens[0][1], attrs)
        self.assertIn(' '.join(corpus.tokens[0][:2]), attrs)
        self.assertIn(' '.join(corpus.tokens[0][:3]), attrs)


class CountVectorizerTests(SklearnVectorizerTests, unittest.TestCase):
    Vectorizer = CountVectorizer


class TfidfVectorizerTests(SklearnVectorizerTests, unittest.TestCase):
    Vectorizer = TfidfVectorizer


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
