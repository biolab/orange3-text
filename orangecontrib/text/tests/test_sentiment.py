import unittest

from numpy.testing import assert_allclose

from orangecontrib.text.corpus import Corpus
from orangecontrib.text.sentiment import LiuHuSentiment, VaderSentiment, \
    MultiSentiment, SentiArt, LilahSentiment


class LiuHuTest(unittest.TestCase):
    def setUp(self):
        self.corpus = Corpus.from_file('deerwester')
        self.method = LiuHuSentiment('English')
        self.new_cols = 1

    def test_transform(self):
        sentiment = self.method.transform(self.corpus)
        self.assertIsInstance(sentiment, Corpus)
        self.assertEqual(len(sentiment.domain.variables),
                         len(self.corpus.domain.variables) + self.new_cols)

    def test_copy(self):
        sentiment_t = self.method.transform(self.corpus)
        self.assertIsNot(self.corpus, sentiment_t)

    def test_compute_values(self):
        sentiment = self.method.transform(self.corpus)
        computed = Corpus.from_table(sentiment.domain, self.corpus)

        self.assertEqual(sentiment.domain, computed.domain)
        self.assertTrue((sentiment.X == computed.X).all())

    def test_compute_values_to_different_domain(self):
        destination = Corpus.from_file('andersen')

        self.assertFalse(self.corpus.domain.attributes)
        self.assertFalse(destination.domain.attributes)

        sentiment = self.method.transform(self.corpus)
        computed = destination.transform(sentiment.domain)

        self.assertTrue(sentiment.domain.attributes)
        self.assertEqual(sentiment.domain.attributes, computed.domain.attributes)

    def test_empty_corpus(self):
        corpus = Corpus.from_file('deerwester')[:0]
        sentiment = self.method.transform(corpus)
        self.assertEqual(len(sentiment.domain.variables),
                         len(self.corpus.domain.variables) + self.new_cols)
        self.assertEqual(len(sentiment), 0)


class LiuHuSlovenian(unittest.TestCase):
    def setUp(self):
        self.corpus = Corpus.from_file('slo-opinion-corpus')
        self.method = LiuHuSentiment('Slovenian')
        self.new_cols = 1

    def test_transform(self):
        sentiment = self.method.transform(self.corpus)
        self.assertIsInstance(sentiment, Corpus)
        self.assertEqual(len(sentiment.domain.variables),
                         len(self.corpus.domain.variables) + self.new_cols)

    def test_copy(self):
        sentiment_t = self.method.transform(self.corpus)
        self.assertIsNot(self.corpus, sentiment_t)

    def test_compute_values(self):
        sentiment = self.method.transform(self.corpus)
        computed = Corpus.from_table(sentiment.domain, self.corpus)

        self.assertEqual(sentiment.domain, computed.domain)
        self.assertTrue((sentiment.X == computed.X).all())

    def test_empty_corpus(self):
        corpus = self.corpus[:0]
        sentiment = self.method.transform(corpus)
        self.assertEqual(len(sentiment.domain.variables),
                         len(self.corpus.domain.variables) + self.new_cols)
        self.assertEqual(len(sentiment), 0)


class VaderTest(LiuHuTest):
    def setUp(self):
        self.corpus = Corpus.from_file('deerwester')
        self.method = VaderSentiment()
        self.new_cols = 4


class MultiSentimentTest(LiuHuTest):
    def setUp(self):
        self.corpus = Corpus.from_file('deerwester')
        self.method = MultiSentiment()
        self.new_cols = 1


class SentiArtTest(LiuHuTest):
    def setUp(self):
        self.corpus = Corpus.from_file('deerwester')
        self.slo_corpus = Corpus.from_file('slo-opinion-corpus')[83:85]
        self.method = SentiArt()
        self.new_cols = 7

    def test_empty_slice_mean(self):
        # this should execute without raising an exception
        result = self.method.transform(self.slo_corpus)
        assert_allclose(result.X[0, -self.new_cols:], 0)


class LilahTest(LiuHuTest):
    def setUp(self):
        self.corpus = Corpus.from_file('deerwester')
        self.method = LilahSentiment()
        self.new_cols = 10


if __name__ == "__main__":
    unittest.main()
