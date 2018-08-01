import unittest

from orangecontrib.text.corpus import Corpus
from orangecontrib.text.sentiment import Liu_Hu_Sentiment, Vader_Sentiment


class LiuHuTest(unittest.TestCase):
    def setUp(self):
        self.corpus = Corpus.from_file('deerwester')
        self.method = Liu_Hu_Sentiment('English')
        self.new_cols = 1

    def test_transform(self):
        sentiment = self.method.transform(self.corpus)
        self.assertIsInstance(sentiment, Corpus)
        self.assertEqual(len(sentiment.domain),
                         len(self.corpus.domain) + self.new_cols)

    def test_copy(self):
        sentiment_t = self.method.transform(self.corpus, copy=True)
        self.assertIsNot(self.corpus, sentiment_t)

        sentiment_f = self.method.transform(self.corpus, copy=False)
        self.assertIs(self.corpus, sentiment_f)

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
        self.assertEqual(len(sentiment.domain),
                         len(self.corpus.domain) + self.new_cols)
        self.assertEqual(len(sentiment), 0)

class Liu_Hu_Slovenian(unittest.TestCase):
    def setUp(self):
        self.corpus = Corpus.from_file('slo-opinion-corpus')
        self.method = Liu_Hu_Sentiment('Slovenian')
        self.new_cols = 1

    def test_transform(self):
        sentiment = self.method.transform(self.corpus)
        self.assertIsInstance(sentiment, Corpus)
        self.assertEqual(len(sentiment.domain),
                         len(self.corpus.domain) + self.new_cols)

    def test_copy(self):
        sentiment_t = self.method.transform(self.corpus, copy=True)
        self.assertIsNot(self.corpus, sentiment_t)

        sentiment_f = self.method.transform(self.corpus, copy=False)
        self.assertIs(self.corpus, sentiment_f)

    def test_compute_values(self):
        sentiment = self.method.transform(self.corpus)
        computed = Corpus.from_table(sentiment.domain, self.corpus)

        self.assertEqual(sentiment.domain, computed.domain)
        self.assertTrue((sentiment.X == computed.X).all())

    def test_empty_corpus(self):
        corpus = self.corpus[:0]
        sentiment = self.method.transform(corpus)
        self.assertEqual(len(sentiment.domain),
                         len(self.corpus.domain) + self.new_cols)
        self.assertEqual(len(sentiment), 0)


class VaderTest(LiuHuTest):
    def setUp(self):
        self.corpus = Corpus.from_file('deerwester')
        self.method = Vader_Sentiment()
        self.new_cols = 4


if __name__ == "__main__":
    unittest.main()
