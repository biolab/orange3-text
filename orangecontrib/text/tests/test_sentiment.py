import unittest

from numpy.testing import assert_allclose

from orangecontrib.text.corpus import Corpus
from orangecontrib.text.sentiment import (
    LiuHuSentiment,
    VaderSentiment,
    MultiSentiment,
    SentiArt,
    LilahSentiment,
)


class BaseTest:
    # base test use to avoid SentimentTests being called by unittest.main
    class SentimentTests(unittest.TestCase):
        METHOD = None
        LANGUAGE = None
        NEW_COLS = 1

        def setUp(self):
            self.corpus = Corpus.from_file("deerwester")
            args = (self.LANGUAGE,) if self.LANGUAGE is not None else ()
            self.method = self.METHOD(*args)

        def test_transform(self):
            sentiment = self.method.transform(self.corpus)
            self.assertIsInstance(sentiment, Corpus)
            self.assertEqual(
                len(sentiment.domain.variables),
                len(self.corpus.domain.variables) + self.NEW_COLS,
            )

        def test_copy(self):
            sentiment_t = self.method.transform(self.corpus)
            self.assertIsNot(self.corpus, sentiment_t)

        def test_compute_values(self):
            sentiment = self.method.transform(self.corpus)
            computed = Corpus.from_table(sentiment.domain, self.corpus)

            self.assertEqual(sentiment.domain, computed.domain)
            self.assertTrue((sentiment.X == computed.X).all())

        def test_compute_values_to_different_domain(self):
            destination = Corpus.from_file("andersen")

            self.assertFalse(self.corpus.domain.attributes)
            self.assertFalse(destination.domain.attributes)

            sentiment = self.method.transform(self.corpus)
            computed = destination.transform(sentiment.domain)

            self.assertTrue(sentiment.domain.attributes)
            self.assertEqual(sentiment.domain.attributes, computed.domain.attributes)

        def test_empty_corpus(self):
            sentiment = self.method.transform(self.corpus[:0])
            self.assertEqual(
                len(sentiment.domain.variables),
                len(self.corpus.domain.variables) + self.NEW_COLS,
            )
            self.assertEqual(len(sentiment), 0)

        def test_language_from_corpus(self):
            """Init method without language, it should be provided by corpus"""
            method = self.METHOD()
            # Lilah does not support english - change language to one supported
            self.corpus.attributes["language"] = self.LANGUAGE
            sentiment = method.transform(self.corpus)
            self.assertIsInstance(sentiment, Corpus)
            self.assertEqual(
                len(sentiment.domain.variables),
                len(self.corpus.domain.variables) + self.NEW_COLS,
            )


class LiuHuTest(BaseTest.SentimentTests):
    METHOD = LiuHuSentiment
    LANGUAGE = "en"


class LiuHuSlovenian(BaseTest.SentimentTests):
    METHOD = LiuHuSentiment
    LANGUAGE = "sl"

    def setUp(self):
        super().setUp()
        self.corpus = Corpus.from_file("slo-opinion-corpus")

    def test_compute_values_to_different_domain(self):
        """Skip test_compute_values_to_different_domain for Slovenian"""
        pass


class VaderTest(BaseTest.SentimentTests):
    METHOD = VaderSentiment
    NEW_COLS = 4


class MultiSentimentTest(BaseTest.SentimentTests):
    METHOD = MultiSentiment
    LANGUAGE = "en"


class SentiArtTest(BaseTest.SentimentTests):
    METHOD = SentiArt
    LANGUAGE = "en"
    NEW_COLS = 7

    def test_empty_slice_mean(self):
        # this should execute without raising an exception
        slo_corpus = Corpus.from_file("slo-opinion-corpus")[83:85]
        result = self.method.transform(slo_corpus)
        assert_allclose(result.X[0, -self.NEW_COLS :], 0)


class LilahTest(BaseTest.SentimentTests):
    METHOD = LilahSentiment
    LANGUAGE = "sl"
    NEW_COLS = 10


if __name__ == "__main__":
    unittest.main()
