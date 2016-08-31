import unittest

import numpy as np

from orangecontrib.text import vectorization
from orangecontrib.text.topics import LdaWrapper, HdpWrapper, LsiWrapper
from orangecontrib.text.corpus import Corpus
from orangecontrib.text import preprocess


class BaseTests:
    def test_fit_transform(self):
        topics = self.model.fit_transform(self.corpus)
        self.assertEqual(len(topics), len(self.corpus))
        # np.testing.assert_allclose(topics.X.sum(axis=1), np.ones(len(self.corpus)), rtol=.01)
        return topics

    def test_get_topic_table_by_id(self):
        self.model.fit(self.corpus)
        topic1 = self.model.get_topics_table_by_id(1)
        self.assertEqual(len(topic1), len(self.corpus.dictionary))
        self.assertEqual(topic1.metas.shape, (len(self.corpus.dictionary), 2))
        # self.assertAlmostEqual(topic1.W.sum(), 1.)
        self.assertFalse(any(topic1.W == np.nan))

    def test_top_words_by_topic(self):
        self.model.fit(self.corpus)
        words = self.model.get_top_words_by_id(1, num_of_words=10)
        self.assertTrue(all([isinstance(word, str) for word in words]))
        self.assertEqual(len(words), 10)

    def test_vectorized(self):
        corpus = vectorization.BowVectorizer().transform(self.corpus, copy=True)
        topics = self.model.fit_transform(corpus)
        self.assertIsInstance(topics, Corpus)

    def test_report_callback(self):
        prev_progress = -1

        def callback(progress):
            nonlocal prev_progress
            self.assertGreater(progress, prev_progress)
            prev_progress = progress

        self.model.fit(self.corpus, on_progress=callback)
        self.assertLessEqual(prev_progress, 100)

    def test_empty_corpus(self):
        p = preprocess.Preprocessor(tokenizer=preprocess.RegexpTokenizer(pattern='unmatchable'))
        empty = p(self.corpus)
        self.assertIsNone(self.model.fit(empty))

    def test_get_top_words(self):
        self.model.fit(self.corpus)
        self.assertRaises(ValueError, self.model.get_topics_table_by_id, 1000)


class LDATests(unittest.TestCase, BaseTests):
    def setUp(self):
        self.corpus = Corpus.from_file('deerwester')
        self.model = LdaWrapper(num_topics=5)

    def test_too_large_id(self):
        self.model.fit(self.corpus)
        with self.assertRaises(ValueError):
            self.model.get_topics_table_by_id(6)

    def test_fit_transform(self):
        corpus = super().test_fit_transform()
        self.assertEqual(len(corpus.domain.attributes), 5)
        self.assertEqual(corpus.X.shape, (len(self.corpus), 5))


class HdpTest(unittest.TestCase, BaseTests):
    def setUp(self):
        self.corpus = Corpus.from_file('deerwester')
        self.model = HdpWrapper()


class LsiTest(unittest.TestCase, BaseTests):
    def setUp(self):
        self.corpus = Corpus.from_file('deerwester')
        self.model = LsiWrapper(num_topics=5)
