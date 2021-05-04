# pylint: disable=missing-docstring
import unittest
from unittest.mock import patch

import numpy as np
from orangecontrib.text.keywords import tfidf_keywords, yake_keywords, \
    rake_keywords, AggregationMethods, embedding_keywords


class TestTfIdf(unittest.TestCase):
    def test_extractor(self):
        tokens = [["foo", "bar", "baz", "baz"],
                  ["foobar"],
                  []]
        keywords = tfidf_keywords(tokens)
        self.assertEqual(len(keywords), 3)
        self.assertEqual(len(keywords[0]), 3)
        self.assertEqual(len(keywords[1]), 1)
        self.assertEqual(len(keywords[2]), 0)

        self.assertEqual(keywords[0][0][0], "baz")
        self.assertGreaterEqual(keywords[0][0][1], 0.8)
        self.assertLessEqual(keywords[0][0][1], 1)

        self.assertEqual(keywords[0][1][0], "bar")
        self.assertEqual(keywords[0][2][0], "foo")

        self.assertEqual(keywords[1][0][0], "foobar")

    def test_empty_tokens(self):
        self.assertRaises(ValueError, tfidf_keywords, [])
        self.assertRaises(ValueError, tfidf_keywords, [[]])

    def test_single_letter_tokens(self):
        keywords = tfidf_keywords([["a", "b", "b", " "]])
        self.assertEqual(keywords[0][0][0], " ")
        self.assertEqual(keywords[0][1][0], "b")
        self.assertEqual(keywords[0][2][0], "a")


class TestYake(unittest.TestCase):
    def test_extractor(self):
        documents = [
            "Human machine interface for lab abc computer applications",
            "A survey of user opinion of computer system response time"
        ]
        keywords = yake_keywords(documents)
        self.assertEqual(len(keywords), 2)
        self.assertEqual(len(keywords[0]), 7)
        self.assertEqual(len(keywords[1]), 7)

    def test_empty_documents(self):
        keywords = yake_keywords([])
        self.assertEqual(len(keywords), 0)

    def test_single_letter_documents(self):
        keywords = yake_keywords(["foo", "", "too"])
        self.assertEqual(len(keywords), 3)
        self.assertEqual(len(keywords[0]), 1)
        self.assertEqual(len(keywords[1]), 0)
        self.assertEqual(len(keywords[2]), 0)


class TestRake(unittest.TestCase):
    def test_extractor(self):
        documents = [
            "Human machine interface for lab abc computer applications",
            "A survey of user opinion of computer system response time"
        ]
        keywords = rake_keywords(documents)
        self.assertEqual(len(keywords), 2)
        self.assertEqual(len(keywords[0]), 0)
        self.assertEqual(len(keywords[1]), 1)

    def test_empty_documents(self):
        keywords = rake_keywords([])
        self.assertEqual(len(keywords), 0)

    def test_single_letter_documents(self):
        keywords = rake_keywords(["foo", "", "too"])
        self.assertEqual(len(keywords), 3)
        self.assertEqual(len(keywords[0]), 1)
        self.assertEqual(len(keywords[1]), 0)
        self.assertEqual(len(keywords[2]), 0)


def mock_embedding(_, tokens, __):
    emb_dict = {"foo": [1, 2, 3], "bar": [2, 3, 4], "baz": [4, 4, 4], "fobar": [1, 2, 3], "a": [1, 2, 3], "b": [4, 5, 6]}
    emb = [np.mean([emb_dict.get(t, [1, 1, 1]) for t in tt], axis=0).tolist() if tt else [0, 0, 0] for tt in tokens]
    return emb


@patch("orangecontrib.text.vectorization.document_embedder.DocumentEmbedder.__call__", mock_embedding)
class TestEmbedding(unittest.TestCase):
    def test_extractor(self):
        tokens = [["foo", "bar", "baz", "baz"],
                  ["foobar"],
                  []]
        keywords = embedding_keywords(tokens)
        self.assertEqual(len(keywords), 3)
        self.assertEqual(len(keywords[0]), 3)
        self.assertEqual(len(keywords[1]), 1)
        self.assertEqual(len(keywords[2]), 0)

        self.assertEqual(keywords[0][0][0], "baz")
        np.testing.assert_almost_equal(keywords[0][0][1], 0.00780, decimal=4)

        self.assertEqual(keywords[0][1][0], "bar")
        self.assertEqual(keywords[0][2][0], "foo")

        self.assertEqual(keywords[1][0][0], "foobar")

    def test_empty_tokens(self):
        keywords = embedding_keywords([])
        self.assertEqual(len(keywords), 0)

    def test_single_letter_tokens(self):
        keywords = embedding_keywords([["a", "b", "b", " "]])
        self.assertEqual(keywords[0][0][0], "b")
        self.assertEqual(keywords[0][1][0], " ")
        self.assertEqual(keywords[0][2][0], "a")


class TestAggregationMethods(unittest.TestCase):
    def test_aggregate_mean(self):
        keywords = [[("foo", 0.1)],
                    [("foo", 0.3), ("bar", 0.6)],
                    [("foo", 0.5)]]
        scores = AggregationMethods.mean(keywords)
        self.assertEqual(scores[0][0], "foo")
        self.assertEqual(scores[1][0], "bar")
        self.assertAlmostEqual(scores[0][1], 0.3)
        self.assertAlmostEqual(scores[1][1], 0.2)

    def test_aggregate_median(self):
        keywords = [[("foo", 0.1)],
                    [("foo", 0.3), ("bar", 0.6)],
                    [("foo", 0.5)]]
        scores = AggregationMethods.median(keywords)
        self.assertEqual(scores[0], ("foo", 0.3))
        self.assertEqual(scores[1], ("bar", 0.6))

    def test_aggregate_min(self):
        keywords = [[("foo", 0.1)],
                    [("foo", 0.3), ("bar", 0.6)],
                    [("foo", 0.5)]]
        scores = AggregationMethods.min(keywords)
        self.assertEqual(scores[0], ("foo", 0.1))
        self.assertEqual(scores[1], ("bar", 0.6))

    def test_aggregate_max(self):
        keywords = [[("foo", 0.1)],
                    [("foo", 0.3), ("bar", 0.6)],
                    [("foo", 0.5)]]
        scores = AggregationMethods.max(keywords)
        self.assertEqual(scores[0], ("foo", 0.5))
        self.assertEqual(scores[1], ("bar", 0.6))

    def test_aggregate(self):
        keywords = [[("foo", 0.1)],
                    [("foo", 0.3), ("bar", 0.6)],
                    [("foo", 0.5)]]
        scores = AggregationMethods.aggregate(keywords, AggregationMethods.MEAN)
        self.assertEqual(scores[0][0], "foo")
        self.assertEqual(scores[1][0], "bar")
        self.assertAlmostEqual(scores[0][1], 0.3)
        self.assertAlmostEqual(scores[1][1], 0.2)

        scores = AggregationMethods.aggregate(keywords,
                                              AggregationMethods.MEDIAN)
        self.assertEqual(scores[0], ("foo", 0.3))
        self.assertEqual(scores[1], ("bar", 0.6))

        scores = AggregationMethods.aggregate(keywords, AggregationMethods.MIN)
        self.assertEqual(scores[0], ("foo", 0.1))
        self.assertEqual(scores[1], ("bar", 0.6))

        scores = AggregationMethods.aggregate(keywords, AggregationMethods.MAX)
        self.assertEqual(scores[0], ("foo", 0.5))
        self.assertEqual(scores[1], ("bar", 0.6))


if __name__ == "__main__":
    unittest.main()
