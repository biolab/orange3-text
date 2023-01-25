# pylint: disable=missing-docstring
import unittest
from unittest.mock import patch, call, ANY

import numpy as np
from Orange.data import Domain, StringVariable

from orangecontrib.text import Corpus
from orangecontrib.text.keywords import (
    tfidf_keywords,
    yake_keywords,
    rake_keywords,
    AggregationMethods,
)
from orangecontrib.text.keywords.mbert import (
    _split_key_phrases,
    _deduplicate,
    mbert_keywords,
    _BertServerCommunicator,
)


def corpus_mock(tokens):
    corpus = Corpus.from_numpy(
        Domain([], metas=[StringVariable("texts")]),
        np.empty((len(tokens), 0)),
        metas=np.array([[" ".join(t)] for t in tokens]),
    )
    return corpus


class TestTfIdf(unittest.TestCase):
    def test_extractor(self):
        corpus = corpus_mock([["foo", "bar", "baz", "baz"], ["foobar"], [" "]])
        keywords = tfidf_keywords(corpus)
        self.assertEqual(len(keywords), 3)
        self.assertEqual(len(keywords[0]), 3)
        self.assertEqual(len(keywords[1]), 1)
        self.assertEqual(len(keywords[2]), 0)

        self.assertEqual(keywords[0][1][0], "baz")
        self.assertGreaterEqual(keywords[0][1][1], 0.8)
        self.assertLessEqual(keywords[0][1][1], 1)

        self.assertEqual(keywords[0][0][0], "bar")
        self.assertEqual(keywords[0][2][0], "foo")

        self.assertEqual(keywords[1][0][0], "foobar")

    def test_empty_tokens(self):
        keywords = tfidf_keywords(corpus_mock([[" "]]))
        self.assertEqual(1, len(keywords))
        self.assertEqual(0, len(keywords[0]))

    def test_single_letter_tokens(self):
        keywords = tfidf_keywords(corpus_mock([["a", "b", "b"]]))
        self.assertEqual(keywords[0][0][0], "a")
        self.assertEqual(keywords[0][1][0], "b")


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


KEYWORDS = [
    [("kw1", 0.5), ("kw2 kw3", 0.3), ("kw2", 0.6)],
    [("kw5 kw8 kw7", 0.2), ("kw8", 0.1)],
]


class TestMBERT(unittest.TestCase):
    def test_split_phrases(self):
        # length 1
        self.assertEqual([KEYWORDS[0][0]], _split_key_phrases(KEYWORDS[0][0], 1))
        self.assertEqual(
            [("kw2", 0.3), ("kw3", 0.3)], _split_key_phrases(KEYWORDS[0][1], 1)
        )
        self.assertEqual(
            [("kw5", 0.2), ("kw8", 0.2), ("kw7", 0.2)],
            _split_key_phrases(KEYWORDS[1][0], 1),
        )

        # length 2
        self.assertEqual([KEYWORDS[0][0]], _split_key_phrases(KEYWORDS[0][0], 2))
        self.assertEqual([("kw2 kw3", 0.3)], _split_key_phrases(KEYWORDS[0][1], 2))
        self.assertEqual(
            [("kw5 kw8", 0.2), ("kw8 kw7", 0.2)],
            _split_key_phrases(KEYWORDS[1][0], 2),
        )

        # length 3
        self.assertEqual([KEYWORDS[0][0]], _split_key_phrases(KEYWORDS[0][0], 3))
        self.assertEqual([("kw2 kw3", 0.3)], _split_key_phrases(KEYWORDS[0][1], 3))
        self.assertEqual([("kw5 kw8 kw7", 0.2)], _split_key_phrases(KEYWORDS[1][0], 3))

    def test_deduplicate(self):
        # no duplicates just sort
        self.assertEqual(
            [("kw4", 0.5), ("kw2", 0.3), ("kw1", 0.2)],
            _deduplicate([("kw1", 0.2), ("kw2", 0.3), ("kw4", 0.5)]),
        )
        # remove duplicates and sort
        self.assertEqual(
            [("kw4", 0.5), ("kw1", 0.3)],
            _deduplicate([("kw1", 0.2), ("kw1", 0.3), ("kw4", 0.5)]),
        )
        self.assertEqual(
            [("kw1", 0.5), ("kw2", 0.3)],
            _deduplicate([("kw1", 0.2), ("kw2", 0.3), ("kw1", 0.5)]),
        )

    @patch(
        "orangecontrib.text.keywords.mbert._BertServerCommunicator.embedd_data",
        return_value=KEYWORDS,
    )
    def test_mbert_keywords(self, _):
        # max len 3 - no postprocessing
        res = mbert_keywords(["Text 1", "Text 2"], max_len=3)
        expected = [
            [("kw2", 0.6), ("kw1", 0.5), ("kw2 kw3", 0.3)],
            [("kw5 kw8 kw7", 0.2), ("kw8", 0.1)],
        ]
        self.assertListEqual(expected, res)

        # max len 2 - some words split
        res = mbert_keywords(["Text 1", "Text 2"], max_len=2)
        expected = [
            [("kw2", 0.6), ("kw1", 0.5), ("kw2 kw3", 0.3)],
            [("kw5 kw8", 0.2), ("kw8 kw7", 0.2), ("kw8", 0.1)],
        ]
        self.assertListEqual(expected, res)

        # max len 1 - all words split and duplicates removed
        res = mbert_keywords(["Text 1", "Text 2"], max_len=1)
        expected = [
            [("kw2", 0.6), ("kw1", 0.5), ("kw3", 0.3)],
            [("kw5", 0.2), ("kw8", 0.2), ("kw7", 0.2)],
        ]
        self.assertListEqual(expected, res)


@patch(
    "orangecontrib.text.keywords.mbert._BertServerCommunicator._send_request",
    return_value=[("kw1", 0.1), ("kw2", 0.2)],
)
class TestBertServerCommunicator(unittest.TestCase):
    def test_data_encoding(self, mock):
        embedder = _BertServerCommunicator(
            model_name="mbert-keywords",
            max_parallel_requests=1,
            server_url="https://api.garaza.io",
            embedder_type="text",
        )
        embedder.clear_cache()
        embedder.embedd_data(["Text1", "Text2"])
        mock.assert_has_awaits(
            [
                call(ANY, b"eNoLSa0oMQQABb4B1w==", ANY),
                call(ANY, b"eNoLSa0oMQIABb8B2A==", ANY),
            ]
        )


class TestAggregationMethods(unittest.TestCase):
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
