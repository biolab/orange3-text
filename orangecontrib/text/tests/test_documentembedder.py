import unittest
from unittest.mock import patch
import asyncio
from numpy.testing import assert_array_equal

from orangecontrib.text.vectorization.document_embedder import DocumentEmbedder
from orangecontrib.text import Corpus

PATCH_METHOD = 'httpx.AsyncClient.post'


class DummyResponse:

    def __init__(self, content):
        self.content = content


def make_dummy_post(response, sleep=0):
    @staticmethod
    async def dummy_post(url, headers, data):
        await asyncio.sleep(sleep)
        return DummyResponse(content=response)
    return dummy_post


class DocumentEmbedderTest(unittest.TestCase):

    def setUp(self):
        self.embedder = DocumentEmbedder()  # default params
        self.corpus = Corpus.from_file('deerwester')

    def tearDown(self):
        self.embedder.clear_cache()

    @patch(PATCH_METHOD)
    def test_with_empty_corpus(self, mock):
        self.assertIsNone(self.embedder(self.corpus[:0])[0])
        self.assertIsNone(self.embedder(self.corpus[:0])[1])
        mock.request.assert_not_called()
        mock.get_response.assert_not_called()
        self.assertEqual(self.embedder._embedder._cache._cache_dict, dict())

    @patch(PATCH_METHOD, make_dummy_post(b'{"embedding": [0.3, 1]}'))
    def test_success_subset(self):
        res, skipped = self.embedder(self.corpus[[0]])
        assert_array_equal(res.X, [[0.3, 1]])
        self.assertEqual(len(self.embedder._embedder._cache._cache_dict), 1)
        self.assertIsNone(skipped)

    @patch(PATCH_METHOD, make_dummy_post(b'{"embedding": [0.3, 1]}'))
    def test_success_shapes(self):
        res, skipped = self.embedder(self.corpus)
        self.assertEqual(res.X.shape, (len(self.corpus), 2))
        self.assertEqual(len(res.domain), len(self.corpus.domain) + 2)
        self.assertIsNone(skipped)

    @patch(PATCH_METHOD, make_dummy_post(b''))
    def test_empty_response(self):
        with self.assertWarns(RuntimeWarning):
            res, skipped = self.embedder(self.corpus[[0]])
        self.assertIsNone(res)
        self.assertEqual(len(skipped), 1)
        self.assertEqual(len(self.embedder._embedder._cache._cache_dict), 0)

    @patch(PATCH_METHOD, make_dummy_post(b'str'))
    def test_invalid_response(self):
        with self.assertWarns(RuntimeWarning):
            res, skipped = self.embedder(self.corpus[[0]])
        self.assertIsNone(res)
        self.assertEqual(len(skipped), 1)
        self.assertEqual(len(self.embedder._embedder._cache._cache_dict), 0)

    @patch(PATCH_METHOD, make_dummy_post(b'{"embeddings": [0.3, 1]}'))
    def test_invalid_json_key(self):
        with self.assertWarns(RuntimeWarning):
            res, skipped = self.embedder(self.corpus[[0]])
        self.assertIsNone(res)
        self.assertEqual(len(skipped), 1)
        self.assertEqual(len(self.embedder._embedder._cache._cache_dict), 0)

    @patch(PATCH_METHOD, make_dummy_post(b'{"embedding": [0.3, 1]}'))
    def test_persistent_caching(self):
        self.assertEqual(len(self.embedder._embedder._cache._cache_dict), 0)
        self.embedder(self.corpus[[0]])
        self.assertEqual(len(self.embedder._embedder._cache._cache_dict), 1)
        self.embedder._embedder._cache.persist_cache()

        self.embedder = DocumentEmbedder()
        self.assertEqual(len(self.embedder._embedder._cache._cache_dict), 1)

        self.embedder.clear_cache()
        self.embedder = DocumentEmbedder()
        self.assertEqual(len(self.embedder._embedder._cache._cache_dict), 0)

    @patch(PATCH_METHOD, make_dummy_post(b'{"embedding": [0.3, 1]}'))
    def test_cache_for_different_languages(self):
        embedder = DocumentEmbedder(language='sl')
        embedder.clear_cache()
        self.assertEqual(len(embedder._embedder._cache._cache_dict), 0)
        embedder(self.corpus[[0]])
        self.assertEqual(len(embedder._embedder._cache._cache_dict), 1)
        embedder._embedder._cache.persist_cache()

        self.embedder = DocumentEmbedder()
        self.assertEqual(len(self.embedder._embedder._cache._cache_dict), 0)
        self.embedder._embedder._cache.persist_cache()

        embedder = DocumentEmbedder(language='sl')
        self.assertEqual(len(embedder._embedder._cache._cache_dict), 1)
        embedder.clear_cache()
        self.embedder.clear_cache()

    @patch(PATCH_METHOD, make_dummy_post(b'{"embedding": [0.3, 1]}'))
    def test_cache_for_different_aggregators(self):
        embedder = DocumentEmbedder(aggregator='max')
        embedder.clear_cache()
        self.assertEqual(len(embedder._embedder._cache._cache_dict), 0)
        embedder(self.corpus[[0]])
        self.assertEqual(len(embedder._embedder._cache._cache_dict), 1)
        embedder._embedder._cache.persist_cache()

        embedder = DocumentEmbedder(aggregator='min')
        self.assertEqual(len(embedder._embedder._cache._cache_dict), 1)
        embedder(self.corpus[[0]])
        self.assertEqual(len(embedder._embedder._cache._cache_dict), 2)

    @patch(PATCH_METHOD, make_dummy_post(b'{"embedding": [0.3, 1]}'))
    def test_with_statement(self):
        with self.embedder as embedder:
            res, skipped = embedder(self.corpus[[0]])
            assert_array_equal(res.X, [[0.3, 1]])

    @patch(PATCH_METHOD, make_dummy_post(b'{"embedding": [0.3, 1]}'))
    def test_cancel(self):
        self.assertFalse(self.embedder._embedder._cancelled)
        self.embedder._embedder._cancelled = True
        with self.assertRaises(Exception):
            self.embedder(self.corpus[[0]])

    @patch(PATCH_METHOD, side_effect=OSError)
    def test_connection_error(self, _):
        embedder = DocumentEmbedder()
        with self.assertRaises(ConnectionError):
            embedder(self.corpus[[0]])

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            self.embedder = DocumentEmbedder(language='eng')
        with self.assertRaises(ValueError):
            self.embedder = DocumentEmbedder(aggregator='average')

    def test_invalid_corpus_type(self):
        with self.assertRaises(ValueError):
            self.embedder(self.corpus[0])
