import unittest
from unittest.mock import patch, ANY
import asyncio

from Orange.misc.utils.embedder_utils import EmbedderCache
from numpy.testing import assert_array_equal

from orangecontrib.text.vectorization.document_embedder import DocumentEmbedder
from orangecontrib.text import Corpus

PATCH_METHOD = 'httpx.AsyncClient.post'


class DummyResponse:

    def __init__(self, content):
        self.content = content


def make_dummy_post(response, sleep=0):
    @staticmethod
    async def dummy_post(url, headers, data=None, content=None):
        assert data or content
        await asyncio.sleep(sleep)
        return DummyResponse(content=response)
    return dummy_post


class DocumentEmbedderTest(unittest.TestCase):

    def setUp(self):
        self.embedder = DocumentEmbedder()  # default params
        self.corpus = Corpus.from_file('deerwester')
        self.embedder.clear_cache("en")

    def tearDown(self):
        self.embedder.clear_cache("en")

    @patch(PATCH_METHOD)
    def test_with_empty_corpus(self, mock):
        self.assertIsNone(self.embedder.transform(self.corpus[:0])[0])
        self.assertIsNone(self.embedder.transform(self.corpus[:0])[1])
        mock.request.assert_not_called()
        mock.get_response.assert_not_called()
        self.assertEqual(EmbedderCache("fasttext-en")._cache_dict, dict())

    @patch(PATCH_METHOD, make_dummy_post(b'{"embedding": [0.3, 1]}'))
    def test_success_subset(self):
        res, skipped = self.embedder.transform(self.corpus[[0]])
        assert_array_equal(res.X, [[0.3, 1]])
        self.assertEqual(len(EmbedderCache("fasttext-en")._cache_dict), 1)
        self.assertIsNone(skipped)

    @patch(PATCH_METHOD, make_dummy_post(b'{"embedding": [0.3, 1]}'))
    def test_success_shapes(self):
        res, skipped = self.embedder.transform(self.corpus)
        self.assertEqual(res.X.shape, (len(self.corpus), 2))
        self.assertEqual(len(res.domain.variables),
                         len(self.corpus.domain.variables) + 2)
        self.assertIsNone(skipped)

    @patch(PATCH_METHOD, make_dummy_post(b''))
    def test_empty_response(self):
        with self.assertWarns(RuntimeWarning):
            res, skipped = self.embedder.transform(self.corpus[[0]])
        self.assertIsNone(res)
        self.assertEqual(len(skipped), 1)
        self.assertEqual(len(EmbedderCache("fasttext-en")._cache_dict), 0)

    @patch(PATCH_METHOD, make_dummy_post(b'str'))
    def test_invalid_response(self):
        with self.assertWarns(RuntimeWarning):
            res, skipped = self.embedder.transform(self.corpus[[0]])
        self.assertIsNone(res)
        self.assertEqual(len(skipped), 1)
        self.assertEqual(len(EmbedderCache("fasttext-en")._cache_dict), 0)

    @patch(PATCH_METHOD, make_dummy_post(b'{"embeddings": [0.3, 1]}'))
    def test_invalid_json_key(self):
        with self.assertWarns(RuntimeWarning):
            res, skipped = self.embedder.transform(self.corpus[[0]])
        self.assertIsNone(res)
        self.assertEqual(len(skipped), 1)
        self.assertEqual(len(EmbedderCache("fasttext-en")._cache_dict), 0)

    @patch(PATCH_METHOD, make_dummy_post(b'{"embedding": [0.3, 1]}'))
    def test_persistent_caching(self):
        self.assertEqual(len(EmbedderCache("fasttext-en")._cache_dict), 0)
        self.embedder.transform(self.corpus[[0]])
        self.assertEqual(len(EmbedderCache("fasttext-en")._cache_dict), 1)

        self.embedder = DocumentEmbedder()
        self.assertEqual(len(EmbedderCache("fasttext-en")._cache_dict), 1)

        self.embedder.clear_cache("en")
        self.embedder = DocumentEmbedder()
        self.assertEqual(len(EmbedderCache("fasttext-en")._cache_dict), 0)

    @patch(PATCH_METHOD, make_dummy_post(b'{"embedding": [0.3, 1]}'))
    def test_different_languages(self):
        self.corpus.attributes["language"] = "sl"

        embedder = DocumentEmbedder()
        embedder.clear_cache("sl")
        self.assertEqual(len(EmbedderCache("fasttext-sl")._cache_dict), 0)
        embedder.transform(self.corpus[[0]])
        self.assertEqual(len(EmbedderCache("fasttext-sl")._cache_dict), 1)
        self.assertEqual(len(EmbedderCache("fasttext-en")._cache_dict), 0)
        self.assertEqual(len(EmbedderCache("fasttext-sl")._cache_dict), 1)
        embedder.clear_cache("sl")

    @patch(PATCH_METHOD, make_dummy_post(b'{"embedding": [0.3, 1]}'))
    def test_cache_for_different_aggregators(self):
        embedder = DocumentEmbedder(aggregator='max')
        embedder.clear_cache("en")

        self.assertEqual(len(EmbedderCache("fasttext-en")._cache_dict), 0)
        embedder.transform(self.corpus[[0]])
        self.assertEqual(len(EmbedderCache("fasttext-en")._cache_dict), 1)

        embedder = DocumentEmbedder(aggregator='min')
        self.assertEqual(len(EmbedderCache("fasttext-en")._cache_dict), 1)
        embedder.transform(self.corpus[[0]])
        self.assertEqual(len(EmbedderCache("fasttext-en")._cache_dict), 2)

    @patch(PATCH_METHOD, side_effect=OSError)
    def test_connection_error(self, _):
        embedder = DocumentEmbedder()
        with self.assertRaises(ConnectionError):
            embedder.transform(self.corpus[[0]])

    def test_invalid_parameters(self):
        with self.assertRaises(AssertionError):
            self.embedder = DocumentEmbedder(language='eng')
        with self.assertRaises(AssertionError):
            self.embedder = DocumentEmbedder(aggregator='average')

    @patch("orangecontrib.text.vectorization.document_embedder._ServerEmbedder")
    def test_set_language(self, m):
        # method 1: language from corpus
        self.corpus.attributes["language"] = "sl"
        embedder = DocumentEmbedder()
        embedder.transform(self.corpus)
        m.assert_called_with(
            "mean",
            model_name="fasttext-sl",
            max_parallel_requests=ANY,
            server_url=ANY,
            embedder_type=ANY,
        )

        # method 2: language explicitly set
        embedder = DocumentEmbedder(language="es")
        embedder.transform(self.corpus)
        m.assert_called_with(
            "mean",
            model_name="fasttext-es",
            max_parallel_requests=ANY,
            server_url=ANY,
            embedder_type=ANY,
        )


if __name__ == "__main__":
    unittest.main()
