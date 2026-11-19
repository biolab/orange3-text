import unittest
from unittest.mock import patch, ANY, MagicMock
import asyncio

from numpy.testing import assert_array_equal

from Orange.misc.utils.embedder_utils import EmbedderCache

from orangecontrib.text.vectorization.document_embedder import DocumentEmbedder, OAIDocumentEmbedder
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
        embedder = DocumentEmbedder(aggregator='Max')
        embedder.clear_cache("en")

        self.assertEqual(len(EmbedderCache("fasttext-en")._cache_dict), 0)
        embedder.transform(self.corpus[[0]])
        self.assertEqual(len(EmbedderCache("fasttext-en")._cache_dict), 1)

        embedder = DocumentEmbedder(aggregator='Min')
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


class TestOAIDocumentEmbedder(unittest.TestCase):
    """Test OAIDocumentEmbedder"""

    def setUp(self):
        self.corpus = Corpus.from_file('deerwester')
        self.base_url = "https://api.openai.com/v1"
        self.api_key = "test-api-key"
        self.model = "text-embedding-ada-002"
        self.embedder = OAIDocumentEmbedder(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model
        )
        self.embedder._cache._cache_dict.clear()

    def _make_mock_embedding(self, embedding_list):
        """Create a mock embedding response object."""
        mock_embedding = MagicMock()
        mock_embedding.embedding = embedding_list
        return mock_embedding

    def _make_mock_response(self, embeddings_list):
        """Create a mock API response with a list of embeddings."""
        mock_response = MagicMock()
        mock_response.data = [self._make_mock_embedding(e) for e in embeddings_list]
        return mock_response

    @patch('openai.OpenAI')
    def test_init(self, mock_openai):
        """Test OAIDocumentEmbedder initialization."""
        embedder = OAIDocumentEmbedder(
            base_url="https://api.example.com",
            api_key="key123",
            model="gpt-4"
        )
        self.assertEqual(embedder.base_url, "https://api.example.com")
        self.assertEqual(embedder.api_key, "key123")
        self.assertEqual(embedder.model, "gpt-4")

    @patch('openai.OpenAI')
    def test_with_empty_corpus(self, mock_openai):
        """Test transform with an empty corpus."""
        empty_corpus = self.corpus[:0]
        result, skipped = self.embedder.transform(empty_corpus)
        # Empty corpus should return None for the first element of the tuple
        self.assertEqual(len(result), 0)
        self.assertIsNone(skipped)
        # No API calls should be made
        mock_openai.return_value.embeddings.create.assert_not_called()
        # Cache should remain empty
        self.assertEqual(len(self.embedder._cache._cache_dict), 0)

    @patch('openai.OpenAI')
    def test_success_single_document(self, mock_openai):
        """Test successful embedding of a single document."""
        mock_client = mock_openai.return_value
        mock_response = self._make_mock_response([[0.1, 0.2, 0.3]])
        mock_client.embeddings.create.return_value = mock_response

        result, skipped = self.embedder.transform(self.corpus[[0]])

        # Check result is a Corpus
        self.assertIsNotNone(result)
        self.assertIsNone(skipped)
        # Check embedding values
        assert_array_equal(result.X, [[0.1, 0.2, 0.3]])
        # Check cache was populated
        self.assertEqual(len(self.embedder._cache._cache_dict), 1)
        # Verify API was called with correct parameters
        mock_client.embeddings.create.assert_called_once()
        call_args = mock_client.embeddings.create.call_args
        self.assertEqual(call_args.kwargs['model'], self.model)
        self.assertEqual(call_args.kwargs['encoding_format'], "float")
        self.assertIn(self.corpus.documents[0], call_args.kwargs['input'])

    @patch('openai.OpenAI')
    def test_success_multiple_documents(self, mock_openai):
        """Test successful embedding of multiple documents."""
        mock_client = mock_openai.return_value
        embeddings = [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6]
        ]
        mock_response = self._make_mock_response(embeddings)
        mock_client.embeddings.create.return_value = mock_response

        result, skipped = self.embedder.transform(self.corpus[[0, 1, 2]])

        self.assertIsNotNone(result)
        self.assertIsNone(skipped)
        assert_array_equal(result.X, embeddings)
        self.assertEqual(len(self.embedder._cache._cache_dict), 3)

    @patch('openai.OpenAI')
    def test_success_shapes(self, mock_openai):
        """Test that output shapes are correct."""
        corpus = self.corpus[:5]
        mock_client = mock_openai.return_value
        mock_response = self._make_mock_response([[0.1, 0.2, 0.3]] * 5)
        mock_client.embeddings.create.return_value = mock_response

        result, skipped = self.embedder.transform(corpus)

        self.assertEqual(result.X.shape, (len(corpus), 3))
        # Check that new features were added
        self.assertEqual(len(result.domain.variables),
                         len(self.corpus.domain.variables) + 3)
        # Verify feature names
        feature_names = [v.name for v in result.domain.attributes]
        self.assertIn("Dim1", feature_names)
        self.assertIn("Dim2", feature_names)
        self.assertIn("Dim3", feature_names)

    @patch('openai.OpenAI')
    def test_persistent_caching(self, mock_openai):
        """Test that cache persists across embedder instances."""
        mock_client = mock_openai.return_value
        mock_response = self._make_mock_response([[0.5, 0.6, 0.7]])
        mock_client.embeddings.create.return_value = mock_response

        # First transform - cache should be empty
        self.assertEqual(len(self.embedder._cache._cache_dict), 0)
        self.embedder.transform(self.corpus[[0]])
        self.assertEqual(len(self.embedder._cache._cache_dict), 1)

        # Create a new embedder instance - cache should still have the data
        new_embedder = OAIDocumentEmbedder(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model
        )
        self.assertEqual(len(new_embedder._cache._cache_dict), 1)

    @patch('openai.OpenAI')
    def test_cache_avoids_duplicate_api_calls(self, mock_openai):
        """Test that cached documents don't trigger additional API calls."""
        mock_client = mock_openai.return_value
        mock_response = self._make_mock_response([[0.1, 0.2, 0.3]])
        mock_client.embeddings.create.return_value = mock_response

        # First transform - should call API
        self.embedder.transform(self.corpus[[0]])
        self.assertEqual(mock_client.embeddings.create.call_count, 1)

        # Second transform with same document - should NOT call API again
        self.embedder.transform(self.corpus[[0]])
        self.assertEqual(mock_client.embeddings.create.call_count, 1)

    @patch('openai.OpenAI')
    def test_cache_partial_hits(self, mock_openai):
        """Test that only uncached documents trigger API calls."""
        mock_client = mock_openai.return_value
        mock_response = self._make_mock_response([[0.1, 0.2, 0.3]])
        mock_client.embeddings.create.return_value = mock_response

        # Embed first document
        self.embedder.transform(self.corpus[[0]])
        self.assertEqual(mock_client.embeddings.create.call_count, 1)

        # Embed second document - should only embed the new one
        self.embedder.transform(self.corpus[[1]])
        self.assertEqual(mock_client.embeddings.create.call_count, 2)

        # Embed both again - should not call API at all
        self.embedder.transform(self.corpus[[0, 1]])
        self.assertEqual(mock_client.embeddings.create.call_count, 2)

    @patch('openai.OpenAI')
    def test_different_models_different_caches(self, mock_openai):
        """Test that different models use different caches."""
        embedder1 = OAIDocumentEmbedder(
            base_url=self.base_url,
            api_key=self.api_key,
            model="model-a"
        )
        embedder2 = OAIDocumentEmbedder(
            base_url=self.base_url,
            api_key=self.api_key,
            model="model-b"
        )

        self.assertNotEqual(embedder1._cache._cache_file_path, embedder2._cache._cache_file_path)

    @patch('openai.OpenAI')
    def test_different_base_urls_different_caches(self, mock_openai):
        """Test that different base URLs use different caches."""
        embedder1 = OAIDocumentEmbedder(
            base_url="https://api.openai.com/v1",
            api_key=self.api_key,
            model="text-embedding-ada-002"
        )
        embedder2 = OAIDocumentEmbedder(
            base_url="https://api.anthropic.com/v1",
            api_key=self.api_key,
            model="claude-embed"
        )

        self.assertNotEqual(embedder1._cache._cache_file_path, embedder2._cache._cache_file_path)

    @patch('openai.OpenAI')
    def test_progress_callback(self, mock_openai):
        """Test that progress callback is called during embedding."""
        mock_client = mock_openai.return_value
        embeddings = [[0.1, 0.2] for _ in range(5)]
        mock_response = self._make_mock_response(embeddings)
        mock_client.embeddings.create.return_value = mock_response

        callback = MagicMock()
        self.embedder.transform(self.corpus[:5], callback=callback)

        # Progress callback should have been called
        callback.assert_called()
        # The last call should indicate completion
        last_call = callback.call_args_list[-1]
        self.assertEqual(last_call[0][0], 1.0)

    @patch('openai.OpenAI')
    def test_api_error_propagates(self, mock_openai):
        """Test that API errors are propagated."""
        mock_client = mock_openai.return_value
        mock_client.embeddings.create.side_effect = Exception("API Error")

        with self.assertRaises(Exception):
            self.embedder.transform(self.corpus[[0]])

    @patch('openai.OpenAI')
    def test_response_as_list(self, mock_openai):
        """Test handling of response when it's already a list (not an object)."""
        mock_client = mock_openai.return_value
        # Simulate response being a list directly
        mock_embedding = self._make_mock_embedding([0.1, 0.2, 0.3])
        mock_client.embeddings.create.return_value = [mock_embedding]

        result, skipped = self.embedder.transform(self.corpus[[0]])

        self.assertIsNotNone(result)
        self.assertIsNone(skipped)
        assert_array_equal(result.X, [[0.1, 0.2, 0.3]])

    @patch('openai.OpenAI')
    def test_multidimensional_embedding_flattened(self, mock_openai):
        """Test that multidimensional embeddings are flattened."""
        mock_client = mock_openai.return_value
        # Embedding with more than 1 dimension (e.g., 2D array)
        mock_embedding = self._make_mock_embedding([[0.1, 0.2], [0.3, 0.4]])
        mock_response = self._make_mock_response([[0.1, 0.2], [0.3, 0.4]])

        # Override to return a 2D embedding for the first document
        mock_embedding_2d = MagicMock()
        mock_embedding_2d.embedding = [[0.1, 0.2], [0.3, 0.4]]
        mock_response_2d = MagicMock()
        mock_response_2d.data = [mock_embedding_2d]
        mock_client.embeddings.create.return_value = mock_response_2d

        result, skipped = self.embedder.transform(self.corpus[[0]])

        # Should be flattened to 1D
        self.assertEqual(result.X.shape[1], 4)
        assert_array_equal(result.X[0], [0.1, 0.2, 0.3, 0.4])

    @patch('openai.OpenAI')
    def test_corpus_extend_attributes(self, mock_openai):
        """Test that corpus is extended with correct feature attributes."""
        mock_client = mock_openai.return_value
        mock_response = self._make_mock_response([[0.1, 0.2]])
        mock_client.embeddings.create.return_value = mock_response

        result, skipped = self.embedder.transform(self.corpus[[0]])

        # Check that embedding features have correct attributes
        for var in result.domain.attributes:
            if var.name.startswith("Dim"):
                self.assertTrue(var.attributes.get("embedding-feature", False))
                self.assertTrue(var.attributes.get("hidden", False))

    @patch('openai.OpenAI')
    def test_transform_returns_none_skipped(self, mock_openai):
        """Test that skipped corpus is always None for OAIDocumentEmbedder."""
        mock_client = mock_openai.return_value
        mock_response = self._make_mock_response([[0.1, 0.2]])
        mock_client.embeddings.create.return_value = mock_response

        result, skipped = self.embedder.transform(self.corpus[[0]])

        self.assertIsNone(skipped)


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
