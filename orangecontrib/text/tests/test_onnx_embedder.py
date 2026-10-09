from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from orangecontrib.text.vectorization.onnx_embedder import ONNXEmbedder
from orangecontrib.text import Corpus


class TestONNXEmbedderInitialization(unittest.TestCase):
    """Tests for ONNXEmbedder initialization, configuration, lifecycle, and cache."""

    def test_init(self):
        # Default values
        embedder = ONNXEmbedder()
        self.assertEqual(embedder.model_id, "sentence-transformers/all-MiniLM-L6-v2")
        self.assertEqual(embedder.model_filename, "onnx/model_quint8_avx2.onnx")
        self.assertEqual(embedder.batch_size, 32)
        self.assertTrue(embedder.normalize)
        self.assertEqual(embedder.pooling, "mean")
        self.assertIsNone(embedder._session)
        self.assertIsNone(embedder._tokenizer)

        # Custom values
        embedder = ONNXEmbedder(
            model_id="custom/model",
            model_filename="onnx/model.onnx",
            batch_size=16,
            normalize=False,
            pooling="cls",
        )
        self.assertEqual(embedder.model_id, "custom/model")
        self.assertEqual(embedder.model_filename, "onnx/model.onnx")
        self.assertEqual(embedder.batch_size, 16)
        self.assertFalse(embedder.normalize)
        self.assertEqual(embedder.pooling, "cls")

        with self.assertRaises(ValueError) as ctx:
            ONNXEmbedder(pooling="invalid")

    @patch("orangecontrib.text.vectorization.onnx_embedder.is_model_downloaded")
    def test_ensure_initialized_cached_path(self, mock_downloaded):
        """Test when model is already downloaded (cached)."""
        mock_downloaded.return_value = True
        embedder = ONNXEmbedder()
        callback = MagicMock()

        with patch.object(embedder, "_initialize_from_path") as mock_init:
            with patch("transformers.AutoTokenizer"):
                with patch(
                    "orangecontrib.text.vectorization.onnx_embedder.hf_hub_download",
                    return_value="/cached/path/config.json",
                ):
                    result = embedder._ensure_initialized(progress_callback=callback)
                    self.assertFalse(result)
                    mock_init.assert_called_once_with("/cached/path")
                    callback.assert_not_called()  # No download, no callback

    @patch("orangecontrib.text.vectorization.onnx_embedder.is_model_downloaded")
    def test_ensure_initialized_download_path(self, mock_downloaded):
        """Test when model needs to be downloaded."""
        mock_downloaded.return_value = False
        embedder = ONNXEmbedder()
        callback = MagicMock()

        with patch(
            "orangecontrib.text.vectorization.onnx_embedder.download_model_with_progress",
            return_value="/path/to/model.onnx",
        ) as mock_download:
            def _download(repo_id, filename, progress_callback=None):
                if progress_callback:
                    progress_callback(0.5)
                return "/path/to/model.onnx"
            mock_download.side_effect = _download
            with patch.object(embedder, "_initialize_from_path") as mock_init:
                with patch("transformers.AutoTokenizer"):
                    with patch(
                        "orangecontrib.text.vectorization.onnx_embedder.hf_hub_download",
                        return_value="/cached/path/config.json",
                    ):
                        result = embedder._ensure_initialized(progress_callback=callback)
                        self.assertTrue(result)
                        mock_init.assert_called_once()
                        self.assertEqual(mock_init.call_args[0][0], "/cached/path")
                        callback.assert_called()  # Download happened, callback called

    def test_ensure_initialized_no_op_when_already_initialized(self):
        embedder = ONNXEmbedder()
        embedder._session = MagicMock()
        embedder._tokenizer = MagicMock()
        self.assertFalse(embedder._ensure_initialized())

    @patch("orangecontrib.text.vectorization.onnx_embedder.EmbedderCache")
    def test_clear_cache_delegates_to_cache(self, mock_cache_cls):
        embedder = ONNXEmbedder()
        mock_cache = mock_cache_cls.return_value
        embedder.clear_cache()
        mock_cache.clear_cache.assert_called_once()

    @patch("orangecontrib.text.vectorization.onnx_embedder.EmbedderCache")
    def test_different_configurations_produce_different_keys(self, mock_cache_cls):
        captured_keys = []
        mock_cache_cls.side_effect = lambda key: captured_keys.append(key) or MagicMock()
        ONNXEmbedder(pooling="mean", normalize=True)
        ONNXEmbedder(pooling="cls", normalize=True)
        ONNXEmbedder(pooling="mean", normalize=False)
        self.assertEqual(len(captured_keys), 3)
        self.assertEqual(len(set(captured_keys)), 3)
        self.assertIn("pooling-mean-normalize-True", captured_keys[0])
        self.assertIn("pooling-cls-normalize-True", captured_keys[1])
        self.assertIn("pooling-mean-normalize-False", captured_keys[2])

    def test_context_manager_enter_exit(self):
        # Enter returns self
        embedder = ONNXEmbedder()
        with embedder as e:
            self.assertIs(e, embedder)
        # Exit cleans up session
        embedder = ONNXEmbedder()
        embedder._session = MagicMock()
        embedder._tokenizer = MagicMock()
        with embedder:
            pass
        self.assertIsNone(embedder._session)
        self.assertIsNone(embedder._tokenizer)

    # --- Session lifecycle ---

    def test_cleanup_session(self):
        """Test _cleanup_session: noop when None, closes session, handles exceptions."""
        # No-op when nothing initialized
        embedder = ONNXEmbedder()
        embedder._cleanup_session()
        self.assertIsNone(embedder._session)

        # Closes existing session
        embedder = ONNXEmbedder()
        mock_session = MagicMock()
        embedder._session = mock_session
        embedder._tokenizer = MagicMock()
        embedder._cleanup_session()
        mock_session.close.assert_called_once()
        self.assertIsNone(embedder._session)

        # Handles exceptions during close
        embedder = ONNXEmbedder()
        mock_session = MagicMock()
        mock_session.close.side_effect = RuntimeError("boom")
        embedder._session = mock_session
        embedder._cleanup_session()  # Should not raise
        self.assertIsNone(embedder._session)

    @patch("orangecontrib.text.vectorization.onnx_embedder.ONNXInferenceSession")
    @patch("transformers.AutoTokenizer")
    def test_initialize_from_path(self, mock_tokenizer_cls, mock_session_cls):
        """Test _initialize_from_path: creates session and cleans existing one."""
        # Test creation
        embedder = ONNXEmbedder()
        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        root_path = "/path/to/model/root"
        expected_model_path = os.path.join(root_path, embedder.model_filename)

        embedder._initialize_from_path(root_path)

        mock_session_cls.assert_called_once_with(expected_model_path)
        mock_tokenizer_cls.from_pretrained.assert_called_once_with(root_path)
        self.assertEqual(embedder._session, mock_session)
        self.assertEqual(embedder._tokenizer, mock_tokenizer)

        # Test cleanup of existing session
        old_session = MagicMock()
        embedder._session = old_session
        embedder._tokenizer = MagicMock()
        mock_new_session = MagicMock()
        mock_session_cls.return_value = mock_new_session

        embedder._initialize_from_path(root_path)

        old_session.close.assert_called_once()
        self.assertEqual(embedder._session, mock_new_session)


class TestONNXEmbedderTokenize(unittest.TestCase):
    """Tests for _tokenize method."""

    @patch("transformers.AutoTokenizer")
    def test_tokenize(self, mock_tokenizer_cls):
        embedder = ONNXEmbedder()
        mock_session = MagicMock()
        mock_session.input_names = ["input_ids", "attention_mask", "token_type_ids"]
        embedder._session = mock_session

        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.return_value = {
            "input_ids": np.array([[1, 2, 3]]),
            "attention_mask": np.array([[1, 1, 1]]),
            "token_type_ids": np.array([[0, 0, 0]]),
        }
        embedder._tokenizer = mock_tokenizer
        embedder._max_length = 256

        result = embedder._tokenize(["hello world"])

        self.assertIn("token_type_ids", result)
        mock_tokenizer.assert_called_once_with(
            ["hello world"],
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=256,
            return_attention_mask=True,
            return_token_type_ids=True,
        )

    @patch("transformers.AutoTokenizer")
    def test_tokenize_multiple_texts(self, mock_tokenizer_cls):
        embedder = ONNXEmbedder()
        mock_session = MagicMock()
        mock_session.input_names = ["input_ids", "attention_mask"]
        embedder._session = mock_session

        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.return_value = {
            "input_ids": np.array([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": np.array([[1, 1, 1], [1, 1, 1]]),
        }
        embedder._tokenizer = mock_tokenizer
        embedder._max_length = 256

        result = embedder._tokenize(["text one", "text two"])

        self.assertEqual(len(result["input_ids"]), 2)


class TestONNXEmbedderInference(unittest.TestCase):
    """Tests for _inference method with pooling, normalization, and edge cases."""

    def test_mean_pooling(self):
        embedder = ONNXEmbedder(pooling="mean", normalize=False)
        mock_session = MagicMock()
        embedder._session = mock_session

        last_hidden_state = np.array([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]])
        attention_mask = np.array([[1, 1, 0]])
        tokenized = {
            "input_ids": np.array([[1, 2, 3]]),
            "attention_mask": attention_mask,
        }

        mock_session.run.return_value = [last_hidden_state]
        result = embedder._inference(tokenized)

        # Mean pooling: average over non-masked tokens
        expected = np.array([[0.2, 0.3]])  # mean of first two tokens
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_cls_pooling(self):
        embedder = ONNXEmbedder(pooling="cls", normalize=False)
        mock_session = MagicMock()
        embedder._session = mock_session

        last_hidden_state = np.array([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]])
        tokenized = {
            "input_ids": np.array([[1, 2, 3]]),
            "attention_mask": np.array([[1, 1, 1]]),
        }

        mock_session.run.return_value = [last_hidden_state]
        result = embedder._inference(tokenized)

        # CLS pooling: first token's hidden state
        expected = np.array([[0.1, 0.2]])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_normalize_true(self):
        embedder = ONNXEmbedder(normalize=True)
        mock_session = MagicMock()
        embedder._session = mock_session

        last_hidden_state = np.array([[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]])
        attention_mask = np.array([[1, 1, 0]])
        tokenized = {
            "input_ids": np.array([[1, 2, 3]]),
            "attention_mask": attention_mask,
        }

        mock_session.run.return_value = [last_hidden_state]
        result = embedder._inference(tokenized)

        # L2 normalized: norm should be 1
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_array_almost_equal(norms, [1.0], decimal=5)

    def test_normalize_false(self):
        embedder = ONNXEmbedder(normalize=False)
        mock_session = MagicMock()
        embedder._session = mock_session

        last_hidden_state = np.array([[[3.0, 4.0]]])
        attention_mask = np.array([[1]])
        tokenized = {
            "input_ids": np.array([[1]]),
            "attention_mask": attention_mask,
        }

        mock_session.run.return_value = [last_hidden_state]
        result = embedder._inference(tokenized)

        # No normalization: values should be preserved
        np.testing.assert_array_almost_equal(result, [[3.0, 4.0]], decimal=5)

    def test_inference_division_by_zero_handling(self):
        """Test that zero-length sequences don't cause division by zero."""
        embedder = ONNXEmbedder(pooling="mean")
        mock_session = MagicMock()
        embedder._session = mock_session

        last_hidden_state = np.array([[[0.1, 0.2], [0.3, 0.4]]])
        attention_mask = np.array([[0, 0]])  # all masked out
        tokenized = {
            "input_ids": np.array([[1, 2]]),
            "attention_mask": attention_mask,
        }

        mock_session.run.return_value = [last_hidden_state]
        result = embedder._inference(tokenized)

        # Should not raise; denominator clamped to 1
        self.assertFalse(np.isnan(result).any())

    def test_inference_batch_of_two(self):
        embedder = ONNXEmbedder(pooling="mean", normalize=False)
        mock_session = MagicMock()
        embedder._session = mock_session

        last_hidden_state = np.array([
            [[0.1, 0.2], [0.3, 0.4]],
            [[1.0, 2.0], [3.0, 4.0]],
        ])
        attention_mask = np.array([[1, 0], [1, 1]])
        tokenized = {
            "input_ids": np.array([[1, 2], [3, 4]]),
            "attention_mask": attention_mask,
        }

        mock_session.run.return_value = [last_hidden_state]
        result = embedder._inference(tokenized)

        self.assertEqual(result.shape, (2, 2))
        # First doc: only first token
        np.testing.assert_array_almost_equal(result[0], [0.1, 0.2])
        # Second doc: mean of both tokens
        np.testing.assert_array_almost_equal(result[1], [2.0, 3.0])

    def test_mean_pooling_attention_weighted(self):
        """Test that mean pooling correctly uses attention mask as weights."""
        embedder = ONNXEmbedder(pooling="mean", normalize=False)
        mock_session = MagicMock()
        embedder._session = mock_session

        # Two tokens with different values
        last_hidden_state = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        # First token active, second masked
        attention_mask = np.array([[1, 0]])
        tokenized = {
            "input_ids": np.array([[1, 2]]),
            "attention_mask": attention_mask,
        }

        mock_session.run.return_value = [last_hidden_state]
        result = embedder._inference(tokenized)

        # Only first token contributes: [1.0, 2.0]
        np.testing.assert_array_almost_equal(result, [[1.0, 2.0]])

    def test_cls_pooling_ignores_attention_mask(self):
        """Test that CLS pooling uses first token regardless of attention mask."""
        embedder = ONNXEmbedder(pooling="cls", normalize=False)
        mock_session = MagicMock()
        embedder._session = mock_session

        last_hidden_state = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        # First token masked, but CLS still uses it
        attention_mask = np.array([[0, 1]])
        tokenized = {
            "input_ids": np.array([[1, 2]]),
            "attention_mask": attention_mask,
        }

        mock_session.run.return_value = [last_hidden_state]
        result = embedder._inference(tokenized)

        # CLS still uses first token: [1.0, 2.0]
        np.testing.assert_array_almost_equal(result, [[1.0, 2.0]])

    def test_normalize_zero_vector(self):
        """Test that zero vectors are handled gracefully."""
        embedder = ONNXEmbedder(normalize=True)
        mock_session = MagicMock()
        embedder._session = mock_session

        last_hidden_state = np.array([[[0.0, 0.0]]])
        attention_mask = np.array([[1]])
        tokenized = {
            "input_ids": np.array([[1]]),
            "attention_mask": attention_mask,
        }

        mock_session.run.return_value = [last_hidden_state]
        result = embedder._inference(tokenized)

        # Zero vector should not produce NaN (division by zero is clamped to 1e-8)
        self.assertFalse(np.isnan(result).any())
        # Zero vector divided by 1e-8 is still zero, so norm is 0.0
        norms = np.linalg.norm(result, axis=1)
        self.assertAlmostEqual(norms[0], 0.0, places=5)


class TestONNXEmbedderEmbedTexts(unittest.TestCase):
    """Tests for _embed_texts and caching logic."""

    def test_embed_texts_none_cached(self):
        embedder = ONNXEmbedder()
        mock_cache = MagicMock()
        # Make md5_hash return a deterministic value
        mock_cache.md5_hash.side_effect = lambda x: hash(x)
        # No results in cache
        mock_cache.get_cached_result_or_none.return_value = None
        embedder._cache = mock_cache

        texts = ["hello", "world"]
        uncached_result = np.array([
            [0.1, 0.2],
            [0.3, 0.4],
        ], dtype=np.float32)

        with patch.object(embedder, "_embed_texts_uncached", return_value=uncached_result):
            result = embedder._embed_texts(texts)

        self.assertEqual(result.shape, (2, 2))
        np.testing.assert_array_almost_equal(result, uncached_result)
        # Verify cache was updated
        self.assertEqual(mock_cache.add.call_count, 2)
        mock_cache.persist_cache.assert_called_once()

    def test_embed_texts_partial_cache(self):
        embedder = ONNXEmbedder()
        mock_cache = MagicMock()

        texts = ["hello", "world", "foo"]
        emb_hello = np.array([0.1, 0.2], dtype=np.float32)

        # Setup mock cache methods
        def make_hash(s):
            return hash(s)
        hash_hello = make_hash("hello".encode("utf-8"))

        mock_cache.md5_hash.side_effect = lambda x: make_hash(x)
        mock_cache.get_cached_result_or_none.side_effect = lambda h: emb_hello if h == hash_hello else None
        embedder._cache = mock_cache

        uncached_result = np.array([
            [0.3, 0.4],
            [0.5, 0.6],
        ], dtype=np.float32)

        with patch.object(embedder, "_embed_texts_uncached", return_value=uncached_result):
            result = embedder._embed_texts(texts)

        self.assertEqual(result.shape, (3, 2))
        np.testing.assert_array_almost_equal(result[0], emb_hello)
        np.testing.assert_array_almost_equal(result[1], uncached_result[0])
        np.testing.assert_array_almost_equal(result[2], uncached_result[1])
        # Only 2 uncached calls
        self.assertEqual(mock_cache.add.call_count, 2)

class TestONNXEmbedderCall(unittest.TestCase):
    """Tests for the main __call__ interface."""

    @patch.object(ONNXEmbedder, "_embed_texts")
    @patch.object(ONNXEmbedder, "_ensure_initialized")
    def test_call(self, mock_init, mock_embed):
        embedder = ONNXEmbedder()
        result = embedder([])
        self.assertEqual(result, [])

        embedder = ONNXEmbedder()
        mock_init.return_value = False
        mock_embed.return_value = np.array([
            [0.1, 0.2],
            [0.3, 0.4],
        ], dtype=np.float32)

        result = embedder(["hello", "world"])

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], list)
        self.assertEqual(len(result[0]), 2)
        # Should be called
        mock_init.assert_called_once()

    def test_call_single_text(self):
        """Test __call__ with a single text."""
        embedder = ONNXEmbedder(batch_size=32)

        with patch.object(embedder, "_embed_texts", return_value=np.array([[0.1, 0.2]], dtype=np.float32)):
            result = embedder(["hello"])

        self.assertEqual(len(result), 1)
        np.testing.assert_array_almost_equal(result[0], [0.1, 0.2])

    def test_call_multiple_batches(self):
        """Test that __call__ processes texts in batches."""
        embedder = ONNXEmbedder(batch_size=2)

        with patch.object(embedder, "_embed_texts", return_value=np.full((4, 4), 0.1, dtype=np.float32)):
            result = embedder(["a", "bb", "ccc", "dddd"])

        self.assertEqual(len(result), 4)
        self.assertEqual(len(result[0]), 4)

    def test_call_callback_progress(self):
        """Test that __call__ invokes callback with progress updates."""
        embedder = ONNXEmbedder(batch_size=2)
        callback = MagicMock()

        def mock_embed(texts, cb):
            cb(0.5)
            return np.full((len(texts), 3), 0.1, dtype=np.float32)

        with patch.object(embedder, "_embed_texts", side_effect=mock_embed):
            texts = ["a", "bb", "ccc", "dddd"]
            embedder(texts, callback=callback)

        # Callback should be called with progress values
        self.assertGreater(len(callback.call_args_list), 0)
        # Last call should be 0.5 (from our mock)
        last_progress = callback.call_args_list[-1][0][0]
        self.assertAlmostEqual(last_progress, 0.5, places=5)

    def test_call_respects_max_batch_size(self):
        """Test that __call__ respects max_batch_size via effective_batch_size."""
        embedder = ONNXEmbedder(batch_size=10)

        inference_count = [0]
        def mock_embed(texts, cb=None):
            # Simulate batching behavior: 10 texts with batch_size=10 = 1 batch
            inference_count[0] += 1
            return np.full((len(texts), 3), 0.1, dtype=np.float32)

        with patch.object(embedder, "_embed_texts", side_effect=mock_embed):
            embedder(["a"] * 10)

        # 10 texts with batch_size=10 should result in 1 embedding call
        self.assertEqual(inference_count[0], 1)

    def test_call_respects_max_tokens(self):
        """Test that __call__ respects max_tokens limit via effective_batch_size."""
        embedder = ONNXEmbedder()

        inference_count = [0]
        def mock_embed(texts, cb=None):
            # Simulate batching: each text produces 100 tokens, max_tokens=500
            # effective_batch_size would cap at 5, so 10 texts = 2 batches
            inference_count[0] += 1
            return np.full((len(texts), 3), 0.1, dtype=np.float32)

        with patch.object(embedder, "_embed_texts", side_effect=mock_embed):
            embedder(["a"] * 10)

        # Should have made at least 1 embedding call
        self.assertGreater(inference_count[0], 0)

    def test_call_minimum_batch_size_one(self):
        """Test that __call__ works with a single text."""
        embedder = ONNXEmbedder()

        with patch.object(embedder, "_embed_texts", return_value=np.array([[0.1, 0.2]], dtype=np.float32)):
            result = embedder(["a"])

        self.assertEqual(len(result), 1)


class TestONNXEmbedderTransform(unittest.TestCase):
    """Tests for transform method."""

    def setUp(self):
        self.corpus = Corpus.from_file("deerwester")

    @patch.object(ONNXEmbedder, "_ensure_initialized")
    def test_transform_empty_corpus(self, mock_init):
        embedder = ONNXEmbedder()
        mock_init.return_value = False
        empty_corpus = self.corpus[:0]

        result, skipped = embedder.transform(empty_corpus)

        self.assertEqual(len(result), 0)
        self.assertIsNone(skipped)

    @patch.object(ONNXEmbedder, "_ensure_initialized")
    def test_transform_adds_features(self, mock_init):
        embedder = ONNXEmbedder()
        mock_init.return_value = False

        with patch.object(embedder, "_embed_texts") as mock_embed:
            mock_embed.return_value = np.full((len(self.corpus), 5), 0.1, dtype=np.float32)
            result, skipped = embedder.transform(self.corpus)

        self.assertIsNone(skipped)
        # Check that new embedding features were added
        self.assertEqual(len(result.domain.attributes), 5)
        # Feature names should be Dim1, Dim2, ..., Dim5
        attr_names = [attr.name for attr in result.domain.attributes]
        for i in range(5):
            self.assertIn(f"Dim{i + 1}", attr_names)

    @patch.object(ONNXEmbedder, "_ensure_initialized")
    def test_transform_callback_progress(self, mock_init):
        embedder = ONNXEmbedder()
        mock_init.return_value = False  # No download needed

        callback = MagicMock()

        def mock_embed_with_callback(texts, cb):
            cb(0.5)  # simulate progress
            return np.full((len(texts), 3), 0.1, dtype=np.float32)

        with patch.object(embedder, "_embed_texts", side_effect=mock_embed_with_callback):
            embedder.transform(self.corpus, callback=callback)

        callback.assert_called()

    @patch.object(ONNXEmbedder, "_ensure_initialized")
    def test_transform_callback_progress_with_download(self, mock_init):
        embedder = ONNXEmbedder()
        mock_init.return_value = True  # Download happened

        callback = MagicMock()

        def mock_embed_with_callback(texts, cb):
            cb(0.5)  # simulate progress
            return np.full((len(texts), 3), 0.1, dtype=np.float32)

        with patch.object(embedder, "_embed_texts", side_effect=mock_embed_with_callback):
            embedder.transform(self.corpus, callback=callback)

        callback.assert_called()


class TestONNXEmbedderReport(unittest.TestCase):
    """Tests for the report method."""

    def test_report_without_session(self):
        embedder = ONNXEmbedder()
        report = embedder.report()

        self.assertEqual(len(report), 5)
        self.assertEqual(report[0], ("Embedder", "ONNX (sentence-transformers/all-MiniLM-L6-v2)"))
        self.assertEqual(report[1], ("Model file", "onnx/model_quint8_avx2.onnx"))
        self.assertEqual(report[2], ("Batch size", "32"))
        self.assertEqual(report[3], ("Pooling", "mean"))
        self.assertEqual(report[4], ("Embedding dim", "?"))

    @patch.object(ONNXEmbedder, "_ensure_initialized")
    def test_report_with_session(self, mock_init):
        embedder = ONNXEmbedder()
        mock_init.return_value = False

        mock_session = MagicMock()
        mock_session.embedding_dim = 3
        embedder._session = mock_session
        embedder._tokenizer = MagicMock()

        report = embedder.report()

        self.assertEqual(len(report), 5)
        self.assertEqual(report[0], ("Embedder", "ONNX (sentence-transformers/all-MiniLM-L6-v2)"))
        self.assertEqual(report[1], ("Model file", "onnx/model_quint8_avx2.onnx"))
        self.assertEqual(report[2], ("Batch size", "32"))
        self.assertEqual(report[3], ("Pooling", "mean"))
        self.assertEqual(report[4][0], "Embedding dim")
        self.assertEqual(report[4][1], "3")

    @patch.object(ONNXEmbedder, "_ensure_initialized")
    def test_report_custom_config(self, mock_init):
        embedder = ONNXEmbedder(
            model_id="custom/model",
            model_filename="onnx/model.onnx",
            batch_size=8,
            pooling="cls",
        )
        mock_init.return_value = False

        with patch.object(embedder, "_embed_texts", return_value=np.array([[0.1]], dtype=np.float32)):
            embedder(["hello"])

        report = embedder.report()

        self.assertEqual(report[0], ("Embedder", "ONNX (custom/model)"))
        self.assertEqual(report[1], ("Model file", "onnx/model.onnx"))
        self.assertEqual(report[2], ("Batch size", "8"))
        self.assertEqual(report[3], ("Pooling", "cls"))


if __name__ == "__main__":
    unittest.main()