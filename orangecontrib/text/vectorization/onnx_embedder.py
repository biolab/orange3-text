"""This module contains a local text embedder using ONNX runtime.

Inference is executed in a subprocess for isolation from onnxruntime's
native library conflicts (see
https://www.riverbankcomputing.com/pipermail/pyqt/2025-November/046378.html).

It supports multiple ONNX models including sentence-transformers/all-MiniLM-L6-v2
and IBM Granite embedding models
"""
from __future__ import annotations
import logging
import os
import json
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from huggingface_hub import hf_hub_download

from Orange.misc.utils.embedder_utils import EmbedderCache
from Orange.util import dummy_callback

from orangecontrib.text import Corpus
from orangecontrib.text.misc import download_model_with_progress, is_model_downloaded, url_to_safe_filename
from orangecontrib.text.vectorization.base import BaseVectorizer
from orangecontrib.text.vectorization.onnx_embedder_runner import ONNXInferenceSession

# HuggingFace model repository for the default ONNX model
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
ONNX_MODEL_FILE = "onnx/model_quint8_avx2.onnx"
BATCH_SIZE = 32
# Maximum tokens per inference batch to limit RAM usage.
# RAM scales with batch_size × sequence_length; for models with long context
# windows (e.g. 32768 tokens in granite-embedding-97m-multilingual-r2), a large
# batch size would consume excessive memory.
MAX_TOKENS_PER_BATCH = 32768

logger = logging.getLogger(__name__)


class ONNXEmbedder(BaseVectorizer):
    """Local text embedder using an ONNX model.

    Uses any ONNX-converted sentence embedding model downloaded from
    HuggingFace Hub. Automatically adapts to the model's input signature
    (e.g. some models use ``token_type_ids``, others do not) and applies
    attention-mask-weighted mean pooling or CLS token pooling over the
    sequence dimension.

    The model and tokenizer are downloaded on first use and cached in
    the HuggingFace cache directory (typically ``~/.cache/huggingface``).

    Attributes
    ----------
    model_id : str
        HuggingFace repository id containing the ONNX model and tokenizer.
    model_filename : str
        Path to the ONNX model file within the repository (relative to
        the repo root). Defaults to ``"onnx/model_quint8_avx2.onnx"``. For models
        with multiple quantized variants (e.g. ``onnx/model.onnx``),
        specify the desired file to select a specific quantization/optimization.
    batch_size : int
        Number of documents processed in a single inference batch.
    normalize : bool
        Whether to L2-normalize embeddings after pooling (default ``True``).
        Most embedding models (including IBM Granite) produce non-normalized
        embeddings that should be normalized for cosine similarity.
        Some models like MiniLM may already be normalized; enabling this
        is safe for both cases.
    pooling : Literal["mean", "cls"]
        Pooling strategy for converting token embeddings into document
        embeddings. ``"mean"`` computes an attention-mask-weighted mean
        over all token hidden states (default). ``"cls"`` uses the
        hidden state of the first token (``[CLS]``) as the document
        embedding. Use ``"cls"`` for models that were trained with CLS
        pooling (e.g. some IBM Granite embedding models).
    """

    name = "ONNX Embedder"
    model_id = MODEL_ID
    model_filename = ONNX_MODEL_FILE
    max_tokens_per_batch = MAX_TOKENS_PER_BATCH


    def __init__(
        self,
        model_id: str = MODEL_ID,
        model_filename: str = ONNX_MODEL_FILE,
        batch_size: int = BATCH_SIZE,
        normalize: bool = True,
        pooling: Literal["mean", "cls"] = "mean",
    ) -> None:
        self.model_id = model_id
        self.model_filename = model_filename
        self.batch_size = batch_size
        self.normalize = normalize
        self.pooling = pooling
        if self.pooling not in ("mean", "cls"):
            raise ValueError(
                f"Invalid pooling strategy: {self.pooling!r}. "
                f"Must be 'mean' or 'cls'."
            )
        self._session: ONNXInferenceSession | None = None
        self._tokenizer = None
        self._max_length: int | None = None
        self._cache = EmbedderCache(
            url_to_safe_filename(
                f"onnx-{self.model_id}-{self.model_filename}-pooling-{self.pooling}-normalize-{self.normalize}"
            )
        )

    def _ensure_initialized(self, progress_callback: Callable | None = None) -> bool:
        """Lazy-initialize the ONNX session and tokenizer.

        Ensures all files are downloaded from HuggingFace Hub before
        initialization.

        Parameters
        ----------
        progress_callback : callable, optional
            A callable that receives a float in [0, 1] representing
            download progress.

        Returns
        -------
        bool
            ``True`` if a download was performed, ``False`` if the model
            was already cached and initialized.
        """
        if self._session is not None and self._tokenizer is not None:
            return False

        from huggingface_hub.errors import EntryNotFoundError
        from transformers import AutoTokenizer  # noqa: local import

        # Download sentence_bert_config.json if it exists for this model
        try:
            hf_hub_download(
                repo_id=self.model_id,
                filename="sentence_bert_config.json",
            )
            logger.debug("Downloaded sentence_bert_config.json for %s", self.model_id)
        except EntryNotFoundError:
            logger.debug("No sentence_bert_config.json found for %s", self.model_id)

        # Download all tokenizer files by calling AutoTokenizer.from_pretrained.
        # This populates the HuggingFace cache with all tokenizer files
        # (vocab, merges, config, etc.).
        AutoTokenizer.from_pretrained(self.model_id)

        # Resolve the local filesystem path for tokenizer files.
        # All tokenizer files are in the same HuggingFace repo directory.
        # We use hf_hub_download with local_files_only=True to get the cached path.
        tokenizer_file_path = hf_hub_download(
            repo_id=self.model_id,
            filename="config.json",
            local_files_only=True,
        )
        
        root_repo = os.path.dirname(tokenizer_file_path)
        logger.debug("Tokenizer files at %s for %s", root_repo, self.model_id)

        # Check if the ONNX model is already downloaded in the HuggingFace cache.
        if is_model_downloaded(self.model_id, self.model_filename):
            self._initialize_from_path(root_repo)
            return False

        # Download the ONNX model
        download_model_with_progress(
            repo_id=self.model_id,
            filename=self.model_filename,
            progress_callback=progress_callback,
        )
        self._initialize_from_path(root_repo)
        return True

    def _initialize_from_path(self, model_root: str) -> None:
        """Load ONNX session and tokenizer from given model root directory.

        The ONNX session is created in a subprocess for isolation.
        Any previously existing session is cleaned up first.

        Parameters
        ----------
        model_root : str
            Absolute path to the root of the model repository (HF-like directory
            containing both the ONNX model file and tokenizer files).
        """
        from transformers import AutoTokenizer  # noqa: local import

        # Clean up any existing session
        self._cleanup_session()
        self._session = ONNXInferenceSession(
            os.path.join(model_root, self.model_filename)
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_root)
        self._max_length = self._get_effective_max_length()

    def _get_effective_max_length(self) -> int:
        """Determine the effective max_length the same way sentence-transformers does.

        sentence-transformers loads ``sentence_bert_config.json`` which may contain
        a ``max_seq_length`` field. This value is used to set the tokenizer's
        ``model_max_length`` before tokenization. If that file doesn't exist or
        doesn't contain ``max_seq_length``, the tokenizer's own ``model_max_length``
        is used as a fallback.

        Returns
        -------
        int
            The effective maximum sequence length for tokenization.
        """
        # Try to load sentence_bert_config.json (already downloaded in _ensure_initialized)
        try:
            config_path = hf_hub_download(
                repo_id=self.model_id,
                filename="sentence_bert_config.json",
                local_files_only=True,
            )
            with open(config_path, "r") as f:
                config = json.load(f)
            max_seq_length = config.get("max_seq_length")
            if max_seq_length is not None:
                logger.debug(
                    "Got max_seq_length=%d from sentence_bert_config.json for %s",
                    max_seq_length,
                    self.model_id,
                )
                return max_seq_length
        except Exception:
            pass

        # Fall back to tokenizer's model_max_length
        fallback = self._tokenizer.model_max_length
        logger.debug(
            "Using tokenizer.model_max_length=%d for %s", fallback, self.model_id
        )
        return fallback

    def _cleanup_session(self) -> None:
        """Close and clean up the existing ONNXInferenceSession if present."""
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                logger.exception("Error cleaning up ONNX session")
            self._session = None
            self._tokenizer = None
            self._max_length = None

    def _tokenize(
        self, texts: list[str]
    ) -> dict[str, np.ndarray]:
        """Tokenize a list of texts.

        Parameters
        ----------
        texts : list of str
            Texts to tokenize.

        Returns
        -------
        dict of str to ndarray
            Dictionary of tokenized tensors (e.g. ``input_ids``,
            ``attention_mask``, and optionally ``token_type_ids``) as
            ndarrays of shape ``(len(texts), sequence_length)``.
        """
        assert self._session is not None
        assert self._max_length is not None
        input_names = self._session.input_names
        inputs = self._tokenizer(
            texts,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_attention_mask=True,
            return_token_type_ids="token_type_ids" in input_names,
        )
        return dict(inputs)

    def _inference(self, tokenized: dict[str, np.ndarray]) -> np.ndarray:
        """Run ONNX inference and apply pooling with optional L2 normalization.

        Inference is executed in a subprocess for isolation.

        Parameters
        ----------
        tokenized : dict
            Tokenized inputs from ``_tokenize``.

        Returns
        -------
        ndarray
            Pooled (and optionally normalized) embeddings of shape
            ``(batch_size, embedding_dim)``.
        """
        assert self._session is not None
        # Run inference in the subprocess
        probe_outputs = self._session.run(tokenized)
        last_hidden_state = probe_outputs[0]

        if self.pooling == "cls":
            # CLS pooling: use the first token's hidden state ([CLS] token)
            embeddings = last_hidden_state[:, 0, :]
        else:
            # Mean pooling: weight by attention mask and average over sequence
            attention_mask = tokenized["attention_mask"]
            denominator = attention_mask.sum(axis=1, keepdims=True)
            denominator = np.maximum(denominator, 1)  # avoid division by zero
            embeddings = (
                last_hidden_state * attention_mask[:, :, np.newaxis]
            ).sum(axis=1) / denominator

        # L2 normalize if requested (important for cosine similarity)
        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)  # avoid division by zero
            embeddings = embeddings / norms

        return embeddings.astype(np.float32)

    def _embed_texts(
        self, texts: list[str], callback: Callable[[float], None] = dummy_callback
    ) -> np.ndarray:
        """Embed a list of texts into dense vectors.

        Parameters
        ----------
        texts : list of str
            Texts to embed.
        callback : callable
            Progress callback receiving a float in [0, 1].

        Returns
        -------
        ndarray
            Embedding matrix of shape ``(len(texts), embedding_dim)``.
        """
        total = len(texts)
        assert total
        # Check cache for already-computed embeddings
        cache = self._cache
        results = [None] * total
        query_texts: list[str] = []
        query_indices: list[int] = []

        for i, txt in enumerate(texts):
            hashed = cache.md5_hash(txt.encode("utf-8"))
            r = cache.get_cached_result_or_none(hashed)
            if r is not None:
                results[i] = r
            else:
                query_texts.append(txt)
                query_indices.append(i)

        # Embed uncached texts in batches
        if query_texts:
            query_results = self._embed_texts_uncached(query_texts, callback)
            # Store results in cache and results list
            for idx, emb in zip(query_indices, query_results):
                cache.add(cache.md5_hash(texts[idx].encode("utf-8")), emb)
                results[idx] = emb
            cache.persist_cache()

        return np.array(results, dtype=np.float32)

    def _embed_texts_uncached(
        self, texts: list[str], callback: Callable[[float], None] = dummy_callback
    ) -> np.ndarray:
        """Embed a list of texts without checking the cache.

        Parameters
        ----------
        texts : list of str
            Texts to embed.
        callback : callable
            Progress callback receiving a float in [0, 1].

        Returns
        -------
        ndarray
            Embedding matrix of shape ``(len(texts), embedding_dim)``.
        """
        assert len(texts)
        # Sort texts by length to minimize padding waste in batches
        seq_len = np.array([len(s) for s in texts], dtype=int)
        sort_indices = np.argsort(seq_len)
        unsort_indices = np.argsort(sort_indices)
        texts_sorted = [texts[i] for i in sort_indices]
        embeddings = self._embed_batched(texts_sorted, callback)
        # Restore order
        return embeddings[unsort_indices]

    def _embed_batched(
            self, texts: list[str], callback: Callable[[float], None]
    )-> np.ndarray:
        all_embeddings = []
        start = 0
        total = len(texts)
        callback(0)
        while start < total:
            # Infer effective batch size
            batch_size = self._effective_batch_size(texts[start:], self.batch_size, self.max_tokens_per_batch)
            batch = self._tokenize(texts[start: start + batch_size])
            batch_embeddings = self._inference(batch)
            all_embeddings.extend(batch_embeddings)
            start += batch_size
            callback(start / total)
        return np.vstack(all_embeddings).astype(np.float32)

    def _effective_batch_size(
        self, texts: list[str], max_batch_size: int, max_tokens: int
        ) -> int:
        """Compute the actual number of texts that can fit in one inference batch.

        Determines the largest batch size that satisfies both the user-configured
        ``max_batch_size`` and the memory limit ``max_tokens``. The token budget
        is computed conservatively: all texts in the batch are assumed to be
        padded to the length of the longest text, so the total token count is
        ``max_sequence_length * batch_size``.

        Parameters
        ----------
        texts : list of str
            The remaining texts to consider for batching (typically a suffix of
            the full text list, starting from the current position).
        max_batch_size : int
            The user-configured maximum number of texts per batch.
        max_tokens : int
            The maximum number of tokens allowed per batch to limit RAM usage.

        Returns
        -------
        int
            The effective batch size, at least 1 and at most ``min(len(texts), max_batch_size)``.
        """
        assert max_tokens > 0 and max_batch_size > 0 and len(texts)
        first = self._tokenize([texts[0]])
        max_seq_length = first["input_ids"].shape[1]
        batch_size = 1
        while batch_size < max_batch_size and batch_size < len(texts):
            probe = self._tokenize([texts[batch_size]])["input_ids"]
            max_seq_length = max(probe.shape[1], max_seq_length)
            if max_seq_length * (batch_size + 1) > max_tokens:
                break
            batch_size += 1
        assert batch_size <= max_batch_size
        return batch_size

    def __call__(
        self,
        texts: list[str],
        callback: Callable[[float], None] = dummy_callback,
    ) -> list[list[float]]:
        """Compute embeddings for given texts.

        Parameters
        ----------
        texts : list of str
            Texts to embed.
        callback : callable[[float], None]
            Progress callback.

        Returns
        -------
        list of list of float
            Embeddings for each text. Empty list if no texts.
        """
        if len(texts) == 0:
            return []
        self._ensure_initialized()
        embs = self._embed_texts(texts, callback)
        return embs.tolist()

    def _transform(
        self,
        corpus: Corpus,
        source_dict: object,
        callback: Callable[[float], None] = dummy_callback,
    ) -> tuple[Corpus, Corpus | None]:
        """Embed documents in the corpus and add them as features.

        Parameters
        ----------
        corpus : Corpus
            Corpus whose text features will be embedded.
        source_dict : object
            Unused; kept for interface compatibility.
        callback : callable[[float], None]
            Progress callback.

        Returns
        -------
        tuple of (Corpus, None)
            Corpus with embedding features appended, and None for skipped.
        """
        downloaded = self._ensure_initialized(
            progress_callback=lambda p: callback(p * 0.5),
        )

        texts = list(corpus.documents)
        if len(texts) == 0:
            return corpus, None

        if downloaded:
            embed_callback = lambda p: callback(0.5 + 0.5 * p)  # noqa: E731
        else:
            embed_callback = callback  # already initialized

        embs = self._embed_texts(texts, embed_callback)

        dim = embs.shape[1]
        new_corpus = corpus.extend_attributes(
            embs,
            feature_names=[f"Dim{i + 1}" for i in range(dim)],
            var_attrs={
                "embedding-feature": True,
                "hidden": True,
            },
        )
        return new_corpus, None

    def report(self) -> tuple[tuple[str, str], ...]:
        """Report current configuration."""
        return (
            ("Embedder", f"ONNX ({self.model_id})"),
            ("Model file", self.model_filename),
            ("Batch size", str(self.batch_size)),
            ("Pooling", self.pooling),
            ("Embedding dim", str(self._session.embedding_dim) if self._session else "?"),
        )

    def clear_cache(self) -> None:
        """Clear the cached embeddings."""
        self._cache.clear_cache()

    def __enter__(self) -> "ONNXEmbedder":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Exit context manager, cleaning up the subprocess pool."""
        self._cleanup_session()
