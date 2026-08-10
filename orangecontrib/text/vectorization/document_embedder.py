"""This module contains classes used for embedding documents
into a vector space.
"""
import base64
import json
import sys
import warnings
import zlib
import re
from typing import Any, Optional, Tuple, Callable

import numpy as np
import openai

from Orange.misc.server_embedder import ServerEmbedderCommunicator
from Orange.misc.utils.embedder_utils import EmbedderCache
from Orange.util import dummy_callback

from orangecontrib.text import Corpus
from orangecontrib.text.vectorization.base import BaseVectorizer

AGGREGATORS = ["mean", "sum", "max", "min"]
AGGREGATORS_ITEMS = ['Mean', 'Sum', 'Max', 'Min']
# fmt: off
LANGUAGES = [
    'en', 'sl', 'de', 'ar', 'az', 'bn', 'zh', 'da', 'nl', 'fi', 'fr', 'el',
    'he', 'hi', 'hu', 'id', 'it', 'ja', 'kk', 'ko', 'ne', 'no', 'nn', 'pl',
    'pt', 'ro', 'ru', 'es', 'sv', 'tg', 'tr'
]
# fmt: on


class DocumentEmbedder(BaseVectorizer):
    """This class is used for obtaining dense embeddings of documents in
    corpus using fastText pretrained models from:
    E. Grave, P. Bojanowski, P. Gupta, A. Joulin, T. Mikolov,
    Learning Word Vectors for 157 Languages.
    Proceedings of the International Conference on Language Resources and
    Evaluation, 2018.

    Embedding is performed on server so the internet connection is a
    prerequisite for using the class.

    Attributes
    ----------
    language : str
        ISO 639-1 (two-letter) code of desired language.
    aggregator : str
        Aggregator which creates document embedding (single
        vector) from word embeddings (multiple vectors).
        Allowed values are Mean, Sum, Max, Min.
    """

    def __init__(
        self, language: Optional[str] = None, aggregator: str = "Mean"
    ) -> None:
        assert (
            language is None or language in LANGUAGES
        ), f"Language should be one of: {LANGUAGES}"
        assert aggregator in AGGREGATORS_ITEMS, f"Aggregator should be one of: {AGGREGATORS_ITEMS}"
        self.aggregator = aggregator
        self.language = language

    def _transform(
        self, corpus: Corpus, _, callback=dummy_callback
    ) -> Tuple[Corpus, Corpus]:
        """Adds matrix of document embeddings to a corpus.

        Parameters
        ----------
        corpus : Corpus or list of lists
            Corpus on which transform is performed.

        Returns
        -------
        Embeddings
            Corpus (original or a copy) with new features added.
        Skipped documents
            Corpus of documents that were not embedded
        """
        language = self.language if self.language else corpus.language
        if language not in LANGUAGES:
            raise ValueError(
                "The FastText embedding does not support the Corpus's language."
            )
        embedder = _ServerEmbedder(
            AGGREGATORS[AGGREGATORS_ITEMS.index(self.aggregator)],
            model_name="fasttext-" + language,
            max_parallel_requests=100,
            server_url="https://api.garaza.io",
            embedder_type="text",
        )
        embs = embedder.embedd_data(
            list(corpus.ngrams) if isinstance(corpus, Corpus) else corpus,
            callback=callback,
        )

        if isinstance(corpus, list):
            return embs

        dim = None
        for emb in embs:  # find embedding dimension
            if emb is not None:
                dim = len(emb)
                break
        # Check if some documents in corpus in weren't embedded
        # for some reason. This is a very rare case.
        skipped_documents = [emb is None for emb in embs]
        embedded_documents = np.logical_not(skipped_documents)

        new_corpus = None
        if np.any(embedded_documents):
            # if at least one embedding is not None, extend attributes
            new_corpus = corpus[embedded_documents]
            new_corpus = new_corpus.extend_attributes(
                np.array(
                    [e for e, ns in zip(embs, embedded_documents) if ns],
                    dtype=float,
                ),
                ["Dim{}".format(i + 1) for i in range(dim)],
                var_attrs={
                    "embedding-feature": True,
                    "hidden": True,
                },
            )

        skipped_corpus = None
        if np.any(skipped_documents):
            skipped_corpus = corpus[skipped_documents].copy()
            skipped_corpus.name = "Skipped documents"
            warnings.warn(
                "Some documents were not embedded for unknown reason. Those "
                "documents are skipped.",
                RuntimeWarning,
            )

        return new_corpus, skipped_corpus

    @staticmethod
    def clear_cache(language):
        """Clears embedder cache"""
        EmbedderCache(f"fasttext-{language}").clear_cache()


class _ServerEmbedder(ServerEmbedderCommunicator):
    def __init__(self, aggregator: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.content_type = 'application/json'
        self.aggregator = aggregator

    async def _encode_data_instance(self, data_instance: Any) -> Optional[bytes]:
        data_string = json.dumps(list(data_instance))
        data = base64.b64encode(zlib.compress(
            data_string.encode('utf-8', 'replace'),
            level=-1)).decode('utf-8', 'replace')
        if sys.getsizeof(data) > 500000:
            # Document in corpus is too large. Size limit is 500 KB
            # (after compression). - document skipped
            return None

        data_dict = {
            "data": data,
            "aggregator": self.aggregator
        }

        json_string = json.dumps(data_dict)
        return json_string.encode('utf-8', 'replace')


def url_to_safe_filename(url: str) -> str:
    """
    Convert an URL into a safe, cross-platform single filesystem filename.
    Args:
        url: The input URL string

    Returns:
        A sanitized, valid single-file filename.
    """
    if not url or not url.strip():
        raise ValueError("'url' cannot be empty")

    # Replace all Windows/POSIX invalid characters and control chars with underscore
    safe = re.sub(r'[<>:"/\\|?*]', '_', url)
    safe = re.sub(r'[\x00-\x1f]', '_', safe)
    # Handle Windows reserved names (CON, PRN, etc.)
    safe = re.sub(
        r'^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-3])$', r'_\1', safe, flags=re.IGNORECASE
    )
    return safe


class OAIDocumentEmbedder(BaseVectorizer):
    def __init__(self, base_url, api_key, model):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        cache_name = url_to_safe_filename(f"{base_url}_{model}")
        self._cache = EmbedderCache(cache_name)

    def _transform(self, corpus, source_dict, callback=dummy_callback):
        texts = list(corpus.documents)
        results = [None] * len(texts)
        # Collect all cached results
        cache = self._cache
        query = []
        indices = []

        for i, txt in  enumerate(texts):
            r = cache.get_cached_result_or_none(cache.md5_hash(txt.encode("utf-8")))
            if r is not None:
                results[i] = r
            else:
                query.append(txt)
                indices.append(i)

        callback(0.0)
        embs = openai_get_embeddings(
            query, self.api_key, self.base_url, self.model,
            progress_callback=lambda a, b: callback(a/b)
        )
        embs = embs.tolist()
        # Update cache and results list
        for i, r, txt in zip(indices, embs, query):
            cache.add(cache.md5_hash(txt.encode("utf-8"),), r)
            results[i] = r
        cache.persist_cache()
        embs = np.array(results)
        if results:
            dim = embs.shape[1]
            new_corpus = corpus.extend_attributes(
                embs,
                feature_names=["Dim{}".format(i + 1) for i in range(dim)],
                var_attrs={
                    "embedding-feature": True,
                    "hidden": True,
                }
            )
        else:
            new_corpus = corpus
        return new_corpus, None


def openai_get_embeddings(
    texts: list[str],
    api_key: str,
    base_url: Optional[str] = None,
    model: str = "gpt-4o",
    batch_size: int = 20,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """Generate embeddings for texts using an OpenAI-compatible API.

    Processed in concurrent batches, each containing up to ``batch_size``
    texts.

    Args:
        texts: List of texts
        api_key: OpenAI-compatible API key.
        base_url: Base URL for the OpenAI-compatible API. If None, uses the
            default OpenAI endpoint.
        model: Model name to use for embeddings. Defaults to "gpt-4o".
        batch_size: Maximum number of concurrent API requests. Defaults to 10.
        progress_callback: Optional callable invoked as
            ``callback(completed_count, total_count)`` after each
            batch finishes.  Use ``None`` to skip progress reporting.

    Returns:
        Array of embedding vectors, one per query text. The order matches
        the order of the input texts list.

    Raises:
        openai.OpenAIError: If the API call fails.
    """
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    embeddings = [[]] * len(texts)

    total = len(texts)
    batch_number = 0

    for start in range(0, total, batch_size):
        batch_number += 1
        batch_end = min(start + batch_size, total)
        batch_indices = range(start, batch_end)
        batch = texts[start:batch_end]
        resp = client.embeddings.create(
            input=batch, model=model, encoding_format="float",
        )
        if isinstance(resp, list):
            resp = resp
        else:
            resp = resp.data
        for i, r in zip(batch_indices, resp):
            emb = np.array(r.embedding)
            if emb.ndim > 1:
                emb = emb.flatten()
            embeddings[i] = emb
        # Notify progress callback: (completed, total, batch_number)
        if progress_callback is not None:
            completed = min(batch_number * batch_size, total)
            progress_callback(completed, total)
    return np.array(embeddings)


if __name__ == '__main__':
    with DocumentEmbedder(language='en', aggregator='Max') as embedder:
        embedder.clear_cache()
        embedder(Corpus.from_file('deerwester'))
