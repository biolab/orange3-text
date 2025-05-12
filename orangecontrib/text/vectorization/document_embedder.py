"""This module contains classes used for embedding documents
into a vector space.
"""
import base64
import json
import sys
import warnings
import zlib
from typing import Any, Optional, Tuple

import numpy as np
from Orange.misc.server_embedder import ServerEmbedderCommunicator
from Orange.misc.utils.embedder_utils import EmbedderCache
from Orange.util import dummy_callback

from orangecontrib.text import Corpus
from orangecontrib.text.vectorization.base import BaseVectorizer

from Orange.misc.utils.embedder_utils import EmbeddingConnectionError
import socket
from urllib.parse import urlparse

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

        try:
            url = urlparse(embedder.server_url)
            host, port = url.hostname, url.port or (443 if url.scheme == "https" else 80)

            try:
                sock = socket.create_connection((host, port), timeout=3)
                sock.close()
            except socket.gaierror as e:
                try:
                    socket.gethostbyname("example.com")
                    raise ConnectionError("The server is not responding (bad hostname)") from e
                except socket.gaierror:
                    raise OSError("No internet connection (DNS failure)") from e
            except (ConnectionRefusedError, socket.timeout, OSError):
                raise ConnectionError("The server is not responding (socket check)")

            embs = embedder.embedd_data(
                list(corpus.ngrams) if isinstance(corpus, Corpus) else corpus,
                callback=callback,
            )
            if not embs or all(e is None for e in embs):
                raise ConnectionError("The server is not responding (no embeddings returned)")

        except OSError as e:
            raise EmbeddingConnectionError("No internet connection") from e
        except ConnectionError as e:
            raise EmbeddingConnectionError("The server is not responding") from e
        except Exception as e:
            raise EmbeddingConnectionError(f"Unknown network error: {e}") from e

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


if __name__ == '__main__':
    with DocumentEmbedder(language='en', aggregator='Max') as embedder:
        embedder.clear_cache()
        embedder(Corpus.from_file('deerwester'))