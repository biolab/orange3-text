import asyncio
import json
import base64
import warnings
import zlib
import sys
from threading import Thread
from typing import Any, List, Optional, Callable, Tuple

import numpy as np
from Orange.misc.server_embedder import ServerEmbedderCommunicator
from Orange.util import dummy_callback

from orangecontrib.text import Corpus
from orangecontrib.text.vectorization.base import BaseVectorizer

# maximum document size that we still send to the server
MAX_PACKAGE_SIZE = 3000000
EMB_DIM = 384


class SBERT(BaseVectorizer):
    def __init__(self) -> None:
        self._server_communicator = _ServerCommunicator(
            model_name="sbert",
            max_parallel_requests=100,
            server_url="https://api.garaza.io",
            embedder_type="text",
        )

    def __call__(
        self, texts: List[str], callback: Callable = dummy_callback
    ) -> List[Optional[List[float]]]:
        """Computes embeddings for given documents.

        Parameters
        ----------
        texts
            A list of raw texts.

        Returns
        -------
        An array of embeddings.
        """
        if len(texts) == 0:
            return []
        # sort text by their lengths that longer texts start to embed first. It
        # prevents that long text with long embedding times start embedding
        # at the end and thus add extra time to the complete embedding time
        sorted_texts = sorted(
            enumerate(texts),
            key=lambda x: len(x[1][0]) if x[1] is not None else 0,
            reverse=True,
        )
        indices, sorted_texts = zip(*sorted_texts)
        # embedd - send to server
        results = self._server_communicator.embedd_data(sorted_texts, callback=callback)
        # unsort and unpack
        return [x if x else None for _, x in sorted(zip(indices, results))]

    def _transform(
        self, corpus: Corpus, _, callback=dummy_callback
    ) -> Tuple[Corpus, Optional[Corpus]]:
        """
        Computes embeddings for given corpus and append results to the corpus

        Parameters
        ----------
        corpus
            Corpus on which transform is performed.

        Returns
        -------
        Embeddings
            Corpus with new features added.
        Skipped documents
            Corpus of documents that were not embedded
        """
        embs = self(corpus.documents, callback)

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
                    [e for e in embs if e],
                    dtype=float,
                ),
                ["Dim{}".format(i + 1) for i in range(EMB_DIM)],
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

    def report(self) -> Tuple[Tuple[str, str], ...]:
        """Reports on current parameters of DocumentEmbedder.

        Returns
        -------
        tuple
            Tuple of parameters.
        """
        return (("Embedder", "Multilingual SBERT"),)

    def clear_cache(self):
        if self._server_communicator:
            self._server_communicator.clear_cache()


class RunThread(Thread):
    def __init__(self, func, *args, **kwargs):
        self.func, self.args, self.kwargs = func, args, kwargs
        self.result = None
        super().__init__()

    def run(self):
        self.result = asyncio.run(self.func(*self.args, **self.kwargs))


class _ServerCommunicator(ServerEmbedderCommunicator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.content_type = "application/json"

    def embedd_data(
        self,
        data: List[Any],
        callback: Callable = dummy_callback,
    ) -> List[Optional[List[float]]]:
        # if there is less items than 10 connection error should be raised earlier
        self.max_errors = min(len(data) * self.MAX_REPEATS, 10)

        # in case of ontology widget it happens that QSelectorEventLoop event
        # loop already exists (non-running, by QT). asyncio.run fails in that case
        # use separate thread in case of existing event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop:
            thread = RunThread(self.embedd_batch, data, callback=callback)
            thread.start()
            thread.join()
            return thread.result
        else:
            return asyncio.run(self.embedd_batch(data, callback=callback))

    async def _encode_data_instance(self, data_instance: Any) -> Optional[bytes]:
        data = base64.b64encode(
            zlib.compress(data_instance.encode("utf-8", "replace"), level=-1)
        ).decode("utf-8", "replace")
        if sys.getsizeof(data) > 500000:
            # Document in corpus is too large. Size limit is 500 KB
            # (after compression). - document skipped
            return None
        return json.dumps(data).encode("utf-8", "replace")
