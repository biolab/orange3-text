import json
import base64
import zlib
import sys
import re
from typing import Any, Optional, List, Optional, Tuple, Callable

import numpy as np

from Orange.misc.server_embedder import ServerEmbedderCommunicator
from orangecontrib.text import Corpus
from orangecontrib.text.misc import wait_nltk_data


MAX_PACKAGE_SIZE = 50000
MIN_CHUNKS = 20


class SemanticSearch:

    def __init__(self) -> None:
        self._server_communicator = _ServerCommunicator(
            model_name='semantic-search',
            max_parallel_requests=100,
            server_url='https://apiv2.garaza.io',
            embedder_type='text',
        )

    def __call__(
        self, texts: List[str], queries: List[str],
        progress_callback: Optional[Callable] = None
    ) -> List[Optional[List[Tuple[Tuple[int, int], float]]]]:
        """Computes matches for given documents and queries.

        Parameters
        ----------
        texts: List[str]
            A list of raw texts to be matched.
        queries: List[str]
            A list of query words/phrases.

        Returns
        -------
        List[Optional[List[Tuple[Tuple[int, int], float]]]]
            The elements of the outer list represent each document. The entries
            are either None or lists of matches. Entries of each list of matches
            are matches for each sentence. Each match is of the form
            ((start_idx, end_idx), score). Note that tuples are actually
            converted to lists before the result is returned.
        """

        if len(texts) == 0 or len(queries) == 0:
            return [None] * len(texts)

        chunks = list()
        chunk = list()
        skipped = list()
        queries_enc = base64.b64encode(
            zlib.compress(
                json.dumps(queries).encode('utf-8', 'replace'),
                level=-1
            )
        ).decode('utf-8', 'replace')

        encoded_texts = list()
        sizes = list()
        chunks = list()
        for i, text in enumerate(texts):
            encoded = base64.b64encode(zlib.compress(
                text.encode('utf-8', 'replace'), level=-1)
            ).decode('utf-8', 'replace')
            size = sys.getsizeof(encoded)
            if size > MAX_PACKAGE_SIZE:
                skipped.append(i)
                continue
            encoded_texts.append(encoded)
            sizes.append(size)

        chunks_ = self._make_chunks(encoded_texts, sizes)
        for chunk in chunks_:
            chunks.append([chunk, queries_enc])

        result_ = self._server_communicator.embedd_data(
            chunks, processed_callback=progress_callback,
        )
        if result_ is None:
            return [None] * len(texts)

        result = list()
        for chunk in result_:
            result.extend(chunk)

        results = list()
        idx = 0
        for i in range(len(texts)):
            if i in skipped:
                results.append(None)
            else:
                results.append(result[idx])
                idx += 1

        return results

    def _make_chunks(self, encoded_texts, sizes, depth=0):
        chunks = np.array_split(encoded_texts, MIN_CHUNKS if depth == 0 else 2)
        chunk_sizes = np.array_split(sizes, MIN_CHUNKS if depth == 0 else 2)
        result = list()
        for i in range(len(chunks)):
            if np.sum(chunk_sizes[i]) > MAX_PACKAGE_SIZE:
                result.extend(self._make_chunks(chunks[i], chunk_sizes[i], depth + 1))
            else:
                result.append(chunks[i])
        return [list(r) for r in result if len(r) > 0]

    def set_cancelled(self):
        if hasattr(self, '_server_communicator'):
            self._server_communicator.set_cancelled()

    def clear_cache(self):
        if self._server_communicator:
            self._server_communicator.clear_cache()

    def __enter__(self):
        return self

    def __exit__(self, ex_type, value, traceback):
        self.set_cancelled()

    def __del__(self):
        self.__exit__(None, None, None)


class _ServerCommunicator(ServerEmbedderCommunicator):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.content_type = 'application/json'

    async def _encode_data_instance(self, data_instance: Any) -> Optional[bytes]:
        return json.dumps(data_instance).encode('utf-8', 'replace')
