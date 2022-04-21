import json
import base64
import zlib
import sys
from typing import Any, List, Optional, Callable

import numpy as np

from Orange.misc.server_embedder import ServerEmbedderCommunicator
from Orange.util import dummy_callback

# maximum document size that we still send to the server
MAX_PACKAGE_SIZE = 3000000
# maximum size of a chunk - when one document is longer send is as a chunk with
# a single document
MAX_CHUNK_SIZE = 50000
MIN_CHUNKS = 20
EMB_DIM = 384


class SBERT:
    def __init__(self) -> None:
        self._server_communicator = _ServerCommunicator(
            model_name='sbert',
            max_parallel_requests=100,
            server_url='https://apiv2.garaza.io',
            embedder_type='text',
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

        skipped = list()

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

        chunks = self._make_chunks(encoded_texts, sizes)

        result_ = self._server_communicator.embedd_data(chunks, callback=callback)
        if result_ is None:
            return [None] * len(texts)

        result = list()
        assert len(result_) == len(chunks)
        for res_chunk, orig_chunk in zip(result_, chunks):
            if res_chunk is None:
                # when embedder fails (Timeout or other error) result will be None
                result.extend([None] * len(orig_chunk))
            else:
                result.extend(res_chunk)

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
            # checking that more than one text in chunk prevent recursion to infinity
            # when one text is bigger than MAX_CHUNK_SIZE
            if len(chunks[i]) > 1 and np.sum(chunk_sizes[i]) > MAX_CHUNK_SIZE:
                result.extend(self._make_chunks(chunks[i], chunk_sizes[i], depth + 1))
            else:
                result.append(chunks[i])
        return [list(r) for r in result if len(r) > 0]

    def clear_cache(self):
        if self._server_communicator:
            self._server_communicator.clear_cache()

    def __enter__(self):
        return self


class _ServerCommunicator(ServerEmbedderCommunicator):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.content_type = 'application/json'

    async def _encode_data_instance(self, data_instance: Any) -> Optional[bytes]:
        return json.dumps(data_instance).encode('utf-8', 'replace')
