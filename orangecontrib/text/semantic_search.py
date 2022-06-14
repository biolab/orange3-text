import asyncio
import json
import base64
import logging
import zlib
import sys
from collections import namedtuple
from typing import Any, List, Optional, Callable, Union, Dict

from Orange.misc.server_embedder import ServerEmbedderCommunicator
from Orange.misc.utils.embedder_utils import get_proxies
from Orange.util import dummy_callback

# maximum document size that we still send to the server
from httpx import AsyncClient
from numpy import linspace

MAX_PACKAGE_SIZE = 3000000
log = logging.getLogger(__name__)


TaskItem = namedtuple("TaskItem", ("item", "no_repeats", "GPU_locked", "CPU_locked"))


class SemanticSearch:
    def __init__(self) -> None:
        self._server_communicator = _ServerCommunicator(
            model_name='semantic-search',
            max_parallel_requests=100,
            server_url='https://api.garaza.io',
            embedder_type='text',
        )

    def __call__(
        self, texts: List[str], queries: List[str], callback: Callable = dummy_callback
    ) -> List[Optional[List[List[Union[List[int], float]]]]]:
        """Computes matches for given documents and queries.

        Parameters
        ----------
        texts
            A list of raw texts to be matched.
        queries
            A list of query words/phrases.

        Returns
        -------
        The elements of the outer list represent each document. The entries
        are either None or lists of matches. Entries of each list of matches
        are matches for each sentence. Each match is of the form
        ((start_idx, end_idx), score). Note that tuples are actually
        lists since JSON (that the server returns) does not support tuples.
        """
        if len(texts) == 0 or len(queries) == 0:
            return [None] * len(texts)

        enc_words = self.encode_queries(queries)

        encoded_texts = []
        for text in texts:
            enc_text = self.encode_text(text)
            size = sys.getsizeof(enc_text)
            encoded_texts.append(
                [enc_text, enc_words] if size < MAX_PACKAGE_SIZE else None
            )

        # sort text by their lengths that longer texts start to embed first. It
        # prevents that long text with long embedding times start embedding
        # at the end and thus add extra time to the complete embedding time
        sorted_texts = sorted(
            enumerate(encoded_texts),
            key=lambda x: len(x[1][0]) if x[1] is not None else 0,
            reverse=True
        )
        indices, queries = zip(*sorted_texts)

        # send data to the server
        result_ = self._server_communicator.embedd_data(queries, callback=callback)
        if result_ is None:
            return [None] * len(texts)

        # restore the original order of the data
        results = [x for _, x in sorted(zip(indices, result_))]
        return results

    @staticmethod
    def encode_queries(queries):
        return base64.b64encode(
            zlib.compress(json.dumps(queries).encode('utf-8', 'replace'), level=-1)
        ).decode('utf-8', 'replace')

    @staticmethod
    def encode_text(text):
        return base64.b64encode(
            zlib.compress(text.encode('utf-8', 'replace'), level=-1)
        ).decode('utf-8', 'replace')

    def clear_cache(self):
        if self._server_communicator:
            self._server_communicator.clear_cache()


class _ServerCommunicator(ServerEmbedderCommunicator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.content_type = 'application/json'
        self.timeout = 300
        self._model_gpu = "semantic-search-gpu"
        self.max_parallel_requests_gpu = 4

    async def embedd_batch(
        self,
        data: List[Any],
        processed_callback: Optional[Callable] = None,
        *,
        callback: Callable = dummy_callback,
    ) -> List[Optional[List[float]]]:
        """
        Function perform embedding of a batch of data items.

        Parameters
        ----------
        data
            A list of data that must be embedded.
        callback
            Callback for reporting the progress in share of embedded items

        Returns
        -------
        List of float list (embeddings) for successfully embedded
        items and Nones for skipped items.

        Raises
        ------
        EmbeddingCancelledException:
            If cancelled attribute is set to True (default=False).
        """
        progress_items = iter(linspace(0, 1, len(data)))

        def success_callback():
            """Callback called on every successful embedding"""
            callback(next(progress_items))

        results = [None] * len(data)
        queue = {}

        # fill the queue with items to embedd
        sorted_texts = sorted(
            enumerate(data),
            key=lambda x: len(x[1][0]) if x[1] is not None else 0,
            reverse=True
        )
        for i, item in sorted_texts:
            queue[i] = TaskItem(item=item, no_repeats=0, GPU_locked=False, CPU_locked=False)

        async with AsyncClient(
            timeout=self.timeout, base_url=self.server_url, proxies=get_proxies()
        ) as client:
            tasks = self._init_workers(client, queue, results, success_callback)
            tasks += self._init_workers(client, queue, results, success_callback)

            try:
                # wait for workers to stop - they stop when queue is empty
                # if one worker raises exception gather will raise it further
                # todo: wait until empty queue
                await asyncio.gather(*tasks)
            finally:
                await self._cancel_workers(tasks)
                self._cache.persist_cache()

        return results

    def _init_workers_cpu(self, client, queue, results, callback):
        """Init required number of workers"""
        t = [
            asyncio.create_task(self._send_to_server(client, queue, results, callback))
            # when number of instances less than max_parallel_requests create
            # only required number of workers
            for _ in range(min(self.max_parallel_requests, len(results)))
        ]
        log.debug("Created %d workers", self.max_parallel_requests)
        return t

    def _init_workers_gpu(self, client, queue, results, callback):
        """Init required number of workers"""
        t = [
            asyncio.create_task(self._send_to_server(client, queue, results, callback))
            # when number of instances less than max_parallel_requests create
            # only required number of workers
            for _ in range(min(self.max_parallel_requests_gpu, len(results)))
        ]
        log.debug("Created %d workers", self.max_parallel_requests)
        return t

    async def _send_to_server_cpu(
        self,
        client: AsyncClient,
        queue: Dict[int, Any],
        results: List,
        proc_callback: Callable,
    ):
        while len(queue):
            # get item from the queue
            for i, (data_instance, num_repeats, gpu_locked, cpu_locked) in reversed(queue.items()):
                if not cpu_locked:
                    queue[i] = TaskItem(data_instance, num_repeats, gpu_locked, True)
                    break
            else:
                # all requests are locked - it means there is enough workers
                # to handle the situation - quit this one
                break

            # load bytes
            data_bytes = await self._encode_data_instance(data_instance)
            if data_bytes is None:
                queue.pop(i)
                continue

            # retrieve embedded item from the local cache
            cache_key = self._cache.md5_hash(data_bytes)
            log.debug("Embedding %s", cache_key)
            emb = self._cache.get_cached_result_or_none(cache_key)

            if emb is None:
                # send the item to the server for embedding if not in the local cache
                log.debug("Sending to the server: %s", cache_key)
                url = (
                    f"/{self.embedder_type}/{self._model}?machine={self.machine_id}"
                    f"&session={self.session_id}&retry={num_repeats+1}"
                )
                emb = await self._send_request(client, data_bytes, url)
                if emb is not None:
                    self._cache.add(cache_key, emb)

            if emb is not None:
                # store result if embedding is successful
                log.debug("Successfully embedded:  %s", cache_key)
                if results[i] is None:
                    # other embedder didn't embedd it yet
                    proc_callback()
                results[i] = emb
                queue.pop(i)
            elif num_repeats+1 < self.MAX_REPEATS:
                log.debug("Embedding unsuccessful - reading to queue:  %s", cache_key)
                # if embedding not successful put the item to queue to be handled at
                # the end - the item is put to the end since it is possible that  server
                # still process the request and the result will be in the cache later
                # repeating the request immediately may result in another fail when
                # processing takes longer
                item = queue[i]  # in case that qpu embedder changed the state
                queue[i] = TaskItem(data_instance, item.num_repeats+1, item.GPU_locked, False)

    async def _send_to_server_gpu(
        self,
        client: AsyncClient,
        queue: Dict[int, Any],
        results: List,
        proc_callback: Callable,
    ):
        while len(queue):
            # get item from the queue
            for i, (data_instance, num_repeats, gpu_locked, cpu_locked) in queue.items():
                if not gpu_locked:
                    queue[i] = TaskItem(data_instance, num_repeats, True, cpu_locked)
                    break
            else:
                # all requests are locked - it means there is enough workers
                # to handle the situation - quit this one
                break

            # load bytes
            data_bytes = await self._encode_data_instance(data_instance)
            if data_bytes is None:
                queue.pop(i)
                continue

            # retrieve embedded item from the local cache
            cache_key = self._cache.md5_hash(data_bytes)
            log.debug("Embedding %s", cache_key)
            emb = self._cache.get_cached_result_or_none(cache_key)

            if emb is None:
                # send the item to the server for embedding if not in the local cache
                log.debug("Sending to the server: %s", cache_key)
                url = (
                    f"/{self.embedder_type}/{self._model_gpu}?machine={self.machine_id}"
                    f"&session={self.session_id}&retry={num_repeats+1}"
                )
                emb = await self._send_request(client, data_bytes, url)
                if emb is not None:
                    self._cache.add(cache_key, emb)

            if emb is not None:
                # store result if embedding is successful
                log.debug("Successfully embedded:  %s", cache_key)
                if results[i] is None:
                    # other embedder didn't embedd it yet
                    proc_callback()
                results[i] = emb
                queue.pop(i)
            elif num_repeats+1 < self.MAX_REPEATS:
                log.debug("Embedding unsuccessful - reading to queue:  %s", cache_key)
                # if embedding not successful put the item to queue to be handled at
                # the end - the item is put to the end since it is possible that  server
                # still process the request and the result will be in the cache later
                # repeating the request immediately may result in another fail when
                # processing takes longer
                item = queue[i]  # in case that qpu embedder changed the state
                queue[i] = TaskItem(data_instance, item.num_repeats+1, False, item.CPU_locked)

    async def _encode_data_instance(self, data_instance: Any) -> Optional[bytes]:
        if data_instance is None:
            return None
        else:
            return json.dumps(data_instance).encode('utf-8', 'replace')


if __name__ == "__main__":
    from orangecontrib.text import Corpus

    corpus = Corpus.from_file("grimm-tales-selected")
    results = SemanticSearch()(corpus.documents, ["book", "is", "rum", "are"])
    for r in results:
        print(r)
