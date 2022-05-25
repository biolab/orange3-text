import json
import base64
import zlib
import sys
from typing import Any, List, Optional, Callable, Union

from Orange.misc.server_embedder import ServerEmbedderCommunicator
from Orange.util import dummy_callback

# maximum document size that we still send to the server
MAX_PACKAGE_SIZE = 3000000


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
