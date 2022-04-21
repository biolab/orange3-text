import unittest
from unittest.mock import patch
from collections.abc import Iterator
import asyncio

from orangecontrib.text.vectorization.sbert import (
    SBERT,
    MIN_CHUNKS,
    MAX_PACKAGE_SIZE,
    EMB_DIM
)
from orangecontrib.text import Corpus

PATCH_METHOD = 'httpx.AsyncClient.post'
RESPONSE = [
    f'{{ "embedding": [{[i] * EMB_DIM}] }}'.encode()
    for i in range(9)
]

IDEAL_RESPONSE = [[i] * EMB_DIM for i in range(9)]


class DummyResponse:

    def __init__(self, content):
        self.content = content


def make_dummy_post(response, sleep=0):
    @staticmethod
    async def dummy_post(url, headers, data):
        await asyncio.sleep(sleep)
        return DummyResponse(
            content=next(response) if isinstance(response, Iterator) else response
        )
    return dummy_post


class TestSBERT(unittest.TestCase):

    def setUp(self):
        self.sbert = SBERT()
        self.corpus = Corpus.from_file('deerwester')

    def tearDown(self):
        self.sbert.clear_cache()

    def test_make_chunks_small(self):
        chunks = self.sbert._make_chunks(
            self.corpus.documents, [100] * len(self.corpus.documents)
        )
        self.assertEqual(len(chunks), min(len(self.corpus.documents), MIN_CHUNKS))

    def test_make_chunks_medium(self):
        num_docs = len(self.corpus.documents)
        documents = self.corpus.documents
        if num_docs < MIN_CHUNKS:
            documents = [documents[0]] * MIN_CHUNKS
        chunks = self.sbert._make_chunks(
            documents, [MAX_PACKAGE_SIZE / MIN_CHUNKS - 1] * len(documents)
        )
        self.assertEqual(len(chunks), MIN_CHUNKS)

    def test_make_chunks_large(self):
        num_docs = len(self.corpus.documents)
        documents = self.corpus.documents
        if num_docs < MIN_CHUNKS:
            documents = [documents[0]] * MIN_CHUNKS * 100
        mps = MAX_PACKAGE_SIZE
        chunks = self.sbert._make_chunks(
            documents,
            [mps / 100] * (len(documents) - 2) + [0.3 * mps, 0.9 * mps, mps]
        )
        self.assertGreater(len(chunks), MIN_CHUNKS)

    @patch(PATCH_METHOD)
    def test_empty_corpus(self, mock):
        self.assertEqual(
            len(self.sbert(self.corpus.documents[:0])), 0
        )
        mock.request.assert_not_called()
        mock.get_response.assert_not_called()
        self.assertEqual(
            self.sbert._server_communicator._cache._cache_dict,
            dict()
        )

    @patch(PATCH_METHOD, make_dummy_post(iter(RESPONSE)))
    def test_success(self):
        result = self.sbert(self.corpus.documents)
        self.assertEqual(result, IDEAL_RESPONSE)

    @patch(PATCH_METHOD, make_dummy_post(iter(RESPONSE[:-1] + [None] * 3)))
    def test_none_result(self):
        result = self.sbert(self.corpus.documents)
        self.assertEqual(result, IDEAL_RESPONSE[:-1] + [None])

    @patch(PATCH_METHOD, make_dummy_post(RESPONSE[0]))
    def test_success_chunks(self):
        num_docs = len(self.corpus.documents)
        documents = self.corpus.documents
        if num_docs < MIN_CHUNKS:
            documents = [documents[0]] * MIN_CHUNKS
        result = self.sbert(documents)
        self.assertEqual(len(result), MIN_CHUNKS)


if __name__ == "__main__":
    unittest.main()
