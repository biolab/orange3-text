import unittest
from unittest.mock import patch
from collections.abc import Iterator
import asyncio

from orangecontrib.text.vectorization.sbert import SBERT, EMB_DIM
from orangecontrib.text import Corpus

PATCH_METHOD = 'httpx.AsyncClient.post'
RESPONSE = [
    f'{{ "embedding": {[i] * EMB_DIM} }}'.encode()
    for i in range(9)
]

IDEAL_RESPONSE = [[i] * EMB_DIM for i in range(9)]


class DummyResponse:

    def __init__(self, content):
        self.content = content


def make_dummy_post(response, sleep=0):
    @staticmethod
    async def dummy_post(url, headers, data=None, content=None):
        assert data or content
        await asyncio.sleep(sleep)
        return DummyResponse(
            content=next(response) if isinstance(response, Iterator) else response
        )
    return dummy_post


class TestSBERT(unittest.TestCase):
    def setUp(self):
        self.sbert = SBERT()
        self.sbert.clear_cache()
        self.corpus = Corpus.from_file('deerwester')

    def tearDown(self):
        self.sbert.clear_cache()

    @patch(PATCH_METHOD)
    def test_empty_corpus(self, mock):
        self.assertEqual(len(self.sbert(self.corpus.documents[:0])), 0)
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

    @patch(PATCH_METHOD, make_dummy_post(iter(RESPONSE)))
    def test_transform(self):
        res, skipped = self.sbert.transform(self.corpus)
        self.assertIsNone(skipped)
        self.assertEqual(len(self.corpus), len(res))
        self.assertTupleEqual(self.corpus.domain.metas, res.domain.metas)
        self.assertEqual(384, len(res.domain.attributes))

    @patch(PATCH_METHOD, make_dummy_post(iter(RESPONSE[:-1] + [None] * 3)))
    def test_transform_skipped(self):
        res, skipped = self.sbert.transform(self.corpus)
        self.assertEqual(len(self.corpus) - 1, len(res))
        self.assertTupleEqual(self.corpus.domain.metas, res.domain.metas)
        self.assertEqual(384, len(res.domain.attributes))

        self.assertEqual(1, len(skipped))
        self.assertTupleEqual(self.corpus.domain.metas, skipped.domain.metas)
        self.assertEqual(0, len(skipped.domain.attributes))


if __name__ == "__main__":
    unittest.main()
