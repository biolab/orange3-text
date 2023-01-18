import base64
import json
import unittest
import zlib
from unittest.mock import patch, ANY
import asyncio

from orangecontrib.text.vectorization.sbert import SBERT, EMB_DIM
from orangecontrib.text import Corpus

PATCH_METHOD = 'httpx.AsyncClient.post'
RESPONSES = {
    t: [i] * EMB_DIM for i, t in enumerate(Corpus.from_file("deerwester").documents)
}
RESPONSE_NONE = RESPONSES.copy()
RESPONSE_NONE[list(RESPONSE_NONE.keys())[-1]] = None
IDEAL_RESPONSE = [[i] * EMB_DIM for i in range(9)]


class DummyResponse:
    def __init__(self, content):
        self.content = content


def _decompress_text(instance):
    return zlib.decompress(base64.b64decode(instance.encode("utf-8"))).decode("utf-8")


def make_dummy_post(responses, sleep=0):
    @staticmethod
    async def dummy_post(url, headers, data=None, content=None):
        assert data or content
        await asyncio.sleep(sleep)
        data = json.loads(content.decode("utf-8", "replace"))
        data_ = data if isinstance(data, list) else [data]
        texts = [_decompress_text(instance) for instance in data_]
        responses_ = [responses[t] for t in texts]
        r = {"embedding": responses_ if isinstance(data, list) else responses_[0]}
        return DummyResponse(content=json.dumps(r).encode("utf-8"))
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

    @patch(PATCH_METHOD, make_dummy_post(RESPONSES))
    def test_success(self):
        result = self.sbert(self.corpus.documents)
        self.assertEqual(result, IDEAL_RESPONSE)

    @patch(PATCH_METHOD, make_dummy_post(RESPONSE_NONE))
    def test_none_result(self):
        result = self.sbert(self.corpus.documents)
        self.assertEqual(result, IDEAL_RESPONSE[:-1] + [None])

    @patch(PATCH_METHOD, make_dummy_post(RESPONSES))
    def test_transform(self):
        res, skipped = self.sbert.transform(self.corpus)
        self.assertIsNone(skipped)
        self.assertEqual(len(self.corpus), len(res))
        self.assertTupleEqual(self.corpus.domain.metas, res.domain.metas)
        self.assertEqual(384, len(res.domain.attributes))

    @patch(PATCH_METHOD, make_dummy_post(RESPONSE_NONE))
    def test_transform_skipped(self):
        res, skipped = self.sbert.transform(self.corpus)
        self.assertEqual(len(self.corpus) - 1, len(res))
        self.assertTupleEqual(self.corpus.domain.metas, res.domain.metas)
        self.assertEqual(384, len(res.domain.attributes))

        self.assertEqual(1, len(skipped))
        self.assertTupleEqual(self.corpus.domain.metas, skipped.domain.metas)
        self.assertEqual(0, len(skipped.domain.attributes))

    @patch(PATCH_METHOD, make_dummy_post(RESPONSES))
    def test_batches_success(self):
        for i in range(1, 11):  # try different batch sizes
            result = self.sbert.embed_batches(self.corpus.documents, i)
            self.assertEqual(result, IDEAL_RESPONSE)

    @patch(PATCH_METHOD, make_dummy_post(RESPONSE_NONE))
    def test_batches_none_result(self):
        for i in range(1, 11):  # try different batch sizes
            result = self.sbert.embed_batches(self.corpus.documents, i)
            self.assertEqual(result, IDEAL_RESPONSE[:-1] + [None])

    @patch("orangecontrib.text.vectorization.sbert._ServerCommunicator.embedd_data")
    def test_reordered(self, mock):
        """Test that texts are reordered according to their length"""
        self.sbert(self.corpus.documents)
        mock.assert_called_with(
            tuple(sorted(self.corpus.documents, key=len, reverse=True)), callback=ANY
        )

        self.sbert([["1", "2"], ["4", "5", "6"], ["0"]])
        mock.assert_called_with((["4", "5", "6"], ["1", "2"], ["0"]), callback=ANY)


if __name__ == "__main__":
    unittest.main()
