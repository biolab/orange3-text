import unittest
from unittest.mock import patch
from collections.abc import Iterator
import asyncio

from orangecontrib.text.semantic_search import SemanticSearch
from orangecontrib.text import Corpus

PATCH_METHOD = 'httpx.AsyncClient.post'
QUERIES = ['test query', 'another test query']
RESPONSE = [
    b'{ "embedding": [[[0, 57], 0.22114424407482147]] }',
    b'{ "embedding": [[[0, 57], 0.5597518086433411]] }',
    b'{ "embedding": [[[0, 40], 0.11774948984384537]] }',
    b'{ "embedding": [[[0, 50], 0.2228381633758545]] }',
    b'{ "embedding": [[[0, 61], 0.19825558364391327]] }',
    b'{ "embedding": [[[0, 47], 0.19025272130966187]] }',
    b'{ "embedding": [[[0, 40], 0.09688498824834824]] }',
    b'{ "embedding": [[[0, 55], 0.2982504367828369]] }',
    b'{ "embedding": [[[0, 12], 0.2982504367828369]] }',
]
IDEAL_RESPONSE = [
    [[[0, 57], 0.22114424407482147]],
    [[[0, 57], 0.5597518086433411]],
    [[[0, 40], 0.11774948984384537]],
    [[[0, 50], 0.2228381633758545]],
    [[[0, 61], 0.19825558364391327]],
    [[[0, 47], 0.19025272130966187]],
    [[[0, 40], 0.09688498824834824]],
    [[[0, 55], 0.2982504367828369]],
    [[[0, 12], 0.2982504367828369]]
]


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


def sort_by_doc_len(documents):
    """
    Sort document in the same order as they are sorted by embedder before they
    are sent to the server. This we can test results given by pathed method.
    """
    return sorted(
        documents, key=lambda doc: len(SemanticSearch.encode_text(doc)), reverse=True
    )


class SemanticSearchTest(unittest.TestCase):

    def setUp(self):
        self.semantic_search = SemanticSearch()
        self.corpus = Corpus.from_file('deerwester')

    def tearDown(self):
        self.semantic_search.clear_cache()

    @patch(PATCH_METHOD)
    def test_empty_corpus(self, mock):
        self.assertEqual(
            len(self.semantic_search(self.corpus.documents[:0], QUERIES)), 0
        )
        mock.request.assert_not_called()
        mock.get_response.assert_not_called()
        self.assertEqual(
            self.semantic_search._server_communicator._cache._cache_dict,
            dict()
        )

    @patch(PATCH_METHOD, make_dummy_post(iter(RESPONSE)))
    def test_success(self):
        result = self.semantic_search(sort_by_doc_len(self.corpus.documents), QUERIES)
        self.assertEqual(result, IDEAL_RESPONSE)

    # added None three times since server will repeate request on None response
    # three times
    @patch(PATCH_METHOD, make_dummy_post(iter(RESPONSE[:-1] + [None] * 3)))
    def test_none_result(self):
        """
        It can happen that the result of an embedding for a chunk is None (server
        fail to respond three times because Timeout or other error).
        Make sure that semantic search module can handle None responses.
        """
        result = self.semantic_search(sort_by_doc_len(self.corpus.documents), QUERIES)
        self.assertEqual(result, IDEAL_RESPONSE[:-1] + [None])

    @patch(PATCH_METHOD, make_dummy_post(None))
    def test_all_none(self):
        result = self.semantic_search(self.corpus.documents * 10, QUERIES)
        self.assertListEqual(result, [None] * len(self.corpus.documents) * 10)


if __name__ == "__main__":
    unittest.main()
