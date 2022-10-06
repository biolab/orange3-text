import unittest
from typing import List, Union
from unittest.mock import patch
from typing import Iterator
import asyncio

import numpy as np

from orangecontrib.text.ontology import Tree, OntologyHandler


EMB_DIM = 384
RESPONSE = [
    f'{{ "embedding": {[i] * EMB_DIM} }}'.encode()
    for i in range(4)
]
RESPONSE2 = [np.zeros(384), np.ones(384), np.zeros(384), np.ones(384)*2]
RESPONSE3 = [np.zeros(384), np.ones(384), np.arange(384), np.ones(384)*2]


def arrays_to_response(array: List[Union[np.ndarray, List]]) -> Iterator[bytes]:
    return iter(array_to_response(a) for a in array)


def array_to_response(array: Union[np.ndarray, List]) -> bytes:
    return f'{{ "embedding": {array.tolist()} }}'.encode()


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


class TestTree(unittest.TestCase):

    def setUp(self):
        self.dict_format = {
            'root': {
                'child1': {},
                'child2': {},
                'child3': {
                    'child31': {},
                    'child32': {}
                }
            }
        }
        self.tree_format = Tree(
            adj_list=[{1, 2, 3}, {0}, {0}, {0, 4, 5}, {3}, {3}],
            labels=['root', 'child1', 'child2', 'child3', 'child31', 'child32'],
            root=0
        )

    def test_from_dict(self):
        tree = Tree.from_dict(self.dict_format)
        self.assertEqual(tree.adj_list, self.tree_format.adj_list)
        self.assertEqual(tree.labels, self.tree_format.labels)
        self.assertEqual(tree.root, self.tree_format.root)

    def test_to_dict(self):
        self.assertEqual(self.tree_format.to_dict(), self.dict_format)

    def test_from_prufer_sequence(self):
        tree = Tree.from_prufer_sequence([1, 0, 3], list(map(str, range(5))))
        self.assertEqual(len(tree.adj_list), 5)

    def test_assertion_errors(self):
        with self.assertRaises(AssertionError):
            Tree([{1}, {2}], ['1', '2'], 1)
        with self.assertRaises(AssertionError):
            Tree([{0}, {1}], ['1'], 0)
        with self.assertRaises(AssertionError):
            Tree.from_prufer_sequence([1, 0, 3], list(map(str, range(4))))


class TestOntologyHandler(unittest.TestCase):
    def setUp(self):
        self.handler = OntologyHandler()

    def tearDown(self):
        self.handler.embedder.clear_cache()

    @patch('orangecontrib.text.ontology.generate_ontology')
    def test_small_trees(self, mock):
        for words in [[], ['1', '2'], ['1', '2']]:
            self.handler.generate(words)
            mock.assert_not_called()

    @patch('httpx.AsyncClient.post', make_dummy_post(arrays_to_response(RESPONSE3)))
    def test_generate_small(self):
        tree, skipped = self.handler.generate(['1', '2', '3'])
        self.assertTrue(isinstance(tree, dict))
        self.assertEqual(skipped, 0)

    @patch('httpx.AsyncClient.post', make_dummy_post(arrays_to_response(RESPONSE3)))
    def test_generate(self):
        tree, skipped = self.handler.generate(['1', '2', '3', '4'])
        self.assertTrue(isinstance(tree, dict))
        self.assertEqual(skipped, 0)

    @patch('httpx.AsyncClient.post', make_dummy_post(iter(RESPONSE)))
    def test_generate_with_unknown_embeddings(self):
        tree, skipped = self.handler.generate(['1', '2', '3', '4'])
        self.assertTrue(isinstance(tree, dict))
        self.assertEqual(skipped, 0)

    @patch('httpx.AsyncClient.post', make_dummy_post(arrays_to_response(RESPONSE2)))
    def test_insert(self):
        tree, skipped = self.handler.generate(['1', '2', '3'])
        self.assertEqual(skipped, 0)
        new_tree, skipped = self.handler.insert(tree, ['4'])
        self.assertGreater(
            len(Tree.from_dict(new_tree).adj_list),
            len(Tree.from_dict(tree).adj_list)
        )
        self.assertEqual(skipped, 0)

    @patch('httpx.AsyncClient.post', make_dummy_post(array_to_response(np.zeros(384))))
    def test_score(self):
        tree, skipped = self.handler.generate(['1', '2', '3'])
        score = self.handler.score(tree)
        self.assertGreater(score, 0)
        self.assertEqual(skipped, 0)

    @patch('httpx.AsyncClient.post', make_dummy_post(b''))
    def test_embedding_fails_generate(self):
        """ Tests the case when embedding fails totally - return empty tree """
        tree, skipped = self.handler.generate(['1', '2', '3'])
        score = self.handler.score(tree)
        self.assertDictEqual(tree, {})
        self.assertEqual(score, 0)
        self.assertEqual(skipped, 3)

    @patch('httpx.AsyncClient.post', make_dummy_post(
        iter(list(arrays_to_response([np.arange(384), np.ones(384)])) + [b""] * 3)
    ))
    def test_some_embedding_fails_generate(self):
        """
        Tests the case when embedding fail partially
        - consider only successfully embedded words
        """
        tree, skipped = self.handler.generate(['1', '2', '3'])
        score = self.handler.score(tree)
        self.assertDictEqual(tree, {'1': {'2': {}}})
        self.assertGreater(score, 0)
        self.assertEqual(skipped, 1)

    @patch('httpx.AsyncClient.post', make_dummy_post(
        # success for generate part and fail for insert part
        iter(list(arrays_to_response(RESPONSE3)) + [b""] * 3)
    ))
    def test_embedding_fails_insert(self):
        """
        Tests the case when embedding fails for word that is tried to be inserted
        - don't insert it
        """
        tree, skipped = self.handler.generate(['1', '2', '3', '4'])
        self.assertEqual(skipped, 0)
        new_tree, skipped = self.handler.insert(tree, ['5'])
        self.assertDictEqual(tree, new_tree)
        self.assertEqual(skipped, 1)

    @patch('httpx.AsyncClient.post', make_dummy_post(
        # success for generate part and fail for part of new inputs
        iter(list(arrays_to_response(RESPONSE3)) + [b""] * 3)
    ))
    def test_some_embedding_fails_insert(self):
        """
        ests the case when embedding fails for some words that are tried to be
        inserted - insert only successfully embedded words
        """
        tree, skipped = self.handler.generate(['1', '2', '3'])
        self.assertEqual(skipped, 0)
        new_tree, skipped = self.handler.insert(tree, ['4', '5'])
        self.assertDictEqual(new_tree, {'1': {'2': {'4': {}}, '3': {}}})
        self.assertEqual(skipped, 1)


if __name__ == '__main__':
    unittest.main()
