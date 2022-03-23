import unittest
from unittest.mock import patch
from collections.abc import Iterator
import os
import asyncio

import numpy as np

from orangecontrib.text.ontology import Tree, EmbeddingStorage, OntologyHandler, EMB_DIM


RESPONSE = [
    f'{{ "embedding": [{[i] * EMB_DIM}] }}'.encode()
    for i in range(4)
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


class TestEmbeddingStorage(unittest.TestCase):

    def setUp(self):
        self.storage = EmbeddingStorage()

    def tearDown(self):
        self.storage.clear_storage()

    def test_clear_storage(self):
        self.storage.save_embedding("testword", np.zeros(3))
        self.assertEqual(len(self.storage.embeddings), 1)
        self.storage.clear_storage()
        self.assertEqual(len(self.storage.embeddings), 0)
        self.assertEqual(len(os.listdir(self.storage.cache_dir)), 0)

    def test_save_embedding(self):
        self.storage.save_embedding("testword", np.zeros(3))
        self.storage.save_embedding("testword2", np.zeros(3))
        self.assertEqual(len(self.storage.embeddings), 2)
        self.assertEqual(len(os.listdir(self.storage.cache_dir)), 2)

    def test_get_embedding(self):
        self.storage.save_embedding("testword", np.arange(3))
        emb = self.storage.get_embedding("testword")
        self.assertEqual(emb.tolist(), [0, 1, 2])

    def test_get_from_cache(self):
        self.storage.save_embedding("testword", np.arange(3))
        self.storage.embeddings = dict()
        emb = self.storage.get_embedding("testword")
        self.assertEqual(emb.tolist(), [0, 1, 2])

    def test_similarities(self):
        self.storage.similarities['a', 'b'] = 0.75
        self.storage.save_similarities()
        storage = EmbeddingStorage()
        self.assertEqual(len(storage.similarities), 1)
        self.assertTrue(('a', 'b') in storage.similarities)
        self.assertEqual(storage.similarities['a', 'b'], 0.75)


class TestOntologyHandler(unittest.TestCase):

    def setUp(self):
        self.handler = OntologyHandler()

    def tearDown(self):
        self.handler.storage.clear_storage()
        self.handler.embedder.clear_cache()

    @patch('orangecontrib.text.ontology.generate_ontology')
    def test_small_trees(self, mock):
        for words in [[], ['1', '2'], ['1', '2']]:
            self.handler.generate(words)
            mock.assert_not_called()

    def test_generate_small(self):
        self.handler.storage.save_embedding('1', np.zeros(384))
        self.handler.storage.save_embedding('2', np.ones(384))
        self.handler.storage.save_embedding('3', np.arange(384))
        tree = self.handler.generate(['1', '2', '3'])
        self.assertTrue(isinstance(tree, dict))

    @patch('httpx.AsyncClient.post')
    def test_generate(self, mock):
        self.handler.storage.save_embedding('1', np.zeros(384))
        self.handler.storage.save_embedding('2', np.ones(384))
        self.handler.storage.save_embedding('3', np.arange(384))
        self.handler.storage.save_embedding('4', np.ones(384) * 2)
        tree = self.handler.generate(['1', '2', '3', '4'])
        self.assertTrue(isinstance(tree, dict))
        mock.request.assert_not_called()
        mock.get_response.assert_not_called()

    @patch('httpx.AsyncClient.post', make_dummy_post(iter(RESPONSE)))
    def test_generate_with_unknown_embeddings(self):
        tree = self.handler.generate(['1', '2', '3', '4'])
        self.assertTrue(isinstance(tree, dict))

    def test_insert(self):
        self.handler.storage.save_embedding('1', np.zeros(384))
        self.handler.storage.save_embedding('2', np.ones(384))
        self.handler.storage.save_embedding('3', np.arange(384))
        self.handler.storage.save_embedding('4', np.ones(384) * 2)
        tree = self.handler.generate(['1', '2', '3'])
        new_tree = self.handler.insert(tree, ['4'])
        self.assertGreater(
            len(Tree.from_dict(new_tree).adj_list),
            len(Tree.from_dict(tree).adj_list)
        )

    def test_score(self):
        self.handler.storage.save_embedding('1', np.zeros(384))
        self.handler.storage.save_embedding('2', np.ones(384))
        self.handler.storage.save_embedding('3', np.arange(384))
        tree = self.handler.generate(['1', '2', '3'])
        score = self.handler.score(tree)
        self.assertGreater(score, 0)


if __name__ == '__main__':
    unittest.main()
