import os
import unittest
from orangecontrib.text.corpus import Corpus


DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'datasets')


class CorpusTests(unittest.TestCase):
    def test_corpus_from_file(self):
        c = Corpus.from_file(os.path.join(DATASET_PATH, 'bookexcerpts.txt'))
        self.assertEqual(len(c), 140)

        self.assertEqual(len(c.domain), 0)
        self.assertEqual(len(c.domain.metas), 2)
        self.assertEqual(c.metas.shape, (140, 2))
