import os
import copy
import unittest
from orangecontrib.text.corpus import Corpus


DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'datasets')


class CorpusTests(unittest.TestCase):
    def test_corpus_from_file(self):
        c = Corpus.from_file(os.path.join(DATASET_PATH, 'bookexcerpts.tab'))
        self.assertEqual(len(c), 140)
        self.assertEqual(len(c.domain), 1)
        self.assertEqual(len(c.domain.metas), 1)
        self.assertEqual(c.metas.shape, (140, 1))

    def test_corpus_from_file_just_text(self):
        c = Corpus.from_file(os.path.join(DATASET_PATH, 'deerwester.tab'))

        self.assertEqual(len(c), 9)
        self.assertEqual(len(c.domain), 0)
        self.assertEqual(len(c.domain.metas), 1)
        self.assertEqual(c.metas.shape, (9, 1))

    def test_extend_corpus(self):
        c = Corpus.from_file(os.path.join(DATASET_PATH, 'bookexcerpts.tab'))
        n_classes = len(c.domain.class_var.values)
        c_copy = c.copy()
        new_y = [c.domain.class_var.values[int(i)] for i in c.Y]
        new_y[0] = 'teenager'
        c.extend_corpus(c.documents, c.metas, new_y)

        self.assertEqual(len(c), len(c_copy)*2)
        self.assertEqual(c.Y.shape[0], c_copy.Y.shape[0]*2)
        self.assertEqual(c.metas.shape[0], c_copy.metas.shape[0]*2)
        self.assertEqual(c.metas.shape[1], c_copy.metas.shape[1])
        self.assertEqual(len(c_copy.domain.class_var.values), n_classes+1)
