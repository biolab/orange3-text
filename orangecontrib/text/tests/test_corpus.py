import os
import unittest
import numpy as np

from Orange.data.domain import Domain, StringVariable

from orangecontrib.text.corpus import Corpus


class CorpusTests(unittest.TestCase):
    def test_corpus_from_file(self):
        c = Corpus.from_file('bookexcerpts')
        self.assertEqual(len(c), 140)
        self.assertEqual(len(c.domain), 1)
        self.assertEqual(len(c.domain.metas), 1)
        self.assertEqual(c.metas.shape, (140, 1))

    def test_corpus_from_file_abs_path(self):
        c = Corpus.from_file('bookexcerpts')
        path = os.path.dirname(__file__)
        file = os.path.abspath(os.path.join(path, '..', 'datasets', 'bookexcerpts.tab'))
        c2 = Corpus.from_file(file)
        self.assertEqual(c, c2)

    def test_corpus_from_file_with_tab(self):
        c = Corpus.from_file('bookexcerpts')
        c2 = Corpus.from_file('bookexcerpts.tab')
        self.assertEqual(c, c2)

    def test_corpus_from_file_missing(self):
        with self.assertRaises(FileNotFoundError):
            Corpus.from_file('missing_file')

    def test_corpus_from_init(self):
        c = Corpus.from_file('bookexcerpts')
        c2 = Corpus(c.documents, c.X, c.Y, c.metas, c.domain)
        self.assertEqual(c, c2)

    def test_corpus_from_file(self):
        c = Corpus.from_file('deerwester')
        self.assertEqual(len(c), 9)
        self.assertEqual(len(c.domain), 1)
        self.assertEqual(len(c.domain.metas), 1)
        self.assertEqual(c.metas.shape, (9, 1))

    def test_extend_corpus(self):
        c = Corpus.from_file('bookexcerpts')
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

    def test_corpus_not_eq(self):
        c = Corpus.from_file('bookexcerpts')
        c2 = Corpus(c.documents[:-1], c.X, c.Y, c.metas, c.domain)
        self.assertNotEqual(c, c2)
        c2 = Corpus(c.documents, np.vstack((c.X, c.X)), c.Y, c.metas, c.domain)
        self.assertNotEqual(c, c2)
        c2 = Corpus(c.documents, c.X, np.vstack((c.Y, c.Y)), c.metas, c.domain)
        self.assertNotEqual(c, c2)
        c2 = Corpus(c.documents, c.X, c.Y, c.metas.T, c.domain)
        self.assertNotEqual(c, c2)
        broken_domain = Domain(c.domain.attributes, c.domain.class_var, [StringVariable('text2')])
        c2 = Corpus(c.documents, c.X, c.Y, c.metas, broken_domain)
        self.assertNotEqual(c, c2)
