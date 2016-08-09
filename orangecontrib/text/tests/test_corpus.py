import os
import unittest
from distutils.version import LooseVersion

import numpy as np
import scipy.sparse as sp

import Orange
from Orange.data import Table, ContinuousVariable, Domain, StringVariable

from orangecontrib.text import preprocess
from orangecontrib.text.corpus import Corpus


class CorpusTests(unittest.TestCase):
    def test_corpus_from_file(self):
        c = Corpus.from_file('bookexcerpts')
        self.assertEqual(len(c), 140)
        self.assertEqual(len(c.domain), 1)
        self.assertEqual(len(c.domain.metas), 1)
        self.assertEqual(c.metas.shape, (140, 1))

        c = Corpus.from_file('deerwester')
        self.assertEqual(len(c), 9)
        self.assertEqual(len(c.domain), 1)
        self.assertEqual(len(c.domain.metas), 1)
        self.assertEqual(c.metas.shape, (9, 1))

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

    def test_extend_corpus(self):
        c = Corpus.from_file('bookexcerpts')
        n_classes = len(c.domain.class_var.values)
        c_copy = c.copy()
        new_y = np.array(c[c.domain.class_var])
        new_y[0] = 'teenager'
        c = c.extend_corpus(c.metas, new_y)

        self.assertEqual(len(c), len(c_copy)*2)
        self.assertEqual(c.Y.shape[0], c_copy.Y.shape[0]*2)
        self.assertEqual(c.metas.shape[0], c_copy.metas.shape[0]*2)
        self.assertEqual(c.metas.shape[1], c_copy.metas.shape[1])
        self.assertEqual(len(c_copy.domain.class_var.values), n_classes+1)

    def test_extend_attributes(self):
        # corpus without features
        c = Corpus.from_file('bookexcerpts')
        X = np.random.random((len(c), 3))
        c.extend_attributes(X, ['1', '2', '3'])
        self.assertEqual(c.X.shape, (len(c), 3))

        # add to non empty corpus
        c.extend_attributes(X, ['1', '2', '3'])
        self.assertEqual(c.X.shape, (len(c), 6))

        # extend sparse
        c.extend_attributes(sp.csr_matrix(X), ['1', '2', '3'])
        self.assertEqual(c.X.shape, (len(c), 9))
        self.assertTrue(sp.issparse(c.X))

    def test_corpus_not_eq(self):
        c = Corpus.from_file('bookexcerpts')
        n_doc = c.X.shape[0]

        c2 = c.copy()
        c2.set_text_features([])
        self.assertNotEqual(c, c2)

        c2 = c.copy()
        c2.domain = Domain([ContinuousVariable("foo")], c.domain.class_vars, c.domain.metas)
        c2["foo"] = np.ones(n_doc)
        self.assertNotEqual(c, c2)

        c2 = c.copy()
        c2[c2.domain.class_var] = [c2.domain.class_var.values[0]] * n_doc
        self.assertNotEqual(c, c2)

        broken_metas = np.copy(c.metas)
        broken_metas[0, 0] = ''
        c2 = c.copy()
        c2[c2.domain.metas[0]] = np.ravel(broken_metas)
        self.assertNotEqual(c, c2)

        c2 = c.copy()
        c2.ngram_range = (2, 4)
        self.assertNotEqual(c, c2)

    def test_from_table(self):
        t = Table.from_file('brown-selected')
        self.assertIsInstance(t, Table)

        c = Corpus.from_table(t.domain, t)
        self.assertIsInstance(c, Corpus)
        self.assertEqual(len(t), len(c))
        np.testing.assert_equal(t.metas, c.metas)
        self.assertEqual(c.text_features, [t.domain.metas[0]])

    def test_from_corpus(self):
        c = Corpus.from_file('bookexcerpts')
        c2 = Corpus.from_corpus(c.domain, c, row_indices=list(range(5)))
        self.assertEqual(len(c2), 5)

    def test_infer_text_features(self):
        c = Corpus.from_file('friends-transcripts')
        tf = c.text_features
        self.assertEqual(len(tf), 1)
        self.assertEqual(tf[0].name, 'Quote')

        c = Corpus.from_file('deerwester')
        tf = c.text_features
        self.assertEqual(len(tf), 1)
        self.assertEqual(tf[0].name, 'text')

    def test_documents(self):
        c = Corpus.from_file('bookexcerpts')
        docs = c.documents
        types = set(type(i) for i in docs)

        self.assertEqual(len(docs), len(c))
        self.assertEqual(len(types), 1)
        self.assertIn(str, types)

    def test_documents_from_features(self):
        c = Corpus.from_file('bookexcerpts')
        docs = c.documents_from_features([c.domain.class_var])
        types = set(type(i) for i in docs)

        self.assertTrue(all(
            [sum(cls in doc for cls in c.domain.class_var.values) == 1
             for doc in docs]))
        self.assertEqual(len(docs), len(c))
        self.assertEqual(len(types), 1)
        self.assertIn(str, types)

    @unittest.skipIf(LooseVersion(Orange.__version__) < LooseVersion('3.3.6'),
                     'Not supported in versions of Orange below 3.3.6')
    def test_documents_from_sparse_features(self):
        t = Table.from_file('brown-selected')
        c = Corpus.from_table(t.domain, t)

        # docs from X, Y and metas
        docs = c.documents_from_features([t.domain.attributes[0], t.domain.class_var, t.domain.metas[0]])
        self.assertEqual(len(docs), len(t))
        for first_attr, class_val, meta_attr, d in zip(t.X[:, 0], c.Y, c.metas[:, 0], docs):
            first_attr = c.domain.attributes[0].str_val(first_attr)
            class_val = c.domain.class_var.values[int(class_val)]
            meta_attr = c.domain.metas[0].str_val(meta_attr)
            self.assertIn(class_val, d)
            self.assertIn(first_attr, d)
            self.assertIn(meta_attr, d)

        # docs only from sparse X
        docs = c.documents_from_features([t.domain.attributes[0]])
        self.assertEqual(len(docs), len(t))
        for first_attr, d in zip(t.X[:, 0], docs):
            first_attr = c.domain.attributes[0].str_val(first_attr)
            self.assertIn(first_attr, d)

    def test_getitem(self):
        # does not currently work, because of a bug when slicing single
        # rows of multi-dtype sparsedataframes, pandas PR pending
        with self.assertRaises(Exception):
            c = Corpus.from_file('bookexcerpts')

            # run default preprocessing
            c.tokens

            sel = c.iloc[0]
            self.assertEqual(len(sel), 1)
            self.assertEqual(len(sel._tokens), 1)
            np.testing.assert_equal(sel._tokens, np.array([c._tokens[0]]))
            self.assertEqual(sel._dictionary, c._dictionary)

            sel = c.iloc[0:5]
            self.assertEqual(len(sel), 5)
            self.assertEqual(len(sel._tokens), 5)
            np.testing.assert_equal(sel._tokens, c._tokens[0:5])
            self.assertEqual(sel._dictionary, c._dictionary)

            ind = [3, 4, 5, 6]
            sel = c.iloc[ind]
            self.assertEqual(len(sel), len(ind))
            self.assertEqual(len(sel._tokens), len(ind))
            np.testing.assert_equal(sel._tokens, c._tokens[ind])
            self.assertEqual(sel._dictionary, c._dictionary)
            self.assertEqual(sel.text_features, c.text_features)
            self.assertEqual(sel.ngram_range, c.ngram_range)
            self.assertEqual(sel.attributes, c.attributes)

    def test_asserting_errors(self):
        c = Corpus.from_file('bookexcerpts')

        with self.assertRaises(AttributeError):
            # float has not attribute shape
            Corpus(c.domain, 1.0, c.Y, c.metas, c.text_features)

        too_large_y = np.vstack((c.Y, c.Y))
        with self.assertRaises(ValueError):
            Corpus(c.domain, c.X, too_large_y, c.metas, c.text_features)

        with self.assertRaises(ValueError):
            c.set_text_features([StringVariable('foobar')])

        with self.assertRaises(ValueError):
            c.set_text_features([c.domain.metas[0], c.domain.metas[0]])

        c.tokens    # preprocess
        with self.assertRaises(KeyError):
            c[..., 0]

    def test_copy(self):
        corpus = Corpus.from_file('deerwester')

        p = preprocess.Preprocessor(tokenizer=preprocess.RegexpTokenizer('\w+\s}'))
        copied = corpus.copy()
        p(copied, inplace=True)
        self.assertIsNot(copied, corpus)
        self.assertNotEqual(copied, corpus)

        p(corpus, inplace=True)
        copied = corpus.copy()
        self.assertIsNot(copied, corpus)
        self.assertEqual(copied, corpus)

    def test_ngrams_iter(self):
        c = Corpus.from_file('deerwester')
        c.ngram_range = (1, 1)
        self.assertEqual(list(c.ngrams), [doc.lower().split() for doc in c.documents])
        expected = [[(token.lower(), ) for token in doc.split()] for doc in c.documents]
        self.assertEqual(list(c.ngrams_iterator(join_with=None)), expected)
        c.ngram_range = (2, 3)

        expected_ngrams = [('machine', 'interface'), ('for', 'lab'),
                           ('machine', 'interface', 'for'), ('abc', 'computer', 'applications')]

        for ngram in expected_ngrams:
            self.assertIn(ngram, list(c.ngrams_iterator(join_with=None))[0])
            self.assertIn('-'.join(ngram), list(c.ngrams_iterator(join_with='-'))[0])
