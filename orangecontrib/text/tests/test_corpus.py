import os
import unittest
from distutils.version import LooseVersion

import numpy as np
from scipy.sparse import csr_matrix, issparse

import Orange
from Orange.data import Table, DiscreteVariable, StringVariable, Domain, ContinuousVariable

from orangecontrib.text import preprocess
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.tag import AveragedPerceptronTagger


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.pos_tagger = AveragedPerceptronTagger()

    def test_init_preserve_shape_of_empty_x(self):
        c = Corpus.from_file('book-excerpts')
        d = c.domain
        new_domain = Domain((ContinuousVariable('c1'),), d.class_vars, d.metas)

        empty_X = csr_matrix((len(c), 1))
        new = Corpus(new_domain, X=empty_X, Y=c.Y, metas=c.metas)

        self.assertEqual(empty_X.nnz, 0)
        self.assertEqual(new.X.shape, empty_X.shape)

    def test_corpus_from_file(self):
        c = Corpus.from_file('book-excerpts')
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
        c = Corpus.from_file('book-excerpts')
        path = os.path.dirname(__file__)
        file = os.path.abspath(os.path.join(path, '..', 'datasets', 'book-excerpts.tab'))
        c2 = Corpus.from_file(file)
        self.assertEqual(c, c2)

    def test_corpus_from_file_with_tab(self):
        c = Corpus.from_file('book-excerpts')
        c2 = Corpus.from_file('book-excerpts.tab')
        self.assertEqual(c, c2)

    def test_corpus_from_file_missing(self):
        with self.assertRaises(FileNotFoundError):
            Corpus.from_file('missing_file')

    def test_corpus_from_init(self):
        c = Corpus.from_file('book-excerpts')
        c2 = Corpus(c.domain, c.X, c.Y, c.metas, c.text_features)
        self.assertEqual(c, c2)

    @unittest.skipIf(LooseVersion(Orange.__version__) < LooseVersion('3.4.3'),
                     'Not supported in versions of Orange below 3.4.3')
    def test_extend(self):
        c = Corpus.from_file('deerwester')
        c2 = c[:5]
        self.assertEqual(len(c2), 5)
        n = len(c)
        self.pos_tagger.tag_corpus(c)
        self.assertIsNot(c._tokens, None)
        self.assertIsNot(c.pos_tags, None)
        self.assertIs(c2._tokens, None)
        self.assertIs(c2.pos_tags, None)

        c.extend(c2)
        self.assertEqual(len(c), n + 5)
        self.assertIs(c._tokens, None)
        self.assertIs(c.pos_tags, None)

        self.pos_tagger.tag_corpus(c)
        self.pos_tagger.tag_corpus(c2)
        c.extend(c2)
        self.assertEqual(len(c), n + 10)
        self.assertEqual(len(c._tokens), n + 10)
        self.assertEqual(len(c.pos_tags), n + 10)

    def test_extend_corpus(self):
        c = Corpus.from_file('book-excerpts')
        n_classes = len(c.domain.class_var.values)
        c_copy = c.copy()
        new_y = [c.domain.class_var.values[int(i)] for i in c.Y]
        new_y[0] = 'teenager'
        c.extend_corpus(c.metas, new_y)

        self.assertEqual(len(c), len(c_copy)*2)
        self.assertEqual(c.Y.shape[0], c_copy.Y.shape[0]*2)
        self.assertEqual(c.metas.shape[0], c_copy.metas.shape[0]*2)
        self.assertEqual(c.metas.shape[1], c_copy.metas.shape[1])
        self.assertEqual(len(c_copy.domain.class_var.values), n_classes+1)

    def test_extend_corpus_non_empty_X(self):
        c = Corpus.from_file('election-tweets-2016')[:10]
        with self.assertRaises(ValueError):
            c.extend_corpus(c.metas, c.Y)

    def test_extend_attributes(self):
        # corpus without features
        c = Corpus.from_file('book-excerpts')
        X = np.random.random((len(c), 3))
        c.extend_attributes(X, ['1', '2', '3'])
        self.assertEqual(c.X.shape, (len(c), 3))

        # add to non empty corpus
        c.extend_attributes(X, ['1', '2', '3'])
        self.assertEqual(c.X.shape, (len(c), 6))

        # extend sparse
        c.extend_attributes(csr_matrix(X), ['1', '2', '3'])
        self.assertEqual(c.X.shape, (len(c), 9))
        self.assertTrue(issparse(c.X))

    def test_corpus_not_eq(self):
        c = Corpus.from_file('book-excerpts')
        n_doc = c.X.shape[0]

        c2 = Corpus(c.domain, c.X, c.Y, c.metas, c.W, [])
        self.assertNotEqual(c, c2)

        c2 = Corpus(c.domain, np.ones((n_doc, 1)), c.Y, c.metas, c.W, c.text_features)
        self.assertNotEqual(c, c2)

        c2 = Corpus(c.domain, c.X, np.ones((n_doc, 1)), c.metas, c.W, c.text_features)
        self.assertNotEqual(c, c2)

        broken_metas = np.copy(c.metas)
        broken_metas[0, 0] = ''
        c2 = Corpus(c.domain, c.X, c.Y, broken_metas, c.W, c.text_features)
        self.assertNotEqual(c, c2)

        new_meta = [StringVariable('text2')]
        broken_domain = Domain(c.domain.attributes, c.domain.class_var, new_meta)
        c2 = Corpus(broken_domain, c.X, c.Y, c.metas, c.W, new_meta)
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

    def test_infer_text_features(self):
        c = Corpus.from_file('friends-transcripts')
        tf = c.text_features
        self.assertEqual(len(tf), 1)
        self.assertEqual(tf[0].name, 'Quote')

        c = Corpus.from_file('deerwester')
        tf = c.text_features
        self.assertEqual(len(tf), 1)
        self.assertEqual(tf[0].name, 'Text')

    def test_documents(self):
        c = Corpus.from_file('book-excerpts')
        docs = c.documents
        types = set(type(i) for i in docs)

        self.assertEqual(len(docs), len(c))
        self.assertEqual(len(types), 1)
        self.assertIn(str, types)

    def test_titles(self):
        c = Corpus.from_file('book-excerpts')

        # no title feature set
        titles = c.titles
        self.assertEqual(len(titles), len(c))
        for title in titles:
            self.assertIn('Document ', title)

        # inferred title from heuristics
        expected = list(map(str, range(len(c))))
        c2 = Corpus(Domain([], [], (StringVariable('heading'),)),
                    None, None, np.c_[expected])
        titles = c2.titles
        self.assertEqual(titles, expected)

        # title feature set
        c.domain[0].attributes['title'] = True
        titles = c.titles
        self.assertEqual(len(titles), len(c))
        for title in titles:
            self.assertIn(title, c.domain.class_var.values)

    def test_documents_from_features(self):
        c = Corpus.from_file('book-excerpts')
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
        c.X = csr_matrix(c.X)

        # docs from X, Y and metas
        docs = c.documents_from_features([t.domain.attributes[0], t.domain.class_var, t.domain.metas[0]])
        self.assertEqual(len(docs), len(t))
        for first_attr, class_val, meta_attr, d in zip(t.X[:, 0], c.Y, c.metas[:, 0], docs):
            first_attr = c.domain.attributes[0].str_val(first_attr)
            class_val = c.domain.class_var.str_val(class_val)
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
        c = Corpus.from_file('book-excerpts')

        # without preprocessing
        self.assertEqual(len(c[:, :]), len(c))

        # run default preprocessing
        c.tokens

        sel = c[:, :]
        self.assertEqual(sel, c)

        sel = c[0]
        self.assertEqual(len(sel), 1)
        self.assertEqual(len(sel._tokens), 1)
        np.testing.assert_equal(sel._tokens, np.array([c._tokens[0]]))
        self.assertEqual(sel._dictionary, c._dictionary)

        sel = c[0:5]
        self.assertEqual(len(sel), 5)
        self.assertEqual(len(sel._tokens), 5)
        np.testing.assert_equal(sel._tokens, c._tokens[0:5])
        self.assertEqual(sel._dictionary, c._dictionary)

        ind = [3, 4, 5, 6]
        sel = c[ind]
        self.assertEqual(len(sel), len(ind))
        self.assertEqual(len(sel._tokens), len(ind))
        np.testing.assert_equal(sel._tokens, c._tokens[ind])
        self.assertEqual(sel._dictionary, c._dictionary)
        self.assertEqual(sel.text_features, c.text_features)
        self.assertEqual(sel.ngram_range, c.ngram_range)
        self.assertEqual(sel.attributes, c.attributes)

        sel = c[...]
        self.assertEqual(sel, c)

    def test_set_text_features(self):
        c = Corpus.from_file('friends-transcripts')[:100]
        c2 = c.copy()
        self.assertEqual(c.set_text_features(None), c2._infer_text_features())

    def test_asserting_errors(self):
        c = Corpus.from_file('book-excerpts')

        with self.assertRaises(TypeError):
            Corpus(1.0, c.Y, c.metas, c.domain, c.text_features)

        too_large_x = np.vstack((c.X, c.X))
        with self.assertRaises(ValueError):
            Corpus(c.domain, too_large_x, c.Y, c.metas, c.W, c.text_features)

        with self.assertRaises(ValueError):
            c.set_text_features([StringVariable('foobar')])

        with self.assertRaises(ValueError):
            c.set_text_features([c.domain.metas[0], c.domain.metas[0]])

    def test_has_tokens(self):
        corpus = Corpus.from_file('deerwester')

        self.assertFalse(corpus.has_tokens())
        corpus.tokens   # default tokenizer
        self.assertTrue(corpus.has_tokens())

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

        self.pos_tagger.tag_corpus(c)
        c.ngram_range = (1, 1)
        for doc in c.ngrams_iterator(join_with='_', include_postags=True):
            for token in doc:
                self.assertRegexpMatches(token, '\w+_[A-Z]+')

    def test_from_documents(self):
        documents = [
            {
                'wheels': 4,
                'engine': 'w4',
                'type': 'car',
                'desc': 'A new car.'
            },
            {
                'wheels': 8.,
                'engine': 'w8',
                'type': 'truck',
                'desc': 'An old truck.'
            },
            {
                'wheels': 12.,
                'engine': 'w12',
                'type': 'truck',
                'desc': 'An new truck.'
            }
        ]

        attrs = [
            (DiscreteVariable('Engine'), lambda doc: doc.get('engine')),
            (ContinuousVariable('Wheels'), lambda doc: doc.get('wheels')),
        ]

        class_vars = [
            (DiscreteVariable('Type'), lambda doc: doc.get('type')),
        ]

        metas = [
            (StringVariable('Description'), lambda doc: doc.get('desc')),
        ]

        dataset_name = 'TruckData'
        c = Corpus.from_documents(documents, dataset_name, attrs, class_vars, metas)

        self.assertEqual(len(c), len(documents))
        self.assertEqual(c.name, dataset_name)
        self.assertEqual(len(c.domain.attributes), len(attrs))
        self.assertEqual(len(c.domain.class_vars), len(class_vars))
        self.assertEqual(len(c.domain.metas), len(metas))

        engine_dv = c.domain.attributes[0]
        self.assertEqual(sorted(engine_dv.values),
                         sorted([d['engine'] for d in documents]))
        self.assertEqual([engine_dv.repr_val(v) for v in c.X[:, 0]],
                         [d['engine'] for d in documents])

    def test_corpus_remove_text_features(self):
        """
        Remove those text features which do not have a column in metas.
        GH-324
        GH-325
        """
        c = Corpus.from_file('deerwester')
        domain = Domain(attributes=c.domain.attributes, class_vars=c.domain.class_vars)
        d = c.transform(domain)
        self.assertFalse(len(d.text_features))
        # Make sure that copying works.
        d.copy()
