import os
from numbers import Integral
from itertools import chain

import nltk
import numpy as np
import scipy.sparse as sp
import pandas as pd
from gensim import corpora, matutils

from Orange.data import Table, SparseTable, SparseTableSeries, TablePanel, Domain, ContinuousVariable, TableBase
from orangecontrib.text.vectorization import CountVectorizer


def get_sample_corpora_dir():
    path = os.path.dirname(__file__)
    directory = os.path.join(path, 'datasets')
    return os.path.abspath(directory)


def _check_arrays(*arrays):
    for a in arrays:
        if not (a is None or isinstance(a, np.ndarray) or sp.issparse(a)):
            raise TypeError('Argument {} should be of type np.array, sparse or None.'.format(a))

    lengths = set(a.shape[0] for a in arrays if a is not None)
    if len(lengths) > 1:
        raise ValueError('Leading dimension mismatch')

    return lengths.pop() if len(lengths) else 0


class Corpus(SparseTable):
    """Internal class for storing a corpus."""

    _metadata = SparseTable._metadata + ["text_features", "_dictionary", "ngram_range",
                                         "_ngrams_corpus_names"]

    # analogous to TableBase._WEIGHTS_COLUMN
    _TOKENS_COLUMN = "__tokens__"
    _NGRAMS_CORPUS_COLUMN = "__ngrams_corpus__"
    _INTERNAL_COLUMN_NAMES = SparseTable._INTERNAL_COLUMN_NAMES + \
                             [_TOKENS_COLUMN, _NGRAMS_CORPUS_COLUMN]

    @property
    def _constructor(self):
        return Corpus

    @property
    def _constructor_sliced(self):
        return CorpusSeries

    @property
    def _constructor_expanddim(self):
        return TablePanel

    def __new__(cls, *args, **kwargs):
        # see TableBase.__new__ for comments on why this is needed
        all_kwargs_are_pandas = len(set(kwargs.keys()).difference(cls.KNOWN_PANDAS_KWARGS)) == 0
        if len(args) == 1 and (isinstance(args[0], pd.core.internals.BlockManager)
                               or (isinstance(args[0], np.ndarray) and len(kwargs) != 0 and all_kwargs_are_pandas)):
            return cls(data=args[0], **kwargs)
        # we break the standard contract with text_features passed as a positional arg
        # we omit it from upstream so it doesn't get interpreted as weights
        if len(args) == 5:  # text_features is the fifth arg
            return super().__new__(cls, *(args[:4]), **kwargs)
        else:
            return super().__new__(cls, *args, **kwargs)

    def __init__(self, domain=None, X=None, Y=None, metas=None, text_features=None, **kwargs):
        """
        Args:
            X (numpy.ndarray): attributes
            Y (numpy.ndarray): class variables
            metas (numpy.ndarray): meta attributes; e.g. text
            domain (Orange.data.domain): the domain for this Corpus
            text_features (list): meta attributes that are used for
                text mining. Infer them if None.
        """
        n_doc = _check_arrays(X, Y, metas)

        # self.X = X if X is not None else np.zeros((n_doc, 0))
        # self.Y = Y if Y is not None else np.zeros((n_doc, 0))
        # self.metas = metas if metas is not None else np.zeros((n_doc, 0))
        # self.W = np.zeros((n_doc, 0))
        # self.domain = domain
        self.text_features = None  # list of text features for mining
        self._dictionary = None
        self.ngram_range = (1, 1)
        self._ngrams_corpus_names = []

        super().__init__(domain, X, Y, metas, **kwargs)
        if self._TOKENS_COLUMN not in self.columns:
            self[self._TOKENS_COLUMN] = np.nan
        if self._NGRAMS_CORPUS_COLUMN not in self.columns:
            self[self._NGRAMS_CORPUS_COLUMN] = np.nan

        # if we are being called from pandas internals
        all_kwargs_are_pandas = len(set(kwargs.keys()).difference(self.KNOWN_PANDAS_KWARGS)) == 0
        # we may been to transfer some properties if copying from another table,
        # because we need the domain later on
        if kwargs and all_kwargs_are_pandas and 'data' in kwargs and isinstance(kwargs['data'], TableBase):
            self._transfer_properties(kwargs['data'], transfer_domain=True)

        if self.domain is not None and text_features is None:
            self._infer_text_features()
        elif self.domain is not None:
            self.set_text_features(text_features)

        self.index = self._new_id(len(self), force_list=True)

    @property
    def tokens(self):
        """
        np.ndarray: Return a list of lists containing tokens. If tokens are not yet
        present, run default preprocessor and save tokens.
        """
        if self[self._TOKENS_COLUMN].isnull().all():
            self._apply_base_preprocessor()
        return self[self._TOKENS_COLUMN].values.values

    def store_tokens(self, tokens, dictionary=None):
        """
        Args:
            tokens (list): List of lists containing tokens.
        """
        self[self._TOKENS_COLUMN] = pd.SparseSeries(tokens, index=self.index)
        self._dictionary = dictionary or corpora.Dictionary(self.tokens)

    @property
    def ngrams_corpus(self):
        if not self._ngrams_corpus_names:
            CountVectorizer().transform(self)
        subset = self[self._ngrams_corpus_names]
        return matutils.Sparse2Corpus(subset.X.T)
        #return self[self._NGRAMS_CORPUS_COLUMN].values.values

    def store_ngrams_corpus(self, names):
        # not a property setter because pandas overrides __setattr__ and
        # that precedes property setters (and it fails)
        self._ngrams_corpus_names = names

    @property
    def ngrams(self):
        return self.ngrams_iterator(join_with=' ')

    def ngrams_iterator(self, join_with=' '):
        if join_with is None:
            return (list(chain(*(nltk.ngrams(doc, n)
                                 for n in range(self.ngram_range[0], self.ngram_range[1] + 1))))
                    for doc in self.tokens)
        else:
            return (list(chain(*((join_with.join(ngram) for ngram in nltk.ngrams(doc, n))
                                 for n in range(self.ngram_range[0], self.ngram_range[1] + 1))))
                    for doc in self.tokens)

    def set_text_features(self, feats):
        """
        Select which meta-attributes to include when mining text.

        Args:
            feats (list): list of text features to include.
        """
        for f in feats:
            if f not in chain(self.domain.variables, self.domain.metas):
                raise ValueError('Feature "{}" not found.'.format(f))
        if len(set(feats)) != len(feats):
            raise ValueError('Text features must be unique.')
        self.text_features = feats
        self[self._TOKENS_COLUMN] = np.nan  # invalidate tokens

    def _infer_text_features(self):
        """
        Infer which text features to use. If nothing was provided
        in the file header, use the first text feature.
        """
        include_feats = []
        first = None
        for attr in self.domain.metas:
            if attr.is_string:
                if first is None:
                    first = attr
                if attr.attributes.get('include', 'False') == 'True':
                    include_feats.append(attr)
        if len(include_feats) == 0 and first:
            include_feats.append(first)
        self.set_text_features(include_feats)

    def extend_corpus(self, metadata, Y):
        """
        Append documents to corpus, returning a new corpus.

        Args:
            metadata (numpy.ndarray): Meta data
            Y (numpy.ndarray): Class variables
        """
        # assert metas are 2D, y is 1D
        num_docs = len(Y)

        cv = self.domain.class_var
        for val in set(Y):
            if val not in cv.values:
                cv.add_value(val)

        new_index = self._new_id(num_docs, force_list=True)
        new = Corpus(index=new_index, columns=self.columns)
        new[self.domain.class_var.name] = Y
        for i, meta in enumerate(self.domain.metas):
            new[meta] = metadata[:, i]

        result = pd.concat([self, new], axis=0, copy=False)
        result[self._TOKENS_COLUMN] = np.nan
        # pandas concatenation doesn't transfer properties
        result._transfer_properties(self, transfer_domain=True)
        return result

    def extend_attributes(self, X, feature_names, var_attrs=None):
        """
        Append features to corpus.

        Args:
            X (numpy.ndarray or scipy.sparse.csr_matrix): Features to append
            feature_names (list): List of string containing feature names
            var_attrs (dict): Additional attributes appended to variable.attributes.
        """
        X = X.tocsc() if sp.issparse(X) else sp.csc_matrix(X)
        for i, name in enumerate(feature_names):
            self[name] = X[:, i].toarray().T[0]

        new_attr = self.domain.attributes
        for f in feature_names:
            var = ContinuousVariable.make(f)
            if isinstance(var_attrs, dict):
                var.attributes.update(var_attrs)
            new_attr += (var, )

        new_domain = Domain(
            attributes=new_attr,
            class_vars=self.domain.class_vars,
            metas=self.domain.metas
        )
        self.domain = new_domain

    @property
    def documents(self):
        """
        Returns: a list of strings representing documents — created by joining
            selected text features.
        """
        return self.documents_from_features(self.text_features)

    def documents_from_features(self, feats):
        """
        Args:
            feats (list): A list of features to join.

        Returns: a list of strings constructed by joining feats.
        """
        # create a Table where feats are in metas
        # automatically transformed into dense (because Table is dense)
        vars = [self.domain[f] for f in feats]
        return [' '.join(v.str_val(s.iloc[i])
                         for s, v in ((self[v], v)
                                      for v in vars))
                for i in range(len(self))]

    def _apply_base_preprocessor(self):
        from orangecontrib.text.preprocess import base_preprocessor
        corpus = base_preprocessor(self)
        self.store_tokens(corpus.tokens, corpus.dictionary)

    @property
    def dictionary(self):
        """
        corpora.Dictionary: A token to id mapper.
        """
        if self._dictionary is None:
            self._apply_base_preprocessor()
        return self._dictionary

    @classmethod
    def from_corpus(cls, domain, source, row_indices=...):
        c = cls.from_table(domain, source, row_indices)
        c.text_features = source.text_features
        return c

    @classmethod
    def from_file(cls, filename):
        if not os.path.exists(filename):    # check the default location
            abs_path = os.path.join(get_sample_corpora_dir(), filename)
            if not abs_path.endswith('.tab'):
                abs_path += '.tab'
            if not os.path.exists(abs_path):
                raise FileNotFoundError('File "{}" not found.'.format(filename))
            else:
                filename = abs_path
        return super().from_file(filename)

    def _equal_dense_or_sparse(self, left, right):
        if sp.issparse(left) and sp.issparse(right):
            return left.shape == right.shape and np.allclose(left.data, right.data)
        else:
            return np.array_equal(left, right)

    def __eq__(self, other):
        return (self.text_features == other.text_features and
                self._dictionary == other._dictionary and
                self._equal_dense_or_sparse(self.X, other.X) and
                self._equal_dense_or_sparse(self.Y, other.Y) and
                self._equal_dense_or_sparse(self.metas, other.metas) and
                self._equal_dense_or_sparse(self.tokens, other.tokens) and
                self.domain == other.domain and
                self.ngram_range == other.ngram_range)

    def __ne__(self, other):
        return not self.__eq__(other)


class CorpusSeries(SparseTableSeries):
    @property
    def _constructor(self):
        return CorpusSeries

    @property
    def _constructor_expanddim(self):
        return Corpus
