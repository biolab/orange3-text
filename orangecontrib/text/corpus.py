import os
from copy import copy
from numbers import Integral
from itertools import chain

import nltk
import numpy as np
import scipy.sparse as sp
from gensim import corpora

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
from orangecontrib.text.vectorization import BowVectorizer


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


class Corpus(Table):
    """Internal class for storing a corpus."""

    def __new__(cls, *args, **kwargs):
        """Bypass Table.__new__."""
        return object.__new__(cls)

    def __init__(self, domain=None, X=None, Y=None, metas=None, W=None, text_features=None):
        """
        Args:
            domain (Orange.data.Domain): the domain for this Corpus
            X (numpy.ndarray): attributes
            Y (numpy.ndarray): class variables
            metas (numpy.ndarray): meta attributes; e.g. text
            W (numpy.ndarray): instance weights
            text_features (list): meta attributes that are used for
                text mining. Infer them if None.
        """
        n_doc = _check_arrays(X, Y, metas)

        self.X = X if X is not None else np.zeros((n_doc, 0))
        self.Y = Y if Y is not None else np.zeros((n_doc, 0))
        self.metas = metas if metas is not None else np.zeros((n_doc, 0))
        self.W = W if W is not None else np.zeros((n_doc, 0))
        self.domain = domain
        self.text_features = None    # list of text features for mining
        self._tokens = None
        self._dictionary = None
        self._ngrams_corpus = None
        self.ngram_range = (1, 1)
        self.attributes = {}
        self.pos_tags = None

        if domain is not None and text_features is None:
            self._infer_text_features()
        elif domain is not None:
            self.set_text_features(text_features)

        Table._init_ids(self)

    def set_text_features(self, feats):
        """
        Select which meta-attributes to include when mining text.

        Args:
            feats (list or None): List of text features to include. If None infer them.
        """
        if feats is not None:
            for f in feats:
                if f not in chain(self.domain.variables, self.domain.metas):
                    raise ValueError('Feature "{}" not found.'.format(f))
            if len(set(feats)) != len(feats):
                raise ValueError('Text features must be unique.')
            self.text_features = feats
        else:
            self._infer_text_features()
        self._tokens = None     # invalidate tokens

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
        Append documents to corpus.

        Args:
            metadata (numpy.ndarray): Meta data
            Y (numpy.ndarray): Class variables
        """

        self.metas = np.vstack((self.metas, metadata))

        cv = self.domain.class_var
        for val in set(Y):
            if val not in cv.values:
                cv.add_value(val)
        new_Y = np.array([cv.to_val(i) for i in Y])[:, None]
        self._Y = np.vstack((self._Y, new_Y))

        self.X = self.W = np.zeros((len(self), 0))
        Table._init_ids(self)

        self._tokens = None     # invalidate tokens

    def extend_attributes(self, X, feature_names, var_attrs=None):
        """
        Append features to corpus.

        Args:
            X (numpy.ndarray or scipy.sparse.csr_matrix): Features to append
            feature_names (list): List of string containing feature names
            var_attrs (dict): Additional attributes appended to variable.attributes.
        """

        if self.X.size == 0:
            self.X = X
        elif sp.issparse(self.X) or sp.issparse(X):
            self.X = sp.hstack((self.X, X)).tocsr()
        else:
            self.X = np.hstack((self.X, X))

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

    @property
    def titles(self):
        """ Returns a list of titles. """
        attrs = [attr for attr in chain(self.domain.variables, self.domain.metas)
                 if attr.attributes.get('title', False)]
        # Alternatively, use heuristics
        if not attrs:
            for var in sorted(chain(self.domain.metas, self.domain),
                              key=lambda var: var.name,
                              reverse=True):  # reverse so that title < heading < filename
                if var.name.lower() in ('title', 'heading', 'h1', 'filename'):
                    attrs = [var]
                    break
        if attrs:
            return self.documents_from_features(attrs)
        else:
            return ['Document {}'.format(i+1) for i in range(len(self))]

    def documents_from_features(self, feats):
        """
        Args:
            feats (list): A list fo features to join.

        Returns: a list of strings constructed by joining feats.
        """
        # create a Table where feats are in metas
        data = Table(Domain([], [], [i.name for i in feats],
                            source=self.domain), self)

        # When we use only features coming from sparse X data.metas is sparse.
        # Transform it to dense.
        if sp.issparse(data.metas):
            data.metas = data.metas.toarray()

        return [' '.join(f.str_val(val) for f, val in zip(data.domain.metas, row))
                for row in data.metas]

    def store_tokens(self, tokens, dictionary=None):
        """
        Args:
            tokens (list): List of lists containing tokens.
        """
        self._tokens = np.array(tokens)
        self._dictionary = dictionary or corpora.Dictionary(self.tokens)

    @property
    def tokens(self):
        """
        np.ndarray: A list of lists containing tokens. If tokens are not yet
        present, run default preprocessor and save tokens.
        """
        if self._tokens is None:
            self._apply_base_preprocessor()
        return self._tokens

    def has_tokens(self):
        """ Return whether corpus is preprocessed or not. """
        return self._tokens is not None

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

    def ngrams_iterator(self, join_with=' ', include_postags=False):
        if self.pos_tags is None:
            include_postags = False

        if include_postags:
            data = zip(self.tokens, self.pos_tags)
        else:
            data = self.tokens

        if join_with is None:
            processor = lambda doc, n: nltk.ngrams(doc, n)
        elif include_postags:
            processor = lambda doc, n: (join_with.join(token + '_' + tag for token, tag in ngram)
                                        for ngram in nltk.ngrams(zip(*doc), n))
        else:
            processor = lambda doc, n: (join_with.join(ngram) for ngram in nltk.ngrams(doc, n))

        return (list(chain(*(processor(doc, n)
                for n in range(self.ngram_range[0], self.ngram_range[1]+1))))
                for doc in data)

    @property
    def ngrams_corpus(self):
        if self._ngrams_corpus is None:
            return BowVectorizer().transform(self).ngrams_corpus
        return self._ngrams_corpus

    @ngrams_corpus.setter
    def ngrams_corpus(self, value):
        self._ngrams_corpus = value

    @property
    def ngrams(self):
        """generator: Ngram representations of documents."""
        return self.ngrams_iterator(join_with=' ')

    def copy(self):
        """Return a copy of the table."""
        c = self.__class__(self.domain, self.X.copy(), self.Y.copy(), self.metas.copy(),
                           self.W.copy(), copy(self.text_features))
        # since tokens and dictionary are considered immutable copies are not needed
        c._tokens = self._tokens
        c._dictionary = self._dictionary
        c.ngram_range = self.ngram_range
        c.pos_tags = self.pos_tags
        c.name = self.name
        return c

    @staticmethod
    def from_documents(documents, name, attributes=None, class_vars=None, metas=None,
                       title_indices=None):
        """
        Create corpus from documents.

        Args:
            documents (list): List of documents.
            name (str): Name of the corpus
            attributes (list): List of tuples (Variable, getter) for attributes.
            class_vars (list): List of tuples (Variable, getter) for class vars.
            metas (list): List of tuples (Variable, getter) for metas.
            title_indices (list): List of indices into domain corresponding to features which will
                be used as titles.

        Returns:
            Corpus.
        """
        attributes = attributes or []
        class_vars = class_vars or []
        metas = metas or []
        title_indices = title_indices or []

        domain = Domain(attributes=[attr for attr, _ in attributes],
                        class_vars=[attr for attr, _ in class_vars],
                        metas=[attr for attr, _ in metas])

        for ind in title_indices:
            domain[ind].attributes['title'] = True

        for attr in domain.attributes:
            if isinstance(attr, DiscreteVariable):
                attr.values = []

        def to_val(attr, val):
            if isinstance(attr, DiscreteVariable):
                attr.val_from_str_add(val)
            return attr.to_val(val)

        X = np.array([[to_val(attr, func(doc)) for attr, func in attributes]
                      for doc in documents])
        Y = np.array([[to_val(attr, func(doc)) for attr, func in class_vars]
                      for doc in documents])
        metas = np.array([[to_val(attr, func(doc)) for attr, func in metas]
                          for doc in documents], dtype=object)

        corpus = Corpus(X=X, Y=Y, metas=metas, domain=domain, text_features=[])
        corpus.name = name
        return corpus

    def __getitem__(self, key):
        c = super().__getitem__(key)
        Corpus.retain_preprocessing(self, c, key)
        return c

    @classmethod
    def from_table(cls, domain, source, row_indices=...):
        t = super().from_table(domain, source, row_indices)
        c = Corpus(t.domain, t.X, t.Y, t.metas, t.W)
        Corpus.retain_preprocessing(source, c, row_indices)
        return c

    @classmethod
    def from_file(cls, filename):
        if not os.path.exists(filename):  # check the default location
            abs_path = os.path.join(get_sample_corpora_dir(), filename)
            if not abs_path.endswith('.tab'):
                abs_path += '.tab'
            if not os.path.exists(abs_path):
                raise FileNotFoundError('File "{}" not found.'.format(filename))
            else:
                filename = abs_path

        table = Table.from_file(filename)
        return cls(table.domain, table.X, table.Y, table.metas, table.W)

    @staticmethod
    def retain_preprocessing(orig, new, key=...):
        """ Set preprocessing of 'new' object to match the 'orig' object. """
        if isinstance(orig, Corpus):
            if orig._tokens is not None:  # retain preprocessing
                if isinstance(key, tuple):  # get row selection
                    key = key[0]
                if isinstance(key, Integral):
                    new._tokens = np.array([orig._tokens[key]])
                    new.pos_tags = None if orig.pos_tags is None else np.array(
                        [orig.pos_tags[key]])
                elif isinstance(key, list) or isinstance(key, np.ndarray) or isinstance(key,
                                                                                        slice):
                    new._tokens = orig._tokens[key]
                    new.pos_tags = None if orig.pos_tags is None else orig.pos_tags[key]
                elif key is Ellipsis:
                    new._tokens = orig._tokens
                    new.pos_tags = orig.pos_tags
                else:
                    raise TypeError('Indexing by type {} not supported.'.format(type(key)))
                new._dictionary = orig._dictionary

            new.text_features = orig.text_features
            new.ngram_range = orig.ngram_range
            new.attributes = orig.attributes

    def __len__(self):
        return len(self.metas)

    def __eq__(self, other):
        return (self.text_features == other.text_features and
                self._dictionary == other._dictionary and
                np.array_equal(self._tokens, other._tokens) and
                np.array_equal(self.X, other.X) and
                np.array_equal(self.Y, other.Y) and
                np.array_equal(self.metas, other.metas) and
                np.array_equal(self.pos_tags, other.pos_tags) and
                self.domain == other.domain and
                self.ngram_range == other.ngram_range)
