import os

import numpy as np

from Orange.data import Table


def get_sample_corpora_dir():
    path = os.path.dirname(__file__)
    directory = os.path.join(path, 'datasets')
    return os.path.abspath(directory)


def _check_arrays(*arrays):
    for a in arrays:
        if not (a is None or isinstance(a, np.ndarray)):
            raise TypeError('Argument {} should be of type np.array or None.'.format(a))

    lengths = set(len(a) for a in arrays if a is not None)
    if len(lengths) > 1:
        raise ValueError('Leading dimension mismatch')

    return lengths.pop() if len(lengths) else 0


class Corpus(Table):
    """Internal class for storing a corpus of orangecontrib.text.corpus.Corpus."""

    def __new__(cls, *args, **kwargs):
        """Bypass Table.__new__."""
        return object.__new__(cls)

    def __init__(self, X=None, Y=None, metas=None, domain=None, text_features=None):
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

        self.X = X if X is not None else np.zeros((n_doc, 0))
        self.Y = Y if Y is not None else np.zeros((n_doc, 0))
        self.metas = metas if metas is not None else np.zeros((n_doc, 0))
        self.W = np.zeros((n_doc, 0))
        self.domain = domain
        self.text_features = None    # list of text features for mining

        if domain and text_features is None:
            self._infer_text_features()
        elif domain:
            self.set_text_features(text_features)

        Table._init_ids(self)

    def set_text_features(self, feats):
        """
        Select which meta-attributes to include when mining text.

        Args:
            feats (list): list of text features to include.
        """
        for f in feats:
            if f not in self.domain.metas:
                raise ValueError('Feature "{}" not found in metas.'.format(f))
        if len(set(feats)) != len(feats):
            raise ValueError('Text features must be unique.')
        self.text_features = feats

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
        self.metas = np.vstack((self.metas, metadata))

        cv = self.domain.class_var
        for val in set(Y):
            if val not in cv.values:
                cv.add_value(val)
        new_Y = np.array([cv.to_val(i) for i in Y])[:, None]
        self._Y = np.vstack((self._Y, new_Y))

        self.X = self.W = np.zeros((len(self), 0))
        Table._init_ids(self)

    @property
    def documents(self):
        """
        Returns a list of strings representing documents.
        Each documents is created by joining selected text features.
        """
        indices = [self.domain.metas.index(f) for f in self.text_features]
        return [' '.join(map(str, i)) for i in self.metas[:, indices]]

    @classmethod
    def from_table(cls, domain, source, row_indices=...):
        t = super().from_table(domain, source, row_indices)
        return Corpus(t.X, t.Y, t.metas, t.domain, None)

    @classmethod
    def from_corpus(cls, domain, source, row_indices=...):
        c = cls.from_table(domain, source, row_indices)
        c.text_features = source.text_features
        return c

    @classmethod
    def from_file(cls, filename, wrapper=None):
        if not os.path.exists(filename):    # check the default location
            abs_path = os.path.join(get_sample_corpora_dir(), filename)
            if not abs_path.endswith('.tab'):
                abs_path += '.tab'
            if not os.path.exists(abs_path):
                raise FileNotFoundError('File "{}" not found.'.format(filename))
            else:
                filename = abs_path

        table = Table.from_file(filename, wrapper)
        return cls(table.X, table.Y, table.metas, table.domain, None)

    def copy(self):
        """Return a copy of the table."""
        c = self.__class__(self.X, self.Y, self.metas, self.domain, self.text_features)
        c.ensure_copy()
        return c

    def __len__(self):
        return len(self.metas)

    def __eq__(self, other):
        return (self.text_features == other.text_features and
                np.array_equal(self.X, other.X) and
                np.array_equal(self.Y, other.Y) and
                np.array_equal(self.metas, other.metas) and
                self.domain == other.domain)
