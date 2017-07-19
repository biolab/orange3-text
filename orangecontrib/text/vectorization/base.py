import numpy as np
from gensim import matutils

from Orange.data.util import SharedComputeValue


class BaseVectorizer:
    """Base class for vectorization objects. """
    name = NotImplemented

    def transform(self, corpus, copy=True, source_dict=None):
        """Transforms a corpus to a new one with additional attributes. """
        if not (len(corpus.dictionary) or source_dict) or not len(corpus):
            return corpus
        if copy:
            corpus = corpus.copy()
        return self._transform(corpus, source_dict)

    def _transform(self, corpus, source_dict):
        raise NotImplementedError

    def report(self):
        """Reports configuration items."""
        raise NotImplementedError

    @staticmethod
    def add_features(corpus, X, dictionary, compute_values=None, var_attrs=None):
        order = np.argsort([dictionary[i] for i in range(len(dictionary))])
        if compute_values is not None:
            compute_values = np.array(compute_values)[order]

        variable_attrs = {
            'hidden': True,
            'skip-normalization': True,
        }
        if isinstance(var_attrs, dict):
            variable_attrs.update(var_attrs)

        corpus.extend_attributes(X[:, order],
                                 feature_names=(dictionary[i] for i in order),
                                 var_attrs=variable_attrs,
                                 compute_values=compute_values,
                                 sparse=True)
        corpus.ngrams_corpus = matutils.Sparse2Corpus(X.T)


class SharedTransform:
    """ Shared computation for transforming new data sets.
    Used as a "shared" part within compute values. """
    def __init__(self, vectorizer, preprocessor=None, **kwargs):
        self.preprocessor = preprocessor
        self.vectorizer = vectorizer
        self.kwargs = kwargs

    def __call__(self, corpus):
        if callable(self.preprocessor):
            corpus = self.preprocessor(corpus)
        corpus = self.vectorizer.transform(corpus,
                                           copy=True,
                                           **self.kwargs)
        # store name to indices mapping so SharedComputeValue can run faster
        corpus.feature_name_to_index = {
            attr.name: i
            for i, attr in enumerate(corpus.domain.attributes)
        }
        return corpus


class VectorizationComputeValue(SharedComputeValue):
    """ Compute Value for vectorization features. """
    def __init__(self, compute_shared, name):
        super().__init__(compute_shared)
        self.name = name

    def compute(self, _, shared_data):
        ind = shared_data.feature_name_to_index[self.name]
        return shared_data.X[:, ind]
