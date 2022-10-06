import numpy as np

from Orange.data.util import SharedComputeValue
from Orange.util import dummy_callback


class BaseVectorizer:
    """Base class for vectorization objects. """
    name = NotImplemented

    def transform(self, corpus, copy=True, source_dict=None, callback=dummy_callback):
        """Transforms a corpus to a new one with additional attributes. """
        if copy:
            corpus = corpus.copy()
        return self._transform(corpus, source_dict, callback)

    def _transform(self, corpus, source_dict, callback):
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

        feature_names = [dictionary[i] for i in order]
        corpus = corpus.extend_attributes(
            X[:, order],
            feature_names=feature_names,
            var_attrs=variable_attrs,
            compute_values=compute_values,
            sparse=True,
            rename_existing=True
        )
        return corpus


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
        corpus = self.vectorizer.transform(corpus, **self.kwargs)

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

    def __setstate__(self, state):
        """
        Before orange3-text version 1.12.0 variable was wrongly set to current
        variable (variable that has this compute value attached) instead of
        original variable which caused fails after latest changes in core
        Orange. Since variable from VectorizationComputeValue is never used in
        practice we do not set it anymore (it is always None for
        VectorizationComputeValue).
        Anyway it is still set in pickles create before 1.12.0 and this line
        removes it when unpickling old pickles.
        """
        state["variable"] = None
        self.__dict__.update(state)
