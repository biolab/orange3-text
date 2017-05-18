import numpy as np
from gensim import matutils
from gensim.corpora import Dictionary


class BaseVectorizer:
    """Base class for vectorization objects. """
    name = NotImplemented

    def transform(self, corpus, copy=True, source_dict=None):
        """Transforms a corpus to a new one with additional attributes. """
        if copy:
            corpus = corpus.copy()

        if not (len(corpus.dictionary) or source_dict) or not len(corpus):
            return corpus
        else:
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
                                 compute_values=compute_values)
        corpus.ngrams_corpus = matutils.Sparse2Corpus(X.T)
