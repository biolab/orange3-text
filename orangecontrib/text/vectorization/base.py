import numpy as np
from gensim import matutils
from gensim.corpora import Dictionary


class BaseVectorizer:
    """Base class for vectorization objects. """
    name = NotImplemented

    def transform(self, corpus, copy=True):
        """Transforms a corpus to a new one with additional attributes. """
        if copy:
            corpus = corpus.copy()

        if not len(corpus.dictionary):
            return corpus
        else:
            return self._transform(corpus)

    def _transform(self, corpus):
        raise NotImplementedError

    def report(self):
        """Reports configuration items."""
        raise NotImplementedError

    @staticmethod
    def add_features(corpus, X, dictionary):
        order = np.argsort([dictionary[i] for i in range(len(dictionary))])
        corpus.extend_attributes(X[:, order],
                                 feature_names=(dictionary[i] for i in order),
                                 var_attrs={'hidden': True, 'skip-normalization': True})
        corpus.ngrams_corpus = matutils.Sparse2Corpus(X.T)
