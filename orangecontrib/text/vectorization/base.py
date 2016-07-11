import numpy as np
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
                                 var_attrs={'hidden': True})
        corpus.ngrams_dictionary = Dictionary()
        corpus.ngrams_dictionary.token2id = {dictionary[i]: j for j, i in enumerate(order)}
        corpus.ngrams_matrix = X[:, order]
