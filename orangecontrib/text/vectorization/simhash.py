import nltk
from simhash import Simhash
import numpy as np

from orangecontrib.text.vectorization.base import BaseVectorizer


class SimhashVectorizer(BaseVectorizer):
    name = "Simhash"
    max_f = 1024

    def __init__(self, shingle_len=10, f=64, hashfunc=None):
        """
        Args:
            shingle_len(int): Length of a shingle.
            f(int): Length of a document fingerprints
            hashfunc(callable): A function that accepts a string and returns
                a unsigned integer
        """
        self.f = f
        self._bin_format = '{:0%db}' % self.f
        self.hashfunc = hashfunc
        self.ngram_len = shingle_len

    @staticmethod
    def get_shingles(tokens, n):
        return map(lambda x: ''.join(x), nltk.ngrams(tokens, n))

    def compute_hash(self, tokens):
        values = self.get_shingles(tokens, self.ngram_len)
        if self.hashfunc is None:
            return Simhash(values, f=self.f).value
        else:
            return Simhash(values, f=self.f, hashfunc=self.hashfunc).value

    def int2binarray(self, num):
        return [int(x) for x in self._bin_format.format(num)]

    def _transform(self, corpus, source_dict):
        """ Computes simhash values from the given corpus
        and creates a new one with a simhash attribute.

        Args:
            corpus (Corpus): a corpus with tokens.

        Returns:
            Corpus with `simhash` variable
        """

        X = np.array([self.int2binarray(self.compute_hash(doc)) for doc in corpus.tokens], dtype=np.float)
        corpus = corpus.extend_attributes(
            X,
            feature_names=[
                'simhash_{}'.format(int(i) + 1) for i in range(self.f)
            ],
            var_attrs={'hidden': True}
        )
        return corpus

    def report(self):
        return (('Hash length', self.f),
                ('Shingle length', self.ngram_len))
