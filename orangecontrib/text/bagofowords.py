from functools import partial
from collections import OrderedDict

import nltk
import numpy as np
from sklearn.preprocessing import normalize
from gensim import corpora, matutils, models
from simhash import Simhash

from orangecontrib.text.corpus import Corpus


__all__ = ['CountVectorizer', 'TfidfVectorizer', 'SimhashVectorizer']


class CountVectorizer:
    name = 'Count Vectorizer'

    def __init__(self, binary=False):
        self.binary = binary

    def transform(self, corpus):
        corpus = corpus.copy()
        dic = corpora.Dictionary(corpus.ngrams_iterator(join_with=' '), prune_at=None)
        X = matutils.corpus2csc(map(dic.doc2bow, corpus.ngrams_iterator(' '))).T

        if self.binary:
            X[X > 1] = 1

        order = np.argsort([dic[i] for i in range(len(dic))])
        corpus.extend_attributes(X[:, order],
                                 feature_names=(dic[i] for i in order),
                                 var_attrs={'hidden': True})
        return corpus

    def report(self):
        return ('Binary', self.binary),


class TfidfVectorizer:
    name = 'Tfidf Vectorizer'

    IDENTITY = 'Identity'
    SMOOTH = 'Smooth'
    BINARY = 'Binary'
    SUBLINEAR = 'Sublinear'
    NONE = '(None)'
    L1 = 'L1 (Sum of elements)'
    L2 = 'L2 (Euclidean)'

    norms = OrderedDict((
        (NONE, None),
        (L1, partial(normalize, norm='l1')),
        (L2, partial(normalize, norm='l2')),
    ))

    wlocals = OrderedDict((
        (IDENTITY, lambda x: x),
        (BINARY, lambda x: int(x > 0)),
        (SUBLINEAR, lambda x: 1 + np.log(x)),
    ))

    wglobals = OrderedDict((
        (IDENTITY, lambda idf, D: idf),
        (SMOOTH, lambda idf, D: idf + 1),
    ))

    def __init__(self, norm=NONE, wlocal=IDENTITY, wglobal=IDENTITY):
        self.norm = norm
        self.wlocal = wlocal
        self.wglobal = wglobal

    def transform(self, corpus):
        result = corpus.copy()

        corpus = list(result.ngrams_iterator(join_with=' '))
        dic = corpora.Dictionary(corpus, prune_at=None)
        corpus = [dic.doc2bow(doc) for doc in corpus]
        model = models.TfidfModel(corpus, normalize=False,
                                  wlocal=self.wlocals[self.wlocal],
                                  wglobal=self.wglobals[self.wglobal])

        X = matutils.corpus2csc(model[corpus], dtype=np.float).T
        norm = self.norms[self.norm]
        if norm:
            X = norm(X)

        order = np.argsort([dic[i] for i in range(len(dic))])
        result.extend_attributes(X[:, order],
                                 feature_names=(dic[i] for i in order),
                                 var_attrs={'hidden': True})
        return result

    def report(self):
        return (('Norm', self.norm),
                ('Tf transformation', self.wlocal),
                ('Idf transformation', self.wglobal))


class SimhashVectorizer:
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
        return Simhash(self.get_shingles(tokens, self.ngram_len), f=self.f, hashfunc=self.hashfunc).value

    def int2binarray(self, num):
        return [int(x) for x in self._bin_format.format(num)]

    def transform(self, corpus):
        """ Computes simhash values from the given corpus
        and creates a new one with a simhash attribute.

        Args:
            corpus (Corpus): a corpus with tokens.

        Returns:
            Corpus with `simhash` variable
        """

        X = np.array([self.int2binarray(self.compute_hash(doc)) for doc in corpus.tokens])
        corpus = corpus.copy()
        corpus.extend_attributes(X, ('simhash_{}'.format(i) for i in range(self.f)),
                                 var_attrs={'hidden': True})
        return corpus

    def fit_transform(self, corpus):
        return self.transform(corpus)

    def report(self):
        return (('Hash length', self.f),
                ('Shingle length', self.ngram_len))
