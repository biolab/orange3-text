import numpy as np
from gensim import corpora, matutils

from orangecontrib.text.vectorization.base import BaseVectorizer


class CountVectorizer(BaseVectorizer):
    name = 'Count Vectorizer'

    def __init__(self, binary=False):
        self.binary = binary

    def _transform(self, corpus):
        dic = corpora.Dictionary(corpus.ngrams, prune_at=None)
        X = matutils.corpus2csc(map(dic.doc2bow, corpus.ngrams)).T

        if self.binary:
            X[X > 1] = 1

        self.add_features(corpus, X, dic)
        return corpus

    def report(self):
        return ('Binary', self.binary),
