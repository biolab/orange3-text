from collections import OrderedDict
from functools import partial

import numpy as np
from gensim import corpora, models, matutils
from sklearn.preprocessing import normalize

from orangecontrib.text.vectorization.base import BaseVectorizer


class TfidfVectorizer(BaseVectorizer):
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
        (IDENTITY, lambda tf: tf),
        (BINARY, lambda tf: int(tf > 0)),
        (SUBLINEAR, lambda tf: 1 + np.log(tf)),
    ))

    wglobals = OrderedDict((
        (IDENTITY, lambda df, N: np.log(N/df)),
        (SMOOTH, lambda df, N: np.log(1 + N/df)),
    ))

    def __init__(self, norm=NONE, wlocal=IDENTITY, wglobal=IDENTITY):
        self.norm = norm
        self.wlocal = wlocal
        self.wglobal = wglobal

    def _transform(self, corpus):
        if corpus.pos_tags is None:
            temp_corpus = list(corpus.ngrams)
        else:
            temp_corpus = list(corpus.ngrams_iterator(' ', include_postags=True))

        dic = corpora.Dictionary(temp_corpus, prune_at=None)
        temp_corpus = [dic.doc2bow(doc) for doc in temp_corpus]
        model = models.TfidfModel(temp_corpus, normalize=False,
                                  wlocal=self.wlocals[self.wlocal],
                                  wglobal=self.wglobals[self.wglobal])

        X = matutils.corpus2csc(model[temp_corpus], dtype=np.float).T
        norm = self.norms[self.norm]
        if norm:
            X = norm(X)

        self.add_features(corpus, X, dic)
        return corpus

    def report(self):
        return (('Norm', self.norm),
                ('Tf transformation', self.wlocal),
                ('Idf transformation', self.wglobal))
