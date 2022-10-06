""" This module constructs a new corpus with tokens as features.

First create a corpus::

    >>> from orangecontrib.text import Corpus
    >>> corpus = Corpus.from_file('deerwester')
    >>> corpus.domain
    [ | Category] {Text}

Then create :class:`BowVectorizer` object and call transform:

    >>> from orangecontrib.text.vectorization.bagofwords import BowVectorizer
    >>> bow = BowVectorizer()
    >>> new_corpus = bow.transform(corpus)
    >>> new_corpus.domain
    [a, abc, and, applications, binary, computer, engineering, eps, error, for,
    generation, graph, human, in, interface, intersection, iv, lab, machine,
    management, measurement, minors, of, opinion, ordering, paths, perceived,
    quasi, random, relation, response, survey, system, testing, the, time, to,
    trees, unordered, user, well, widths | Category] {Text}

"""

from collections import OrderedDict
from functools import partial

import numpy as np
from Orange.util import dummy_callback
from gensim import corpora, models, matutils
from sklearn.preprocessing import normalize

from orangecontrib.text.vectorization.base import BaseVectorizer,\
    SharedTransform, VectorizationComputeValue


class BowVectorizer(BaseVectorizer):
    name = 'BoW Vectorizer'

    COUNT = 'Count'
    BINARY = 'Binary'
    SUBLINEAR = 'Sublinear'
    NONE = '(None)'
    IDF = 'IDF'
    SMOOTH = 'Smooth IDF'
    L1 = 'L1 (Sum of elements)'
    L2 = 'L2 (Euclidean)'

    wlocals = OrderedDict((
        (COUNT, lambda tf: tf),
        (BINARY, lambda tf: np.greater(tf, 0).astype(int) if tf.size
                            else np.array([], dtype=int)),
        (SUBLINEAR, lambda tf: 1 + np.log(tf)),
    ))

    wglobals = OrderedDict((
        (NONE, lambda df, N: 1),
        (IDF, lambda df, N: np.log(N/df)),
        (SMOOTH, lambda df, N: np.log(1 + N/df)),
    ))

    norms = OrderedDict((
        (NONE, None),
        (L1, partial(normalize, norm='l1')),
        (L2, partial(normalize, norm='l2')),
    ))

    def __init__(self, norm=NONE, wlocal=COUNT, wglobal=NONE):
        self.norm = norm
        self.wlocal = wlocal
        self.wglobal = wglobal

    def _transform(self, corpus, source_dict=None, callback=dummy_callback):
        if not (len(corpus.dictionary) or source_dict) or not len(corpus):
            return corpus
        temp_corpus = list(corpus.ngrams_iterator(' ', include_postags=True))
        dic = corpora.Dictionary(temp_corpus, prune_at=None) if not source_dict else source_dict
        callback(0.3)
        temp_corpus = [dic.doc2bow(doc) for doc in temp_corpus]
        model = models.TfidfModel(dictionary=dic, normalize=False,
                                  wlocal=self.wlocals[self.wlocal],
                                  wglobal=self.wglobals[self.wglobal])
        callback(0.6)

        X = matutils.corpus2csc(model[temp_corpus], dtype=float, num_terms=len(dic)).T
        norm = self.norms[self.norm]
        if norm:
            X = norm(X)
        callback(0.9)

        # set compute values
        shared_cv = SharedTransform(self, corpus.used_preprocessor, source_dict=dic)
        cv = [VectorizationComputeValue(shared_cv, dic[i]) for i in range(len(dic))]

        corpus = self.add_features(corpus, X, dic, cv, var_attrs={'bow-feature': True})
        callback(1)
        return corpus

    def report(self):
        return (('Term Frequency', self.wlocal),
                ('Document Frequency', self.wglobal),
                ('Regularization', self.norm),)
