from collections import OrderedDict
from functools import partial

import numpy as np
from gensim import corpora, models, matutils
from sklearn.preprocessing import normalize

from Orange.data.util import SharedComputeValue
from orangecontrib.text.vectorization.base import BaseVectorizer


class BoWPreprocessTransform:
    """
    Shared computation for transforming new data set into the classifiers's BoW domain.
    This will run preprocessing as well as BoW transformation itself.
    """
    def __init__(self, preprocessor, bow_vectorizer, dictionary):
        self.preprocessor = preprocessor
        self.bow_vectorizer = bow_vectorizer
        self.dictionary = dictionary

    def __call__(self, new_corpus):
        new_corpus = self.preprocessor(new_corpus)
        bow_corpus = self.bow_vectorizer.transform(new_corpus, copy=True, source_dict=self.dictionary)
        # store name to indices mapping so BoWComputeValue can run faster
        bow_corpus.feature_name_to_index = {attr.name: i for i, attr in enumerate(bow_corpus.domain.attributes)}
        return bow_corpus


class BoWComputeValue(SharedComputeValue):
    """
    Compute Value for Bow features. This enables applying a
    classifier — that was trained on a BoW model — on new data.
    """
    def __init__(self, name, compute_shared):
        super().__init__(compute_shared)
        self.name = name

    def compute(self, data, shared_data):
        ind = shared_data.feature_name_to_index[self.name]
        return shared_data.X[:, ind]


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
        (BINARY, lambda tf: int(tf > 0)),
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

    def _transform(self, corpus, source_dict=None):
        if corpus.pos_tags is None:
            temp_corpus = list(corpus.ngrams)
        else:
            temp_corpus = list(corpus.ngrams_iterator(' ', include_postags=True))

        dic = corpora.Dictionary(temp_corpus, prune_at=None) if not source_dict else source_dict
        temp_corpus = [dic.doc2bow(doc) for doc in temp_corpus]
        model = models.TfidfModel(temp_corpus, normalize=False,
                                  wlocal=self.wlocals[self.wlocal],
                                  wglobal=self.wglobals[self.wglobal])

        X = matutils.corpus2csc(model[temp_corpus], dtype=np.float, num_terms=len(dic)).T
        norm = self.norms[self.norm]
        if norm:
            X = norm(X)

        # set compute values
        shared_cv = BoWPreprocessTransform(corpus.used_preprocessor, self, dic)
        cv = [BoWComputeValue(dic[i], shared_cv) for i in range(len(dic))]

        self.add_features(corpus, X, dic, cv)
        return corpus

    def report(self):
        return (('Term Frequency', self.wlocal),
                ('Document Frequency', self.wglobal),
                ('Regularization', self.norm),)
