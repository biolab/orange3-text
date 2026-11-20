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
import numpy as np

from Orange.util import dummy_callback
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

from orangecontrib.text.preprocess.dictionary import Dictionary
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
        (COUNT, True),
        (BINARY, True),
        (SUBLINEAR, True),
    ))

    wglobals = OrderedDict((
        (NONE, True),
        (IDF, True),
        (SMOOTH, True),
    ))

    norms = OrderedDict((
        (NONE, None),
        (L1, 'l1'),
        (L2, 'l2'),
    ))

    def __init__(self, norm=NONE, wlocal=COUNT, wglobal=NONE):
        self.norm = norm
        self.wlocal = wlocal
        self.wglobal = wglobal

    def _transform(self, corpus, source_dict=None, callback=dummy_callback):
        if len(corpus) == 0:
            return corpus
        temp_corpus = list(corpus.ngrams_iterator(' ', include_postags=True))
        norm = self.norms[self.norm]
        binary = self.wlocal == self.BINARY
        use_idf = self.wglobal != self.NONE
        smooth_idf = self.wglobal == self.SMOOTH
        sublinear_tf = self.wlocal == self.SUBLINEAR
        if not source_dict:
            corpus.store_tokens(temp_corpus)
            dic = Dictionary(temp_corpus, prune_at=None)
            if len(dic) == 0:
                return corpus
            callback(0.3)
            vectorizer = CountVectorizer(
                tokenizer=lambda x: x,
                preprocessor=lambda x: x,
                token_pattern=None,
                lowercase=False,
                binary=binary,
                vocabulary=dic.token2id,
            )
            X_counts = vectorizer.fit_transform(temp_corpus)

            callback(0.6)

            transformer = TfidfTransformer(
                norm=norm,
                use_idf=use_idf,
                smooth_idf=smooth_idf,
                sublinear_tf=sublinear_tf
            )
            X = transformer.fit_transform(X_counts)
        else:
            dic = source_dict
            callback(0.3)
            vectorizer = CountVectorizer(
                tokenizer=lambda x: x,
                preprocessor=lambda x: x,
                token_pattern=None,
                lowercase=False,
                binary=binary,
                vocabulary=dic.token2id,
            )
            X_counts = vectorizer.transform(temp_corpus)

            callback(0.6)

            # Fit the transformer directly on source_dict counts
            transformer = TfidfTransformer(
                norm=norm,
                use_idf=use_idf,
                smooth_idf=smooth_idf,
                sublinear_tf=sublinear_tf
            )

            if use_idf:
                # Manually set IDF values from gensim Dictionary
                num_tokens = len(dic.token2id)
                idf_values = np.ones(num_tokens)

                for token_id, doc_freq in dic.dfs.items():
                    if smooth_idf:
                        idf_values[token_id] = np.log(
                            (dic.num_docs + 1) / (doc_freq + 1)) + 1
                    else:
                        idf_values[token_id] = np.log(dic.num_docs / doc_freq) + 1

                transformer.idf_ = idf_values
            else:
                # Fit on current data when not using pre-computed IDF
                transformer.fit(X_counts)

            X = transformer.transform(X_counts)

        callback(0.9)

        # set compute values
        shared_cv = SharedTransform(self, corpus.used_preprocessor, source_dict=dic)
        cv = [VectorizationComputeValue(shared_cv, dic[i]) for i in range(len(dic))]

        corpus = self.add_features(corpus, X, dic, cv, var_attrs={'bow-feature': True})
        callback(1)
        return corpus
