import warnings

import numpy as np

from gensim import corpora, matutils
from gensim.models.tfidfmodel import TfidfModel


class BagOfWords():

    def __init__(self, progress_callback=None, error_callback=None):
        self.progress_callback = progress_callback
        self.error_callback = error_callback
        self.dictionary = None

    def __call__(self, corpus, use_tfidf=False):
        if corpus is None:
            raise ValueError(
                    'Cannot compute Bag of Words without an input corpus.'
            )

        self.check_progress()  # Step 1

        self.dictionary = corpus.dictionary

        # Term frequencies.
        bag_of_words = [self.dictionary.doc2bow(i) for i in corpus.tokens]

        self.check_progress()  # Step 2

        if use_tfidf:
            tfidf_model = TfidfModel(bag_of_words)
            bag_of_words = tfidf_model[bag_of_words]

        self.check_progress()  # Step 3

        X = matutils.corpus2dense(
                bag_of_words,
                num_terms=len(self.dictionary.keys()),
                dtype=np.float64
        ).T

        bow_corpus = corpus.copy()
        feats = [v for k, v in sorted(self.dictionary.items())]
        bow_corpus.extend_attributes(X, feats, var_attrs={'hidden': True})

        self.check_progress()  # Step 4

        return bow_corpus

    def check_progress(self):
        if self.progress_callback:
            self.progress_callback()
