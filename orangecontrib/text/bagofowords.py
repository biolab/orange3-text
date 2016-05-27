import warnings

import numpy as np

from gensim import corpora, matutils
from gensim.models.tfidfmodel import TfidfModel

from orangecontrib.text.preprocess import Preprocessor


class BagOfWords():

    def __init__(self, progress_callback=None, error_callback=None):
        self.progress_callback = progress_callback
        self.error_callback = error_callback
        self.vocabulary = None

    def __call__(self, corpus, use_tfidf=False):
        if corpus is None:
            raise ValueError(
                    'Cannot compute Bag of Words without an input corpus.'
            )

        has_tokens = hasattr(corpus, 'tokens') and corpus.tokens is not None
        if not has_tokens:  # Perform default pre-processing.
            preprocessor = Preprocessor()
            corpus = preprocessor(corpus)

        self.check_progress()  # Step 1

        dictionary = corpora.Dictionary(corpus.tokens, prune_at=np.inf)
        self.vocabulary = dictionary

        # Term frequencies.
        bag_of_words = [dictionary.doc2bow(i) for i in corpus.tokens]

        self.check_progress()  # Step 2

        if use_tfidf:
            tfidf_model = TfidfModel(bag_of_words)
            bag_of_words = tfidf_model[bag_of_words]

        self.check_progress()  # Step 3

        X = matutils.corpus2dense(
                bag_of_words,
                num_terms=len(dictionary.keys()),
                dtype=np.float64
        ).T

        bow_corpus = corpus.copy()
        feats = [v for k, v in sorted(dictionary.items())]
        bow_corpus.extend_attributes(X, feats, var_attrs={'bow_feature': True})
        self.check_progress()  # Step 4

        return bow_corpus

    def check_progress(self):
        if self.progress_callback:
            self.progress_callback()
