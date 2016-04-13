import warnings

from gensim import corpora, matutils
from gensim.models.tfidfmodel import TfidfModel
from nltk.tokenize import word_tokenize


class BagOfWords():

    def __init__(self, error_callback=None):
        self.error_callback = error_callback

    def __call__(self, corpus):
        if corpus is None:
            raise ValueError(
                    'Cannot compute Bag of Words without an input corpus.'
            )

        try:
            has_tokens = not corpus.tokens is None
        except:
            has_tokens = False

        if not has_tokens:  # Perform default pre-processing.
            # TODO: Use this code, when Preprocessor update is merged.
            # preprocessor = Preprocessor(lowercase=False)
            # corpus = preprocessor(corpus)

            # Temporary pre-processing.
            corpus.tokens = [
                word_tokenize(document)
                for document
                in corpus.documents
            ]

        try:
            dictionary = corpora.Dictionary(
                    corpus.tokens,
                    prune_at=float('Inf')
            )
            frequencies = [dictionary.doc2bow(i) for i in corpus.tokens]
            X = matutils.corpus2dense(
                    frequencies,
                    num_terms=len(dictionary.keys()),
                    dtype=int
            ).T

            tfidf = TfidfModel(corpus)

            corpus.update_attributes(X, dictionary)
        except Exception as e:
            warnings.warn(
                    'An exception occurred ({0}).'.format(e),
                    RuntimeWarning
            )
            if self.error_callback:
                self.error_callback(e)
            return

        return corpus
