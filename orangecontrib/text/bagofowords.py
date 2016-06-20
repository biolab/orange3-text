import nltk
import numpy as np
from simhash import Simhash

from sklearn.feature_extraction import text

from orangecontrib.text.corpus import Corpus


__all__ = ['CountVectorizer', 'TfidfVectorizer', 'SimhashVectorizer']


class SklearnVectorizerMixin:

    def fit(self, data, y=None):
        if isinstance(data, Corpus):
            return super().fit(data.tokens, y)
        return super().fit(data, y)

    def fit_transform(self, corpus, y=None):
        if isinstance(corpus, Corpus):
            matrix = super().fit_transform(corpus.tokens, y)
            return self.__extend_corpus(corpus, matrix)
        return super().fit_transform(corpus, y)

    def transform(self, corpus):
        if isinstance(corpus, Corpus):
            matrix = super().transform(corpus.tokens)
            return self.__extend_corpus(corpus, matrix)
        return super().transform(corpus)

    def __extend_corpus(self, corpus, matrix):
        new_corpus = corpus.copy()
        new_corpus.extend_attributes(matrix, sorted(self.vocabulary_.keys()),
                                     var_attrs={'hidden': True})
        return new_corpus


class CountVectorizer(SklearnVectorizerMixin, text.CountVectorizer):
    name = 'Count Vectorizer'

    def __init__(self, binary=False, ngram_range=(1, 1)):
        super().__init__(input='content', encoding='utf-8',
                         decode_error='strict', strip_accents=None,
                         lowercase=False, preprocessor=lambda x: x, tokenizer=lambda x: x,
                         stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                         ngram_range=ngram_range, analyzer='word',
                         max_df=1.0, min_df=1, max_features=None,
                         vocabulary=None, binary=binary, dtype=np.int64)

    def report(self):
        return (('N-gram range', self.ngram_range),
                ('Binary', self.binary))


class TfidfVectorizer(SklearnVectorizerMixin, text.TfidfVectorizer):
    name = 'Tfidf Vectorizer'

    def __init__(self, binary=False, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False, ngram_range=(1, 1)):
        super().__init__(input='content', encoding='utf-8',
                         decode_error='strict', strip_accents=None, lowercase=False,
                         preprocessor=lambda x: x, tokenizer=lambda x: x, analyzer='word',
                         stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                         ngram_range=ngram_range, max_df=1.0, min_df=1,
                         max_features=None, vocabulary=None, binary=binary,
                         dtype=np.int64, norm=norm, use_idf=use_idf, smooth_idf=smooth_idf,
                         sublinear_tf=sublinear_tf)

    def report(self):
        return (('N-gram range', self.ngram_range),
                ('Binary', self.binary),
                ('Use idf', self.use_idf),
                ('Smooth idf', self.smooth_idf),
                ('Sublinear tf', self.sublinear_tf))


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
