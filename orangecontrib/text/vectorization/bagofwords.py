from gensim import corpora, matutils

from orangecontrib.text.vectorization.base import BaseVectorizer


class CountVectorizer(BaseVectorizer):
    name = 'Count Vectorizer'

    def __init__(self, binary=False):
        self.binary = binary

    def _transform(self, corpus):
        if corpus.pos_tags is None:
            ngrams = list(corpus.ngrams)
        else:
            ngrams = list(corpus.ngrams_iterator(' ', include_postags=True))

        dic = corpora.Dictionary(ngrams, prune_at=None)
        X = matutils.corpus2csc(map(dic.doc2bow, ngrams)).T

        if self.binary:
            X[X > 1] = 1

        self.add_features(corpus, X, dic)
        return corpus

    def report(self):
        return ('Binary', self.binary),
