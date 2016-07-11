from gensim import models

from .topics import GensimWrapper


class LdaWrapper(GensimWrapper):
    name = 'Lda Model'
    Model = models.LdaModel

    def __init__(self, num_topics):
        self.num_topics = num_topics

    def reset_model(self, corpus):
        self.model = self.Model(id2word=corpus.ngrams_dictionary, num_topics=self.num_topics)
