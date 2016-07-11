from gensim import models

from .topics import GensimWrapper

models.LsiModel.update = models.LsiModel.add_documents


class LsiWrapper(GensimWrapper):
    name = 'Lsi Model'
    Model = models.LsiModel

    def __init__(self, num_topics):
        self.num_topics = num_topics

    def reset_model(self, corpus):
        self.model = self.Model(id2word=corpus.ngrams_dictionary, num_topics=self.num_topics)
