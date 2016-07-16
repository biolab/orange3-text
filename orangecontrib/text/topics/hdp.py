from gensim import models

from .topics import GensimWrapper


class HdpModel(models.HdpModel):
    def __init__(self, corpus, id2word, **kwargs):
        # disable fitting during initialization
        _update = self.update
        self.update = lambda x: x
        super().__init__(corpus, id2word, **kwargs)
        self.update = _update


class HdpWrapper(GensimWrapper):
    name = 'Hdp Model'
    Model = HdpModel

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = None

    def reset_model(self, corpus):
        self.model = self.Model(corpus=corpus,
                                id2word=corpus.ngrams_dictionary, **self.kwargs)

    @property
    def num_topics(self):
        return self.model.m_lambda.shape[0] if self.model else 0
