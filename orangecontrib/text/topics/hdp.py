from gensim import models

from .topics import GensimWrapper


class HdpWrapper(GensimWrapper):
    name = 'Hierarchical Dirichlet Process'
    Model = models.HdpModel

    @property
    def num_topics(self):
        return self.model.m_lambda.shape[0] if self.model else 0
