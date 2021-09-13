from numpy import float64
from gensim import models

from .topics import GensimWrapper


class LdaWrapper(GensimWrapper):
    name = 'Latent Dirichlet Allocation'
    Model = models.LdaModel

    def __init__(self, **kwargs):
        super().__init__(random_state=0, **kwargs, dtype=float64)
