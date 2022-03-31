from gensim import models

from .topics import GensimWrapper


class NmfWrapper(GensimWrapper):
    name = 'Negative Matrix Factorization'
    Model = models.Nmf

    def __init__(self, **kwargs):
        super().__init__(random_state=0, **kwargs, w_max_iter=200, passes=5)
