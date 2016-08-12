from gensim import models

from .topics import GensimWrapper


class LdaWrapper(GensimWrapper):
    name = 'Latent Dirichlet Allocation'
    Model = models.LdaModel
