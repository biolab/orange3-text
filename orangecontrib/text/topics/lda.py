from gensim import models

from .topics import GensimWrapper


class LdaWrapper(GensimWrapper):
    name = 'Lda Model'
    Model = models.LdaModel
