from gensim import models

from .topics import GensimWrapper

models.LsiModel.update = models.LsiModel.add_documents
models.LsiModel.add_documents = lambda self, *args, **kwargs: self.update(*args, **kwargs)


class LsiWrapper(GensimWrapper):
    name = 'Latent Semantic Indexing'
    Model = models.LsiModel
