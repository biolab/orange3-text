from gensim import models

from .topics import GensimWrapper


class LdaSeqWrapper(GensimWrapper):
    name = 'Dynamic Topic Modeling'
    Model = models.LdaSeqModel

    def __init__(self, **kwargs):
        super().__init__(random_state=0, **kwargs, lda_inference_max_iter=200, passes=5)
