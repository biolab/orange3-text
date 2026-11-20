from numpy import float64
from sklearn.decomposition import LatentDirichletAllocation

from .topics import SklearnWrapper


class LdaWrapper(SklearnWrapper):
    name = 'Latent Dirichlet Allocation'
    Model = LatentDirichletAllocation

    def __init__(self, **kwargs):
        # default max_iter=10 is often too low, 200 is more robust
        super().__init__(
            random_state=0,
            max_iter=200,
            learning_method='online',  # similar to gensim's passes
            **kwargs
        )