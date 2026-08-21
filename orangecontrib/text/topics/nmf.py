from sklearn.decomposition import NMF

from .topics import SklearnWrapper


class NmfWrapper(SklearnWrapper):
    name = 'Negative Matrix Factorization'
    Model = NMF

    def __init__(self, **kwargs):
        super().__init__(
            random_state=0,
            max_iter=400,  # NMF sometimes needs more iterations
            **kwargs
        )
