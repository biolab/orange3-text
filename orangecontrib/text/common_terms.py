import numpy as np
import scipy.sparse as sp

from Orange.misc import DistMatrix


def common_terms(data, rows=None, cols=None):
    """Accepts np.array and returns a distance matrix where values are
    counts of non-zero column values."""
    matrix = np.zeros((data.shape[0], data.shape[0]))

    for i in range(len(matrix) - 1):
        for j in range(i+1,len(matrix)):
            if sp.issparse(data):
                w = len(set(data[i].indices).intersection(data[j].indices))
            else:
                w = sum((data[i] > 0) & (data[j] > 0))
            matrix[i, j] = w
            matrix[j, i] = w

    return DistMatrix(matrix, row_items=rows, col_items=cols)

