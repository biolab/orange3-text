import numpy as np
from scipy import stats, sparse

def is_sorted(l):
    return all(l[i] <= l[i+1] for i in range(len(l)-1))

def hypergeom_p_values(data, selected, callback=None):
    """
    Calculates p_values using Hypergeometric distribution for two numpy arrays.
    Works on a matrices containing zeros and ones. All other values are truncated to zeros and ones.

    Args:
        data (numpy.array): all examples in rows, theirs features in columns.
        selected (numpy.array): selected examples in rows, theirs features in columns.
        callback: callback function used for printing progress.

    Returns: p-values for features

    """
    def col_sum(x):
        if sparse.issparse(x):
            return np.squeeze(np.asarray(x.sum(axis=0)))
        else:
            return np.sum(x, axis=0)

    if data.shape[1] != selected.shape[1]:
        raise ValueError("Number of columns does not match.")

    # clip values to a binary variables
    data = data > 0
    selected = selected > 0

    num_features = selected.shape[1]
    pop_size = data.shape[0]                # population size = number of all data examples
    sam_size = selected.shape[0]            # sample size = number of selected examples
    pop_counts = col_sum(data)              # number of observations in population = occurrences of words all data
    sam_counts = col_sum(selected)          # number of observations in sample = occurrences of words in selected data
    step = 250
    p_vals = []

    for i, (pc, sc) in enumerate(zip(pop_counts, sam_counts)):
        hyper = stats.hypergeom(pop_size, pc, sam_size)
        # since p-value is probability of equal to or "more extreme" than what was actually observed
        # we calculate it as 1 - cdf(sc-1). sf is survival function defined as 1-cdf.
        p_vals.append(hyper.sf(sc-1))
        if callback and i % step == 0:
            callback(100*i/num_features)
    return p_vals
