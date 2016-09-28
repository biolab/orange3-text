import numpy as np
from scipy import stats, sparse
import math

# To speed-up FDR, calculate ahead sum([1/i for i in range(1, m+1)]), for m in [1,100000].
# For higher values of m use an approximation, with error less or equal to 4.99999157277e-006.
# (sum([1/i for i in range(1, m+1)])  ~ log(m) + 0.5772..., 0.5572 is an Euler-Mascheroni constant)
_c = [1.0]
for m in range(2, 100000):
    _c.append(_c[-1] + 1.0/m)


def is_sorted(l):
    return all(l[i] <= l[i+1] for i in range(len(l)-1))


def false_discovery_rate(p_values, dependent=False, m=None, ordered=False):
    """
    `False Discovery Rate <http://en.wikipedia.org/wiki/False_discovery_rate>`_ correction on a list of p-values.

    Args:
        p_values: a list of p-values.
        dependent: use correction for dependent hypotheses.
        m: number of hypotheses tested (default ``len(p_values)``).
        ordered: prevent sorting of p-values if they are already sorted.

    Returns: A list of corrected p-values.

    """
    if not ordered:
        ordered = is_sorted(p_values)

    if not ordered:
        joined = [ (v,i) for i,v in enumerate(p_values) ]
        joined.sort()
        p_values = [ p[0] for p in joined ]
        indices = [ p[1] for p in joined ]

    if not m:
        m = len(p_values)
    if m <= 0 or not p_values:
        return []

    if dependent: # correct q for dependent tests
        k = _c[m-1] if m <= len(_c) else math.log(m) + 0.57721566490153286060651209008240243104215933593992
        m = m * k

    tmp_fdrs = [p*m/(i+1.0) for (i, p) in enumerate(p_values)]
    fdrs = []
    cmin = tmp_fdrs[-1]
    for f in reversed(tmp_fdrs):
        cmin = min(f, cmin)
        fdrs.append( cmin)
    fdrs.reverse()

    if not ordered:
        new = [ None ] * len(fdrs)
        for v,i in zip(fdrs, indices):
            new[i] = v
        fdrs = new

    return fdrs


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
