import numpy as np
import scipy as sp
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

    :param p_values: a list of p-values.
    :param dependent: use correction for dependent hypotheses (default False).
    :param m: number of hypotheses tested (default ``len(p_values)``).
    :param ordered: prevent sorting of p-values if they are already sorted (default False).
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


def hypergeom_p_values(selected, other, callback):
    """
    Calculates p_values using Hypergeometric distribution for two numpy arrays.

    :param selected: selected examples in rows, theirs features in columns
    :type selected: numpy.array
    :param other: other examples in rows, theirs features in columns
    :type other: numpy.array
    :return: p-values
    """
    if selected.shape[1] != other.shape[1]:
        raise ValueError("Number of columns does not match.")

    sel_ex = selected.shape[0]
    oth_ex = other.shape[0]
    all_ex = sel_ex + oth_ex
    all_features = selected.shape[1]

    sel = np.sum(selected, axis=0)
    oth = np.sum(other, axis=0)

    i = 0
    step = max(int(all_features/100), 1)
    p_vals = []

    for ns, no in zip(sel, oth):
        hyper = sp.stats.hypergeom(all_ex, sel_ex, ns+no)
        p_vals.append(hyper.sf(ns-1))    # P(we select >= ns docs) = 1 - cdf(ns-1)
        i += 1
        if i % step == 0:
            callback(int(i/step))
    return p_vals
