from functools import wraps
from math import ceil

import numpy as np
import scipy.sparse as sp

def chunks(iterable, chunk_size):
    """ Splits iterable objects into chunk of fixed size.
    The last chunk may be truncated.
    """
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def chunkable(method):
    """ This decorators wraps methods that can be executed by passing data by chunks.

    It allows you to pass additional arguments like `chunk_number` and `on_progress` callback
    to monitor the execution's progress.

    Note:
        If no callback is provided data won't be splitted.
    """

    @wraps(method)
    def wrapper(self, data, chunk_number=100, on_progress=None, *args, **kwargs):
        if on_progress:
            chunk_size = ceil(len(data) / chunk_number)
            progress = 0
            res = []
            for i, chunk in enumerate(chunks(data, chunk_size=chunk_size)):
                chunk_res = method(self, chunk, *args, **kwargs)
                if chunk_res:
                    res.extend(chunk_res)

                progress += len(chunk)
                on_progress(progress/len(data))
        else:
            res = method(self, data, *args, **kwargs)

        return res

    return wrapper


def np_sp_sum(x, axis=None):
    """ Wrapper for summing either sparse or dense matrices.
    Required since with scipy==0.17.1 np.sum() crashes."""
    if sp.issparse(x):
        r = x.sum(axis=axis)
        if axis is not None:
            r = np.array(r).ravel()
        return r
    else:
        return np.sum(x, axis=axis)
