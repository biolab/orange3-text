from functools import wraps
from math import ceil
from typing import Union, List, Callable, Any, Tuple

import numpy as np
import scipy.sparse as sp
from Orange.data import Domain, DiscreteVariable
from gensim.matutils import Sparse2Corpus

from orangecontrib.text import Corpus
from orangecontrib.text.language import infer_language_from_variable


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


class Sparse2CorpusSliceable(Sparse2Corpus):
    """
    Sparse2Corpus support only retrieving a vector for single document.
    This class implements slice operation on the Sparse2Corpus object.

    Todo: this implementation is temporary, remove it when/if implemented in gensim
    """

    def __getitem__(
        self, key: Union[int, List[int], np.ndarray, type(...), slice]
    ) -> Sparse2Corpus:
        """Retrieve a document vector from the corpus by its index.

        Parameters
        ----------
        key
            Index of document or slice for documents

        Returns
        -------
        Selected subset of sparse data from self.
        """
        sparse = self.sparse.__getitem__((slice(None, None, None), key))
        return Sparse2CorpusSliceable(sparse)


def create_corpus(
    documents: List[Any],
    attributes: List[Tuple[Callable, Callable]],
    class_vars: List[Tuple[Callable, Callable]],
    metas: List[Tuple[Callable, Callable]],
    title_indices: List[int],
    text_features: List[str],
    name: str,
    language_attribute: str,
):
    """
    Create a corpus from list of features/documents produced by modelu such as
    Guardian/NYT

    Parameters
    ----------
    documents
        List with values downloaded from API
    attributes
        List of attributes and recipes on how to extract values from documents.
    class_vars
        List of class attributes and recipes on how to extract values from documents.
    metas
        List of meta and recipes on how to extract values from documents.
    title_indices
        The index of the title attribute.
    text_features
        Names of text features
    name
        The name of the Corpus
    language_attribute
        The attribute to infer the language from.

    Returns
    -------
    Corpus with documents.
    """
    domain = Domain(
        attributes=[attr() for attr, _ in attributes],
        class_vars=[attr() for attr, _ in class_vars],
        metas=[attr() for attr, _ in metas],
    )
    for ind in title_indices:
        domain[ind].attributes["title"] = True

    def to_val(attr, val):
        if isinstance(attr, DiscreteVariable):
            attr.val_from_str_add(val)
        return attr.to_val(val)

    X = [
        [to_val(a, f(doc)) for a, (_, f) in zip(domain.class_vars, attributes)]
        for doc in documents
    ]
    Y = [
        [to_val(a, f(doc)) for a, (_, f) in zip(domain.class_vars, class_vars)]
        for doc in documents
    ]
    metas = [
        [to_val(a, f(doc)) for a, (_, f) in zip(domain.metas, metas)]
        for doc in documents
    ]
    X = np.array(X, dtype=np.float64)
    Y = np.array(Y, dtype=np.float64)
    metas = np.array(metas, dtype=object)

    language = infer_language_from_variable(domain[language_attribute])
    corpus = Corpus.from_numpy(
        domain=domain,
        X=X,
        Y=Y,
        metas=metas,
        text_features=[domain[f] for f in text_features],
        language=language,
    )
    corpus.name = name
    return corpus
