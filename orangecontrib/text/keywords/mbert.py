import zlib
from base64 import b64encode
from itertools import chain
from operator import itemgetter
from typing import Optional, Callable, Tuple, List, Iterable

import numpy as np
from nltk import ngrams
from Orange.misc.server_embedder import ServerEmbedderCommunicator
from Orange.util import dummy_callback

from orangecontrib.text import Corpus


KeywordType = Tuple[str, float]


def _split_key_phrases(kw: KeywordType, max_len: int) -> List[KeywordType]:
    """
    Split key-phrases that are longer than max_len to all possible n-grams
    inside keyword and attach the key-phrase's score. Keep phrases which length
    is shorter or equal to max_len.
    """
    splitted = kw[0].split()
    if len(splitted) <= max_len:
        return [kw]
    else:
        return [(" ".join(x), kw[1]) for x in ngrams(splitted, max_len)]


def _deduplicate(kws: Iterable[KeywordType]) -> List[KeywordType]:
    """
    After splitting key-phrases duplicates can appear among key-phrases. This
    function remove only appearance with the highest score.
    """
    res_kws, res_set = [], set()
    for kw, sc in sorted(kws, key=itemgetter(1), reverse=True):
        if kw not in res_set:
            res_set.add(kw)
            res_kws.append((kw, sc))
    return res_kws


def mbert_keywords(
    documents: List[str],
    max_len: int = 1,
    progress_callback: Callable = dummy_callback,
) -> List[Optional[List[KeywordType]]]:
    """
    Extract keywords using server MBERT keyword extractor.

    Parameters
    ----------
    documents
        Lists of tokens
    max_len
        Maximum number of words in keyword
    progress_callback
        Function for reporting progress.

    Returns
    -------
    Keywords with scores
    """
    emb = _BertServerCommunicator(
        model_name="mbert-keywords",
        max_parallel_requests=30,
        server_url="https://api.garaza.io",
        embedder_type="text",
    )
    keywords = emb.embedd_data(documents, callback=progress_callback)
    processed_kws = []
    for kws in keywords:
        if kws is not None:
            kws = chain.from_iterable(_split_key_phrases(k, max_len) for k in kws)
            kws = _deduplicate(kws)
        processed_kws.append(kws)
    return processed_kws


class _BertServerCommunicator(ServerEmbedderCommunicator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.content_type = "application/octet-stream"

    async def _encode_data_instance(self, instance: str) -> Optional[bytes]:
        return b64encode(zlib.compress(instance.encode("utf-8", "replace"), level=9))


if __name__ == "__main__":
    from Orange.data import Domain, StringVariable

    text = """
    I propose that an employer who orders workers to work overtime, and who also has people working part-time because they cannot be put in other jobs, must pay those whom he orders to work overtime at least double the crisis allowance.
    This would avoid the employer milking the state at the expense of the corona of a law that is designed to address a shortage of work, and would really only order overtime when it could not be done otherwise. Because the reality is that the picture is very different now, and employers are exploiting some workers with overtime, while keeping others on part-time contracts. I believe that the corona law to help businesses is not intended to do this, but to keep businesses alive with as few redundancies as possible.
    The loophole in the law which reads 'During the subsidy period and for one month thereafter, an employer may not order overtime, unevenly distribute or temporarily reallocate working time, provided that he can carry out this work with workers who are ordered to work part-time.'. has come in very handy for some employers to make additional use of, as it is an individual judgement.
    However, a double allowance would prevent such exploitation of both the State and
    """
    c = Corpus.from_numpy(
        Domain([], metas=[StringVariable("Text")]),
        np.empty((1, 0)),
        metas=np.array([[text]])
    )
    print(mbert_keywords(c.documents))
