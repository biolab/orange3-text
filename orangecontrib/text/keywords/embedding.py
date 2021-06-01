from collections import defaultdict
from operator import itemgetter
from typing import List, Tuple, Dict, Set, Callable, Collection

import numpy as np
from Orange.util import dummy_callback, wrap_callback

from orangecontrib.text import Corpus
from orangecontrib.text.vectorization import document_embedder
from orangecontrib.text.vectorization.document_embedder import DocumentEmbedder
from sklearn.metrics.pairwise import cosine_similarity


EMBEDDING_LANGUAGE_MAPPING = document_embedder.LANGS_TO_ISO


def _embedd_tokens(
    tokens: Collection[List[str]], language: str, progress_callback: Callable
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Set[int]]]:
    """
    Embedd document and words and create a mapping dictionary between words and
    documents
    """
    # extract words
    word2doc = defaultdict(set)
    for i, doc_tokens in enumerate(tokens):
        for t in doc_tokens:
            word2doc[t].add(i)
    words = list(word2doc.keys())

    # TODO: currently embedding report success unify them to report progress float
    ticks = iter(np.linspace(0, 1, len(tokens) + len(words)))

    def emb_cb(sucess: bool):
        if sucess:
            progress_callback(next(ticks))

    # embedd documents
    embedder = DocumentEmbedder(language=language)
    # tokens is tranformedt to list in case it is np.ndarray
    doc_embs = np.array(embedder(list(tokens), emb_cb))

    # embedd words
    word_embs = np.array(embedder([[w] for w in words], emb_cb))

    return doc_embs, word_embs, word2doc


def cos_dist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 1 - cosine_similarity(x, y)


def embedding_keywords(
    tokens: Collection[List[str]],
    language: str = "English",
    progress_callback: Callable = None,
) -> List[List[Tuple[str, float]]]:
    """
    Extract keywords using Embeddings.

    Parameters
    ----------
    tokens
        Lists of tokens
    language
        Language of documents
    progress_callback
        Function for reporting progress.

    Returns
    -------
    Keywords with scores
    """
    if len(tokens) == 0:
        return []
    if progress_callback is None:
        progress_callback = dummy_callback

    # prepare structures
    language = EMBEDDING_LANGUAGE_MAPPING[language]
    doc_embs, word_embs, word2doc = _embedd_tokens(
        tokens, language, wrap_callback(progress_callback, 0, 0.7)
    )
    doc2word = [set(t) for t in tokens]
    word2ind = {w: i for i, w in enumerate(word2doc)}

    # many combinations of distances will not be used since each document do
    # not include all words. Anyway it is still much faster to compute all
    # distances pairs because of matrix calculations
    distances = cos_dist(doc_embs, word_embs)
    # the sum of document embeddings for each word
    dist_sums = {
        w: distances[list(dcs), i].sum() for i, (w, dcs) in enumerate(word2doc.items())
    }

    cb = wrap_callback(progress_callback, 0.7, 1)
    # compute keywords scores
    doc_desc = []
    for j in range(doc_embs.shape[0]):
        scores = []
        for word in doc2word[j]:
            l_ = len(word2doc[word])
            dist = distances[j, word2ind[word]]
            mean_distance = ((dist_sums[word] - dist) / (l_ - 1)) if l_ > 1 else 0
            scores.append((word, dist - mean_distance))
        doc_desc.append(sorted(scores, key=itemgetter(1)))
        cb((j + 1) / len(doc_embs))
    return doc_desc


if __name__ == "__main__":
    tokens = Corpus.from_file("book-excerpts").tokens[:3]
    res = embedding_keywords(tokens, "English")[0][-3:]
    print(res)
