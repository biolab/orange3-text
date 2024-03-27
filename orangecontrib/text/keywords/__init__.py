"""
Module for keyword extraction.
"""
from collections import defaultdict
from itertools import chain
from typing import List, Tuple, Callable

import yake
from Orange.data import Domain
from nltk.corpus import stopwords
import numpy as np

from Orange.util import dummy_callback

from orangecontrib.text import Corpus
from orangecontrib.text.keywords.mbert import mbert_keywords
from orangecontrib.text.keywords.rake import Rake
from orangecontrib.text.language import ISO2LANG
from orangecontrib.text.preprocess import StopwordsFilter
from orangecontrib.text.vectorization import BowVectorizer

# all available languages for RAKE
RAKE_LANGUAGES = StopwordsFilter.supported_languages()
# all available languages for YAKE!
# fmt: off
YAKE_LANGUAGES = [
    "ar", "hy", "br", "bg", "zh", "hr", "cs", "da", "nl", "en", "et", "fi",
    "fr", "de", "el", "hi", "hu", "id", "it", "ja", "lv", "lt", "no", "fa",
    "pl", "pt", "ro", "ru", "sk", "sl", "es", "sv", "tr", "uk"
]
# fmt: on


def tfidf_keywords(
    corpus: Corpus, progress_callback: Callable = None
) -> List[List[Tuple[str, float]]]:
    """
    Extract keywords using TF-IDF.

    Parameters
    ----------
    tokens : list
        Lists of tokens.
    progress_callback : callable
        Function for reporting progress.

    Returns
    -------
    keywords : list
    """
    if progress_callback is None:
        progress_callback = dummy_callback

    # empty X part - to know that every feature of X is bag of wrds
    domain = Domain([], class_vars=corpus.domain.class_vars, metas=corpus.domain.metas)
    corpus = corpus.from_table(domain, corpus)

    vectorizer = BowVectorizer(
        wlocal=BowVectorizer.COUNT,
        wglobal=BowVectorizer.IDF if len(corpus) > 1 else BowVectorizer.NONE,
        norm=BowVectorizer.L2,
    )
    res = vectorizer.transform(corpus)
    X, words = res.X, [a.name for a in res.domain.attributes]

    keywords = []
    n_docs = X.shape[0]
    for i, row in enumerate(X):
        progress_callback(i / n_docs)
        nonzero = row.nonzero()
        if len(nonzero) > 1:
            keywords.append([(words[i], row[0, i]) for i in nonzero[1]])
        else:
            keywords.append([])
    return keywords


def yake_keywords(
        texts: List[str],
        language: str = "en",
        max_len: int = 1,
        progress_callback: Callable = None
) -> List[List[Tuple[str, float]]]:
    """
    Extract keywords using YAKE!.

    Parameters
    ----------
    texts : list
        List of documents.
    language : str
        Selected language.
    max_len : int
        Maximum number of tokens.
    progress_callback : callable
        Function for reporting progress.

    Returns
    -------
    keywords : list
    """
    if progress_callback is None:
        progress_callback = dummy_callback

    extractor = yake.KeywordExtractor(lan=language, n=max_len)

    keywords = []
    n_docs = len(texts)
    for i, text in enumerate(texts):
        progress_callback(i / n_docs)
        keywords.append(extractor.extract_keywords(text))
    return keywords


def rake_keywords(
        texts: List[str],
        language: str = "en",
        max_len: int = 1,
        progress_callback: Callable = None
) -> List[List[Tuple[str, float]]]:
    """
    Extract keywords from text with RAKE method.

    Parameters
    ----------
    texts : list
        List of texts from which keywords are extracted
    language : str
        The language of texts
    max_len : int
        Maximal length of keywords/keyphrases extracted
    progress_callback : callable
        Function for reporting progress.

    Returns
    -------
    List which contains list of keywords for each of the documents in texts.
    For each keyword function returns a tuple with keyword and it's score.
    """
    if progress_callback is None:
        progress_callback = dummy_callback

    if language not in RAKE_LANGUAGES:
        raise ValueError(f"Language must be one of: {RAKE_LANGUAGES}")

    language = ISO2LANG[language]
    # some languages (e.g. Slovenian have different name than ISO name in nltk)
    language = StopwordsFilter.LANG2NLTK.get(language, language)
    stop_words_ = [x.strip() for x in stopwords.words(language.lower())]
    rake_object = Rake(stop_words_, max_words_length=max_len)

    keywords = []
    n_docs = len(texts)
    for i, text in enumerate(texts):
        progress_callback(i / n_docs)
        keywords.append(rake_object.run(text))
    return keywords


class ScoringMethods:
    """
    Scoring methods enum.
    """
    TF_IDF, RAKE, YAKE, MBERT = "TF-IDF", "Rake", "YAKE!", "mBERT"
    ITEMS = list(zip(
        (TF_IDF, YAKE, RAKE, MBERT),
        (tfidf_keywords, yake_keywords, rake_keywords, mbert_keywords)
    ))

    TOKEN_METHODS = TF_IDF,
    DOCUMENT_METHODS = RAKE, YAKE, MBERT


class AggregationMethods:
    """
    Aggregation methods enum and helper functions.
    """
    MEAN, MEDIAN, MIN, MAX = range(4)
    ITEMS = "Mean", "Median", "Min", "Max"

    @staticmethod
    def aggregate(
            keywords: List[List[Tuple[str, float]]],
            agg_method: int
    ) -> List[Tuple[str, float]]:
        """
        Aggregate scores.

        Parameters
        ----------
        keywords : list
            List of keywords for each document.
        agg_method : int
            Method type. One of: MEAN, MEDIAN, MIN, MAX.

        Returns
        -------
        Aggregated keyword scores.
        """
        compute_mean = agg_method == AggregationMethods.MEAN
        aggregator = [np.mean, np.median, np.min, np.max][agg_method]
        unique_scores = defaultdict(lambda: 0. if compute_mean else [])

        for word, score in chain.from_iterable(keywords):
            unique_scores[word] += score if compute_mean else [score]

        for word, score in unique_scores.items():
            # compute mean amongst all keywords
            unique_scores[word] = score / len(keywords) if compute_mean \
                else aggregator(score)

        return list(unique_scores.items())
