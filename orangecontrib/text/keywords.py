"""
Module for keyword extraction.
"""
from collections import defaultdict
from itertools import chain
from typing import List, Tuple, Callable

import yake
from sklearn.feature_extraction.text import TfidfVectorizer

from Orange.util import dummy_callback

YAKE_LANGUAGE_MAPPING = {
    "Arabic": "ar",
    "Armenian": "hy",
    "Breton": "br",
    "Bulgarian": "bg",
    "Chinese": "zh",
    "Croatian": "hr",
    "Czech": "cz",
    "Danish": "da",
    "Dutch": "nl",
    "English": "en",
    "Estonian": "et",
    "Finnish": "fi",
    "French": "fr",
    "German": "de",
    "Greek": "el",
    "Hindi": "hi",
    "Hungarian": "hu",
    "Indonesian": "id",
    "Italian": "it",
    "Japanese": "ja",
    "Latvian": "lv",
    "Lithuanian": "lt",
    "Norwegian": "no",
    "Persian": "fa",
    "Polish": "pl",
    "Portuguese": "pt",
    "Romanian": "ro",
    "Russian": "ru",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Spanish": "es",
    "Swedish": "sv",
    "Turkish": "tr",
    "Ukrainian": "uk"
}


def tfidf_keywords(
        tokens: List[List[str]],
        progress_callback: Callable = None
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

    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
    X = vectorizer.fit_transform(tokens)
    words = vectorizer.get_feature_names()

    keywords = []
    n_docs = X.shape[0]
    for i, row in enumerate(X):
        progress_callback(i / n_docs)
        keywords.append([(words[i], row[0, i]) for i in row.nonzero()[1]])
    return keywords


def yake_keywords(
        documents: List[str],
        language: str = "English",
        max_len: int = 1,
        progress_callback: Callable = None
) -> List[List[Tuple[str, float]]]:
    """
    Extract keywords using YAKE!.

    Parameters
    ----------
    documents : list
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

    language = YAKE_LANGUAGE_MAPPING[language]
    extractor = yake.KeywordExtractor(lan=language, n=max_len)

    keywords = []
    n_docs = len(documents)
    for i, text in enumerate(documents):
        progress_callback(i / n_docs)
        keywords.append(extractor.extract_keywords(text))
    return keywords


class ScoringMethods:
    """
    Scoring methods enum.
    """
    TF_IDF, RAKE, YAKE, EMBEDDING = "TF-IDF", "Rake", "YAKE!", "Embedding"
    ITEMS = list(zip((TF_IDF, YAKE),
                     (tfidf_keywords, yake_keywords)))

    TOKEN_METHODS = TF_IDF, EMBEDDING
    DOCUMENT_METHODS = RAKE, YAKE


class AggregationMethods:
    """
    Aggregation methods enum and helper functions.
    """
    MEAN, MIN, MAX = range(3)
    ITEMS = "Mean", "Minimum", "Maximum"

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
            Method type. One of: MEAN, MIN, MAX.

        Returns
        -------
        Aggregated keyword scores.
        """
        return [AggregationMethods.mean,
                AggregationMethods.min,
                AggregationMethods.max][agg_method](keywords)

    @staticmethod
    def mean(
            keywords: List[List[Tuple[str, float]]]
    ) -> List[Tuple[str, float]]:
        """
        'mean' aggregation function.

        Parameters
        ----------
        keywords : list
            List of keywords for each document.

        Returns
        -------
        Aggregated keyword scores.
        """
        scores = list(chain.from_iterable(keywords))
        unique_scores = defaultdict(lambda: 0.)
        for word, score in scores:
            unique_scores[word] += score
        for word, score in unique_scores.items():
            unique_scores[word] = score / len(keywords)
        return list(unique_scores.items())

    @staticmethod
    def min(
            keywords: List[List[Tuple[str, float]]]
    ) -> List[Tuple[str, float]]:
        """
        'min' aggregation function.

        Parameters
        ----------
        keywords : list
            List of keywords for each document.

        Returns
        -------
        Aggregated keyword scores.
        """
        scores = list(chain.from_iterable(keywords))
        unique_scores = defaultdict(lambda: 1.)
        for word, score in scores:
            assert score <= 1
            if unique_scores[word] > score:
                unique_scores[word] = score
        return list(unique_scores.items())

    @staticmethod
    def max(
            keywords: List[List[Tuple[str, float]]]
    ) -> List[Tuple[str, float]]:
        """
        'max' aggregation function.

        Parameters
        ----------
        keywords : list
            List of keywords for each document.

        Returns
        -------
        Aggregated keyword scores.
        """
        scores = list(chain.from_iterable(keywords))
        unique_scores = defaultdict(lambda: 0.)
        for word, score in scores:
            assert score >= 0
            if unique_scores[word] < score:
                unique_scores[word] = score
        return list(unique_scores.items())
