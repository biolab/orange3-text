"""
Module for keyword extraction.
"""
from collections import defaultdict
from itertools import chain
from typing import List, Tuple, Callable

import yake
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from Orange.util import dummy_callback

from orangecontrib.text.keywords.rake import Rake
from orangecontrib.text.keywords.embedding import embedding_keywords, \
    EMBEDDING_LANGUAGE_MAPPING
from orangecontrib.text.preprocess import StopwordsFilter

# all available languages for RAKE
RAKE_LANGUAGES = StopwordsFilter.supported_languages()
# all available languages for YAKE!
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
        texts: List[str],
        language: str = "English",
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

    language = YAKE_LANGUAGE_MAPPING[language]
    extractor = yake.KeywordExtractor(lan=language, n=max_len)

    keywords = []
    n_docs = len(texts)
    for i, text in enumerate(texts):
        progress_callback(i / n_docs)
        keywords.append(extractor.extract_keywords(text))
    return keywords


def rake_keywords(
        texts: List[str],
        language: str = "English",
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

    if language.lower() not in [l.lower() for l in RAKE_LANGUAGES]:
        raise ValueError(f"Language must be one of: {RAKE_LANGUAGES}")

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
    TF_IDF, RAKE, YAKE, EMBEDDING = "TF-IDF", "Rake", "YAKE!", "Embedding"
    ITEMS = list(zip(
        (TF_IDF, YAKE, RAKE, EMBEDDING),
        (tfidf_keywords, yake_keywords, rake_keywords, embedding_keywords)
    ))

    TOKEN_METHODS = TF_IDF, EMBEDDING
    DOCUMENT_METHODS = RAKE, YAKE


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
        return [AggregationMethods.mean,
                AggregationMethods.median,
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
    def median(
            keywords: List[List[Tuple[str, float]]]
    ) -> List[Tuple[str, float]]:
        """
        'median' aggregation function.

        Parameters
        ----------
        keywords : list
            List of keywords for each document.

        Returns
        -------
        Aggregated keyword scores.
        """
        scores = list(chain.from_iterable(keywords))
        unique_scores = defaultdict(lambda: [])
        for word, score in scores:
            unique_scores[word].append(score)
        for word, score in unique_scores.items():
            unique_scores[word] = np.median(score)
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
        unique_scores = defaultdict(lambda: [])
        for word, score in scores:
            unique_scores[word].append(score)
        for word, score in unique_scores.items():
            unique_scores[word] = np.min(score)
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
        unique_scores = defaultdict(lambda: [])
        for word, score in scores:
            unique_scores[word].append(score)
        for word, score in unique_scores.items():
            unique_scores[word] = np.max(score)
        return list(unique_scores.items())
