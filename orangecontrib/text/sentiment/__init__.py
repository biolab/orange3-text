import os
import pickle
from typing import Callable, List, Optional, Set, Tuple

import numpy as np
import serverfiles
from nltk.corpus import opinion_lexicon
from nltk.sentiment import SentimentIntensityAnalyzer
from Orange.misc.environ import data_dir
from Orange.util import dummy_callback, wrap_callback
from requests.exceptions import ConnectionError

from orangecontrib.text import Corpus
from orangecontrib.text.misc import wait_nltk_data
from orangecontrib.text.vectorization.base import (
    SharedTransform,
    VectorizationComputeValue,
)

ScoresType = List[List[float]]


def read_file(file: str) -> List[str]:
    with open(file, 'r', encoding='utf8') as f:
        return f.read().split('\n')


def read_pickle(file: str) -> List[str]:
    with open(file, 'rb') as f:
        return pickle.loads(f.read())


def compute_from_dict(
    tokens: np.ndarray, pos: Set[str], neg: Set[str], callback: Callable
) -> ScoresType:
    scores = []
    for i, doc in enumerate(tokens):
        pos_words = len(pos.intersection(doc)) if pos else 0
        neg_words = len(neg.intersection(doc)) if neg else 0
        scores.append([100 * (pos_words - neg_words) / max(len(doc), 1)])
        callback((i + 1) / len(tokens))
    return scores


class DictionaryNotFound(FileNotFoundError):
    pass


class Sentiment:
    sentiments = None
    name = None

    def get_scores(self, corpus: Corpus, callback: Callable) -> ScoresType:
        return NotImplemented

    def transform(self, corpus: Corpus, callback: Callable = dummy_callback) -> Corpus:
        scores = self.get_scores(corpus, wrap_callback(callback, end=0.9))
        X = np.array(scores).reshape((-1, len(self.sentiments)))

        # set compute values
        shared_cv = SharedTransform(self, corpus.used_preprocessor)
        cv = [VectorizationComputeValue(shared_cv, col) for col in self.sentiments]
        return corpus.extend_attributes(X, self.sentiments, compute_values=cv)


class LiuHuSentiment(Sentiment):
    sentiments = ('sentiment',)  # output column names
    name = 'Liu Hu'

    class SloSentiment:
        resources_folder = os.path.dirname(__file__)

        @classmethod
        def positive(cls) -> List[str]:
            f = os.path.join(
                cls.resources_folder, "resources/positive_words_Slolex.txt"
            )
            return read_file(f)

        @classmethod
        def negative(cls) -> List[str]:
            f = os.path.join(
                cls.resources_folder, "resources/negative_words_Slolex.txt"
            )
            return read_file(f)

    DEFAULT_LANG = "en"
    LANGUAGES = {"en": opinion_lexicon, "sl": SloSentiment}

    @wait_nltk_data
    def __init__(self, language: Optional[str] = None):
        self.language = language

    def get_scores(self, corpus: Corpus, callback: Callable) -> ScoresType:
        language = self.language or corpus.language
        try:
            positive = set(self.LANGUAGES[language].positive())
            negative = set(self.LANGUAGES[language].negative())
        except LookupError:
            raise DictionaryNotFound
        return compute_from_dict(corpus.tokens, positive, negative, callback)


class VaderSentiment(Sentiment):
    sentiments = ("positive", "negative", "neutral", "compound")  # output column names
    name = "Vader"

    @wait_nltk_data
    def __init__(self):
        try:
            self.vader = SentimentIntensityAnalyzer()
        except LookupError:
            raise DictionaryNotFound

    def get_scores(self, corpus: Corpus, callback: Callable) -> ScoresType:
        scores = []
        for i, text in enumerate(corpus.documents):
            pol_sc = self.vader.polarity_scores(text)
            scores.append([pol_sc[x] for x in ("pos", "neg", "neu", "compound")])
            callback((i + 1) / len(corpus))
        return scores


class SentimentDictionaries:
    server_url = None

    def __init__(self):
        self.local_path = os.path.join(data_dir(versioned=False), 'sentiment/')
        self.serverfiles = serverfiles.ServerFiles(self.server_url)
        self.localfiles = serverfiles.LocalFiles(self.local_path,
                                                 serverfiles=self.serverfiles)
        self._supported_languages = []

    def __getitem__(self, language: str) -> List[str]:
        return NotImplemented

    @property
    def lang_files(self):
        try:
            return self.serverfiles.listfiles()
        except ConnectionError:
            return self.localfiles.listfiles()


class MultisentimentDictionaries(SentimentDictionaries):
    server_url = "http://file.biolab.si/files/sentiment/"

    def __init__(self):
        super().__init__()

    def __getitem__(self, language: str) -> Tuple[Set[str], Set[str]]:
        try:
            pos = self.localfiles.localpath_download(f"positive_words_{language}.txt")
            pos = set(read_file(pos))
            neg = self.localfiles.localpath_download(f"negative_words_{language}.txt")
            neg = set(read_file(neg))
        except (FileNotFoundError, ConnectionError):
            raise DictionaryNotFound
        return pos, neg


class MultiSentiment(Sentiment):
    sentiments = ('sentiment',)
    name = 'Multilingual Sentiment'

    # fmt: off
    DEFAULT_LANG = "en"
    LANGUAGES = [
        'af', 'ar', 'az', 'be',  'bg', 'bs',  'ca',  'cs', 'da',  'de', 'el',
        'es', 'et',  'en', 'fa',  'fi',  'fr', 'ga', 'he', 'hi',  'hr',  'hu',
        'id', 'it', 'ja', 'ko', 'la',  'lt', 'lv', 'mk', 'nl',  'nn', 'no', 'pl',
        'pt', 'ro', 'ru', 'sk', 'sl', 'sr', 'sv', 'tr', 'uk', 'zh', 'zh_char'
    ]
    # fmt: on

    def __init__(self, language: Optional[str] = None):
        self.language = language

    def get_scores(self, corpus: Corpus, callback: Callable) -> ScoresType:
        language = self.language or corpus.language
        positive, negative = MultisentimentDictionaries()[language]
        return compute_from_dict(corpus.tokens, positive, negative, callback)


class SentiArtDictionaries(SentimentDictionaries):
    server_url = "http://file.biolab.si/files/sentiart/"

    def __init__(self):
        super().__init__()

    def __getitem__(self, language: str) -> List[str]:
        try:
            filtering_dict = read_pickle(
                self.localfiles.localpath_download(f"SentiArt_{language}.pickle")
            )
        except (FileNotFoundError, ConnectionError):
            raise DictionaryNotFound
        return filtering_dict


class SentiArt(Sentiment):
    sentiments = ('sentiment', 'anger', 'fear', 'disgust', 'happiness',
                  'sadness', 'surprise')
    name = 'SentiArt'

    DEFAULT_LANG = "en"
    LANGUAGES = ["en", "de"]

    def __init__(self, language: Optional[str] = None):
        self.language = language

    def get_scores(self, corpus: Corpus, callback: Callable) -> ScoresType:
        language = self.language or corpus.language
        dictionary = SentiArtDictionaries()[language.upper()]
        scores = []
        for i, doc in enumerate(corpus.tokens):
            score = np.array(
                [list(dictionary[word].values()) for word in doc if word in dictionary]
            )
            scores.append(
                score.mean(axis=0)
                if score.shape[0] > 0
                else np.zeros(len(self.sentiments))
            )
            callback((i + 1) / len(corpus))
        return scores


class LilahDictionaries(SentimentDictionaries):
    server_url = "http://file.biolab.si/files/sentiment-lilah/"

    def __init__(self):
        super().__init__()

    def __getitem__(self, language: str) -> List[str]:
        try:
            filtering_dict = read_pickle(
                self.localfiles.localpath_download(f"LiLaH-{language}.pickle")
            )
        except (FileNotFoundError, ConnectionError):
            raise DictionaryNotFound
        return filtering_dict


class LilahSentiment(Sentiment):
    sentiments = ('Positive', 'Negative', 'Anger', 'Anticipation', 'Disgust',
                  'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust')
    name = 'LiLaH Sentiment'

    DEFAULT_LANG = "sl"
    LANGUAGES = ["hr", "nl", "sl"]

    def __init__(self, language: Optional[str] = None):
        self.language = language

    def get_scores(self, corpus: Corpus, callback: Callable) -> ScoresType:
        language = self.language or corpus.language
        dictionary = LilahDictionaries()[language.upper()]
        scores = []
        for i, doc in enumerate(corpus.tokens):
            score = np.array(
                [list(dictionary[word].values()) for word in doc if word in dictionary]
            )
            scores.append(
                score.mean(axis=0)
                if score.shape[0] > 0
                else np.zeros(len(self.sentiments))
            )
            callback((i + 1) / len(corpus))
        return scores


class CustomDictionaries(Sentiment):
    sentiments = ('sentiment',)
    name = 'Custom Dictionaries'

    def __init__(self, pos: Optional[str], neg: Optional[str]):
        self.positive = set(read_file(pos)) if pos else None
        self.negative = set(read_file(neg)) if neg else None

    def get_scores(self, corpus: Corpus, callback: Callable) -> ScoresType:
        return compute_from_dict(corpus.tokens, self.positive, self.negative, callback)


if __name__ == "__main__":
    c = Corpus.from_file("deerwester")
    liu = LiuHuSentiment()
    c2 = liu.transform(c[:5])
