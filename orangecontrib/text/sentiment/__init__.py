import os
import pickle
import re

import serverfiles
from requests.exceptions import ConnectionError

import numpy as np
from nltk.corpus import opinion_lexicon
from nltk.sentiment import SentimentIntensityAnalyzer

from Orange.misc.environ import data_dir
from orangecontrib.text.misc import wait_nltk_data

from orangecontrib.text import Corpus
from orangecontrib.text.vectorization.base import SharedTransform, \
    VectorizationComputeValue


def read_file(file):
    with open(file, 'r', encoding='utf8') as f:
        return f.read().split('\n')


def read_pickle(file):
    with open(file, 'rb') as f:
        return pickle.loads(f.read())


def compute_from_dict(tokens, pos, neg):
    scores = []
    for doc in tokens:
        pos_words = len(pos.intersection(doc)) if pos else 0
        neg_words = len(neg.intersection(doc)) if neg else 0
        scores.append([100 * (pos_words - neg_words) / max(len(doc), 1)])
    return scores


class Sentiment:
    sentiments = None
    name = None

    def get_scores(self, corpus):
        return NotImplemented

    def transform(self, corpus):
        scores = self.get_scores(corpus)
        X = np.array(scores).reshape((-1, len(self.sentiments)))

        # set compute values
        shared_cv = SharedTransform(self, corpus.used_preprocessor)
        cv = [VectorizationComputeValue(shared_cv, col)
              for col in self.sentiments]

        corpus = corpus.extend_attributes(X, self.sentiments, compute_values=cv)
        return corpus

    @staticmethod
    def check_language(language: str) -> bool:
        """
        Check whether the language of the corpus is supported by the method.
        It should override this method if it has limited set of supported languages.

        Parameters
        ----------
        language
            Language to check in ISO standard

        Returns
        -------
        True if language is supported by method False otherwise
        """
        return True  # for methods which does not limit language support


class LiuHuSentiment(Sentiment):
    sentiments = ('sentiment',)  # output column names
    name = 'Liu Hu'

    class SloSentiment:
        resources_folder = os.path.dirname(__file__)

        @classmethod
        def positive(cls):
            f = os.path.join(cls.resources_folder,
                             'resources/positive_words_Slolex.txt')
            return read_file(f)

        @classmethod
        def negative(cls):
            f = os.path.join(cls.resources_folder,
                             'resources/negative_words_Slolex.txt')
            return read_file(f)

    methods = {"en": opinion_lexicon, "sl": SloSentiment}

    @wait_nltk_data
    def __init__(self, language):
        self.language = language
        self.positive = set(self.methods[self.language].positive())
        self.negative = set(self.methods[self.language].negative())

    def get_scores(self, corpus):
        return compute_from_dict(corpus.tokens, self.positive, self.negative)

    @staticmethod
    def check_language(language: str) -> bool:
        return language in LiuHuSentiment.methods


class VaderSentiment(Sentiment):
    sentiments = ("pos", "neg", "neu", "compound")  # output column names
    name = "Vader"

    @wait_nltk_data
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()

    def get_scores(self, corpus):
        scores = []
        for text in corpus.documents:
            pol_sc = self.vader.polarity_scores(text)
            scores.append([pol_sc[x] for x in self.sentiments])
        return scores

    @staticmethod
    def check_language(language: str) -> bool:
        return language == "en"


class SentimentDictionaries:
    server_url = None

    def __init__(self):
        self.local_path = os.path.join(data_dir(versioned=False), 'sentiment/')
        self.serverfiles = serverfiles.ServerFiles(self.server_url)
        self.localfiles = serverfiles.LocalFiles(self.local_path,
                                                 serverfiles=self.serverfiles)
        self._supported_languages = []

    def __getitem__(self, language):
        return NotImplemented

    @property
    def supported_languages(self):
        return NotImplemented

    @property
    def lang_files(self):
        try:
            return self.serverfiles.listfiles()
        except ConnectionError:
            return self.localfiles.listfiles()

    @property
    def online(self):
        try:
            self.serverfiles.listfiles()
            return True
        except ConnectionError:
            return False


class MultisentimentDictionaries(SentimentDictionaries):
    server_url = "http://file.biolab.si/files/sentiment/"

    def __init__(self):
        super().__init__()

    def __getitem__(self, language):
        pos = set(read_file(self.localfiles.localpath_download(
            f"positive_words_{language}.txt")))
        neg = set(read_file(self.localfiles.localpath_download(
            f"negative_words_{language}.txt")))
        return pos, neg

    def supported_languages(self):
        re_pos = r"positive_words_(.*)\.txt"
        re_neg = r"negative_words_(.*)\.txt"
        pos = neg = set()
        for i in self.lang_files:
            res_pos = re.fullmatch(re_pos, i[0])
            res_neg = re.fullmatch(re_neg, i[0])
            if res_pos:
                pos.add(res_pos.group(1))
            elif res_neg:
                neg.add(res_neg.group(1))
        return pos.intersection(neg)


class MultiSentiment(Sentiment):
    sentiments = ('sentiment',)
    name = 'Multilingual Sentiment'

    # fmt: off
    LANGS = {
        'af', 'ar', 'az', 'be',  'bg', 'bs',  'ca',  'cs', 'da',  'de', 'el',
        'es', 'et',  'en', 'fa',  'fi',  'fr', 'ga', 'he', 'hi',  'hr',  'hu',
        'id', 'it', 'ja', 'ko', 'la',  'lt', 'lv', 'mk', 'nl',  'nn', 'no', 'pl',
        'pt', 'ro', 'ru', 'sk', 'sl', 'sr', 'sv', 'tr', 'uk', 'zh', 'zhw'
    }
    # fmt: on

    def __init__(self, language="en"):
        self.positive, self.negative = MultisentimentDictionaries()[language]

    def get_scores(self, corpus):
        return compute_from_dict(corpus.tokens, self.positive, self.negative)

    @staticmethod
    def check_language(language: str) -> bool:
        return language in MultiSentiment.LANGS


class SentiArtDictionaries(SentimentDictionaries):
    server_url = "http://file.biolab.si/files/sentiart/"

    def __init__(self):
        super().__init__()

    def __getitem__(self, language):
        filtering_dict = read_pickle(self.localfiles.localpath_download(
                                     f"SentiArt_{language}.pickle"))
        return filtering_dict

    def supported_languages(self):
        regex = r"SentiArt_(.*)\.pickle"
        supported_languages = set()
        for i in self.lang_files:
            res = re.fullmatch(regex, i[0])
            if res:
                supported_languages.add(res.group(1))
        return supported_languages


class SentiArt(Sentiment):
    sentiments = ('sentiment', 'anger', 'fear', 'disgust', 'happiness',
                  'sadness', 'surprise')
    name = 'SentiArt'

    LANGS = {"en", "de"}

    def __init__(self, language="en"):
        self.language = language
        self.dictionary = SentiArtDictionaries()[language.upper()]

    def get_scores(self, corpus):
        scores = []
        for doc in corpus.tokens:
            score = np.array([list(self.dictionary[word].values()) for word in
                              doc if word in self.dictionary])
            scores.append(score.mean(axis=0) if score.shape[0] > 0
                          else np.zeros(len(self.sentiments)))
        return scores

    @staticmethod
    def check_language(language: str) -> bool:
        return language in SentiArt.LANGS


class LilahDictionaries(SentimentDictionaries):
    server_url = "http://file.biolab.si/files/sentiment-lilah/"

    def __init__(self):
        super().__init__()

    def __getitem__(self, language):
        filtering_dict = read_pickle(self.localfiles.localpath_download(
                                     f"LiLaH-{language}.pickle"))
        return filtering_dict

    def supported_languages(self):
        regex = r"LiLaH-(.*)\.pickle"
        supported_languages = set()
        for i in self.lang_files:
            res = re.fullmatch(regex, i[0])
            if res:
                supported_languages.add(res.group(1))
        return supported_languages


class LilahSentiment(Sentiment):
    sentiments = ('Positive', 'Negative', 'Anger', 'Anticipation', 'Disgust',
                  'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust')
    name = 'LiLaH Sentiment'

    LANGS = {"sl", "hr", "nl"}

    def __init__(self, language="sl"):
        self.language = language
        self.dictionary = LilahDictionaries()[language.upper()]

    def get_scores(self, corpus):
        scores = []
        for doc in corpus.tokens:
            score = np.array([list(self.dictionary[word].values()) for word in
                              doc if word in self.dictionary])
            scores.append(score.mean(axis=0) if score.shape[0] > 0
                          else np.zeros(len(self.sentiments)))
        return scores

    @staticmethod
    def check_language(language: str) -> bool:
        return language in LilahSentiment.LANGS


class CustomDictionaries(Sentiment):
    sentiments = ('sentiment',)
    name = 'Custom Dictionaries'

    @wait_nltk_data
    def __init__(self, pos, neg):
        self.positive = set(read_file(pos)) if pos else None
        self.negative = set(read_file(neg)) if neg else None

    def get_scores(self, corpus):
        return compute_from_dict(corpus.tokens, self.positive, self.negative)


if __name__ == "__main__":
    c = Corpus.from_file("deerwester")
    liu = LiuHuSentiment("sl")
    c2 = liu.transform(c[:5])
