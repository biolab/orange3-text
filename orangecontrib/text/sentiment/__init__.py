import os
import re

import serverfiles
from requests.exceptions import ConnectionError

import numpy as np
from nltk.corpus import opinion_lexicon
from nltk.sentiment import SentimentIntensityAnalyzer

from Orange.misc.environ import data_dir
from orangecontrib.text.misc import wait_nltk_data

from orangecontrib.text import Corpus
from orangecontrib.text.preprocess import WordPunctTokenizer
from orangecontrib.text.vectorization.base import SharedTransform, \
    VectorizationComputeValue
from orangecontrib.text.sentiment.filter_lexicon import FilterSentiment, \
    SloSentiment


class LiuHuSentiment:
    sentiments = ('sentiment',)
    name = 'Liu Hu'

    methods = {'English': opinion_lexicon,
               'Slovenian': SloSentiment}

    @wait_nltk_data
    def __init__(self, language):
        self.language = language
        self.positive = set(self.methods[self.language].positive())
        self.negative = set(self.methods[self.language].negative())

    def transform(self, corpus, copy=True):
        scores = []
        tokenizer = WordPunctTokenizer()
        tokens = tokenizer(corpus.documents)

        for doc in tokens:
            pos_words = sum(word in self.positive for word in doc)
            neg_words = sum(word in self.negative for word in doc)
            scores.append([100*(pos_words - neg_words)/max(len(doc), 1)])
        X = np.array(scores).reshape((-1, len(self.sentiments)))

        # set  compute values
        shared_cv = SharedTransform(self)
        cv = [VectorizationComputeValue(shared_cv, col)
              for col in self.sentiments]

        if copy:
            corpus = corpus.copy()
        corpus.extend_attributes(X, self.sentiments, compute_values=cv)
        return corpus


class VaderSentiment:
    sentiments = ('pos', 'neg', 'neu', 'compound')
    name = 'Vader'

    @wait_nltk_data
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()

    def transform(self, corpus, copy=True):
        scores = []
        for text in corpus.documents:
            pol_sc = self.vader.polarity_scores(text)
            scores.append([pol_sc[x] for x in self.sentiments])
        X = np.array(scores).reshape((-1, len(self.sentiments)))

        # set  compute values
        shared_cv = SharedTransform(self)
        cv = [VectorizationComputeValue(shared_cv, col)
              for col in self.sentiments]

        if copy:
            corpus = corpus.copy()
        corpus.extend_attributes(X, self.sentiments, compute_values=cv)
        return corpus


class MultiSentiment:
    sentiments = ('sentiment',)
    name = 'Multilingual Sentiment'

    LANGS = {'Afrikaans': 'af', 'Arabic': 'ar', 'Azerbaijani': 'az',
             'Belarusian': 'be', 'Bulgarian': 'bg',
             'Bosnian': 'bs', 'Catalan': 'ca', 'Czech': 'cs',
             'Danish': 'da', 'German': 'de', 'Greek': 'el',
             'Spanish': 'es', 'Estonian': 'et', 'English': 'en',
             'Farsi': 'fa', 'Finnish': 'fi', 'French': 'fr', 'Gaelic': 'ga',
             'Hebrew': 'he', 'Hindi': 'hi', 'Croatian': 'hr', 'Hungarian': 'hu',
             'Indonesian': 'id', 'Italian': 'it', 'Japanese': 'ja',
             'Korean': 'ko', 'Latin': 'la', 'Lithuanian': 'lt', 'Latvian': 'lv',
             'Macedonian': 'mk', 'Dutch': 'nl', 'Norwegian Nynorsk': 'nn',
             'Norwegian': 'no', 'Polish': 'pl', 'Portuguese': 'pt',
             'Romanian': 'ro', 'Russian': 'ru', 'Slovak': 'sk', 'Slovene': 'sl',
             'Serbian': 'sr', 'Swedish': 'sv', 'Turkish': 'tr',
             'Ukrainian': 'uk', 'Chinese': 'zh',
             'Chinese Characters': 'zhw'}

    def __init__(self, language='English'):
        self._language = language
        self.dictionaries = SentimentDictionaries()
        self.positive = None
        self.negative = None

    def load_dict(self):
        if self.positive is None or self.negative is None:
            code = self.LANGS[self._language]
            self.positive, self.negative = self.dictionaries[code]

    def transform(self, corpus, copy=True):
        self.load_dict()
        scores = []
        tokenizer = WordPunctTokenizer()
        tokens = tokenizer(corpus.documents)

        for doc in tokens:
            pos_words = sum(word in self.positive for word in doc)
            neg_words = sum(word in self.negative for word in doc)
            scores.append([100 * (pos_words - neg_words) / max(len(doc), 1)])
        X = np.array(scores).reshape((-1, len(self.sentiments)))

        # set  compute values
        shared_cv = SharedTransform(self)
        cv = [VectorizationComputeValue(shared_cv, col)
              for col in self.sentiments]

        if copy:
            corpus = corpus.copy()
        corpus.extend_attributes(X, self.sentiments, compute_values=cv)
        return corpus

    @property
    def language(self):
        return self._language

    @language.setter
    def language(self, value):
        self._language = value
        self.positive = None
        self.negative = None

    def __getstate__(self):
        return {'language': self.language}

    def __setstate__(self, state):
        self.__init__(state['language'])


class SentimentDictionaries:
    server_url = "http://file.biolab.si/files/sentiment/"

    def __init__(self):
        self.local_data = os.path.join(data_dir(versioned=False), 'sentiment/')
        self.serverfiles = serverfiles.ServerFiles(self.server_url)
        self.localfiles = serverfiles.LocalFiles(self.local_data,
                                                 serverfiles=self.serverfiles)
        self._supported_languages = []

    def __getitem__(self, language):
        pos = set(FilterSentiment.read_file(self.localfiles.localpath_download(
            f"positive_words_{language}.txt")))
        neg = set(FilterSentiment.read_file(self.localfiles.localpath_download(
            f"negative_words_{language}.txt")))
        return pos, neg

    @property
    def lang_files(self):
        try:
            return self.serverfiles.listfiles()
        except ConnectionError:
            return self.localfiles.listfiles()

    @property
    def supported_languages(self):
        regex = "(?<=negative_words_|positive_words_)(.*)(?=\.txt)"
        self._supported_languages = [re.search(regex, i[0]).group(0) for i in
                                     list(self.lang_files)]
        return self._supported_languages

    @property
    def online(self):
        try:
            self.serverfiles.listfiles()
            return True
        except ConnectionError:
            return False


if __name__ == "__main__":
    c = Corpus.from_file('deerwester')
    liu = LiuHuSentiment('Slovenian')
    c2 = liu.transform(c[:5])
