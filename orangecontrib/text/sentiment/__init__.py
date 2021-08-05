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


def compute_from_dict_multilingual(tokens, languages, pos, neg):
    scores = list()
    for tok, lang in zip(tokens, languages):
        pos_words = len(pos[lang].intersection(tok)) if pos[lang] else 0
        neg_words = len(neg[lang].intersection(tok)) if neg[lang] else 0
        scores.append([100 * (pos_words - neg_words) / max(len(tok), 1)])
    return scores


class Sentiment:
    sentiments = None
    name = None

    def get_scores(self, corpus, languages=None):
        return NotImplemented

    def transform(self, corpus, languages=None):
        scores = self.get_scores(corpus, languages=languages)
        X = np.array(scores).reshape((-1, len(self.sentiments)))

        # set compute values
        shared_cv = SharedTransform(self, corpus.used_preprocessor)
        cv = [VectorizationComputeValue(shared_cv, col)
              for col in self.sentiments]

        corpus = corpus.extend_attributes(X, self.sentiments, compute_values=cv)
        return corpus


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

    methods = {'English': opinion_lexicon,
               'Slovenian': SloSentiment}

    @wait_nltk_data
    def __init__(self, language):
        self.language = language
        if isinstance(language, str):
            self.positive = set(self.methods[self.language].positive())
            self.negative = set(self.methods[self.language].negative())
        else:
            self.positive = {
                'en': set(self.methods['English'].positive()),
                'sl': set(self.methods['Slovenian'].positive()),
            }
            self.negative = {
                'en': set(self.methods['English'].negative()),
                'sl': set(self.methods['Slovenian'].negative()),
            }

    def get_scores(self, corpus, languages=None):
        if isinstance(self.language, str):
            return compute_from_dict(corpus.tokens, self.positive, self.negative)
        else:
            return compute_from_dict_multilingual(
                corpus.tokens, languages, self.positive, self.negative)


class VaderSentiment(Sentiment):
    sentiments = ('pos', 'neg', 'neu', 'compound') # output column names
    name = 'Vader'

    @wait_nltk_data
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()

    def get_scores(self, corpus, languages=None):
        scores = []
        for text in corpus.documents:
            pol_sc = self.vader.polarity_scores(text)
            scores.append([pol_sc[x] for x in self.sentiments])
        return scores


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

    # mapping for nicer language labels
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
        self.language = language
        if isinstance(language, str):
            code = self.LANGS[self.language]
            self.positive, self.negative = MultisentimentDictionaries()[code]
        else:
            self.positive = dict()
            self.negative = dict()
            for lang in language:
                pos, neg = MultisentimentDictionaries()[lang]
                self.positive[lang] = pos
                self.negative[lang] = neg

    def get_scores(self, corpus, languages=None):
        if isinstance(self.language, str):
            return compute_from_dict(corpus.tokens, self.positive, self.negative)
        else:
            return compute_from_dict_multilingual(
                corpus.tokens, languages, self.positive, self.negative)

    def __getstate__(self):
        return {'language': self.language}

    def __setstate__(self, state):
        self.__init__(state['language'])


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

    LANGS = {'English': 'EN', 'German': 'DE'}

    def __init__(self, language='English'):
        self.language = language
        if isinstance(language, str):
            self.dictionary = SentiArtDictionaries()[self.LANGS[self.language]]
        else:
            self.dictionaries = {
                'en': SentiArtDictionaries()['EN'],
                'de': SentiArtDictionaries()['DE'],
            }

    def get_scores(self, corpus, languages=None):
        scores = []
        if isinstance(self.language, str):
            for doc in corpus.tokens:
                score = np.array([list(self.dictionary[word].values()) for word in
                                  doc if word in self.dictionary])
                scores.append(
                    score.mean(axis=0) if score.shape[0] > 0
                    else np.zeros(len(self.sentiments))
                )
        else:
            for tok, lang in zip(corpus.tokens, languages):
                score = np.array([list(self.dictionaries[lang][word].values()) for word in
                                  tok if word in self.dictionaries[lang]])
                scores.append(
                    score.mean(axis=0) if score.shape[0] > 0
                    else np.zeros(len(self.sentiments))
                )
        return scores


class CustomDictionaries(Sentiment):
    sentiments = ('sentiment',)
    name = 'Custom Dictionaries'

    @wait_nltk_data
    def __init__(self, pos, neg):
        self.positive = set(read_file(pos)) if pos else None
        self.negative = set(read_file(neg)) if neg else None

    def get_scores(self, corpus, languages=None):
        return compute_from_dict(corpus.tokens, self.positive, self.negative)


if __name__ == "__main__":
    c = Corpus.from_file('deerwester')
    liu = LiuHuSentiment('Slovenian')
    c2 = liu.transform(c[:5])
