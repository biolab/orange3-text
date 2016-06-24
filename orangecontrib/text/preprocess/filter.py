import os

from gensim import corpora
from nltk.corpus import stopwords

__all__ = ['BaseTokenFilter', 'StopwordsFilter', 'LexiconFilter', 'FrequencyFilter']


class BaseTokenFilter:
    name = NotImplemented

    def __call__(self, corpus):
        if len(corpus) == 0:
            return corpus
        if isinstance(corpus[0], str):
            return self.filter(corpus)
        return [self.filter(tokens) for tokens in corpus]

    def filter(self, tokens):
        return list(filter(self.check, tokens))

    def check(self, token):
        raise NotImplementedError

    def on_change(self):
        pass

    def __str__(self):
        return self.name


class WordListMixin:
    def __init__(self, word_list=None):
        self.file_path = None
        self._word_list = word_list or []

    @property
    def word_list(self):
        return self._word_list

    @word_list.setter
    def word_list(self, value):
        self._word_list = value
        self.on_change()

    def from_file(self, path):
        self.file_path = path
        if not path:
            self.word_list = []
        else:
            with open(path) as f:
                self.word_list = [line.strip() for line in f]


class StopwordsFilter(BaseTokenFilter, WordListMixin):
    name = 'Stopwords'

    supported_languages = [file for file in os.listdir(stopwords._get_root())
                           if file.islower()]

    def __init__(self, language='english', word_list=None):
        WordListMixin.__init__(self, word_list)
        super().__init__()
        self.language = language

    @property
    def language(self):
        return self._language

    @language.setter
    def language(self, value):
        self._language = value
        if not self._language:
            self.stopwords = []
        else:
            self.stopwords = set(stopwords.words(self.language.lower()))
        self.on_change()

    def __str__(self):
        config = ''
        config += 'Language: {}, '.format(self.language.capitalize()) if self.language else ''
        config += 'File: {}, '.format(self.file_path) if self.file_path else ''
        return '{} ({})'.format(self.name, config.strip(', '))

    def check(self, token):
        return token not in self.stopwords and token not in self.word_list


class LexiconFilter(BaseTokenFilter, WordListMixin):
    name = 'Lexicon'

    def __init__(self, lexicon=None):
        WordListMixin.__init__(self, word_list=lexicon)

    @property
    def lexicon(self):
        return self.word_list

    @lexicon.setter
    def lexicon(self, value):
        self.word_list = value

    def check(self, token):
        return not self.lexicon or token in self.lexicon

    def __str__(self):
        return '{} ({})'.format(self.name, 'File: {}'.format(self.file_path))


class FrequencyFilter(LexiconFilter):
    name = 'Document frequency'
    tooltip = 'Use either absolute or relative frequency.'

    def __init__(self, min_df=0., max_df=1., keep_n=None):
        super().__init__()
        self._corpus_len = 0
        self._keep_n = keep_n
        self._max_df = max_df
        self._min_df = min_df

    def fit_filter(self, corpus):
        self._corpus_len = len(corpus)
        tokens = getattr(corpus, 'tokens', corpus)
        dictionary = corpora.Dictionary(tokens)
        dictionary.filter_extremes(self.min_df, self.max_df, self.keep_n)
        self._word_list = dictionary.token2id.keys()
        return self(tokens), dictionary

    @property
    def keep_n(self):
        return self._keep_n

    @keep_n.setter
    def keep_n(self, value):
        self._keep_n = value
        self.on_change()

    @property
    def max_df(self):
        if isinstance(self._max_df, int):
            return self._max_df / self._corpus_len if self._corpus_len else 1.
        else:
            return self._max_df

    @max_df.setter
    def max_df(self, value):
        self._max_df = value
        self.on_change()

    @property
    def min_df(self):
        if isinstance(self._min_df, float):
            return int(self._corpus_len * self._min_df) or 1
        else:
            return self._min_df

    @min_df.setter
    def min_df(self, value):
        self._min_df = value
        self.on_change()

    def __str__(self):
        keep = ', keep {}'.format(self.keep_n) if self.keep_n else ''
        return "{} (range [{}, {}]{})".format(self.name, self._min_df,
                                              self._max_df, keep)

    def check(self, token):
        return token in self.lexicon
