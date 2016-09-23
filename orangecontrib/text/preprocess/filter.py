import os

import re
from gensim import corpora
from nltk.corpus import stopwords

__all__ = ['BaseTokenFilter', 'StopwordsFilter', 'LexiconFilter', 'RegexpFilter', 'FrequencyFilter']


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

    def __str__(self):
        return self.name


class WordListMixin:
    def __init__(self, word_list=None):
        self.file_path = None
        self.word_list = word_list or []

    def from_file(self, path):
        self.file_path = path
        if not path:
            self.word_list = []
        else:
            with open(path) as f:
                self.word_list = set([line.strip() for line in f])


class StopwordsFilter(BaseTokenFilter, WordListMixin):
    """ Remove tokens present in NLTK's language specific lists or a file. """
    name = 'Stopwords'

    supported_languages = [file.capitalize() for file in os.listdir(stopwords._get_root())
                           if file.islower()]

    def __init__(self, language='English', word_list=None):
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

    def __str__(self):
        config = ''
        config += 'Language: {}, '.format(self.language.capitalize()) if self.language else ''
        config += 'File: {}, '.format(self.file_path) if self.file_path else ''
        return '{} ({})'.format(self.name, config.strip(', '))

    def check(self, token):
        return token not in self.stopwords and token not in self.word_list


class LexiconFilter(BaseTokenFilter, WordListMixin):
    """ Keep only tokens present in a file. """
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


class RegexpFilter(BaseTokenFilter):
    """ Remove tokens matching this regular expressions. """
    name = 'Regexp'

    def __init__(self, pattern=r'\.|,|:|!|\?'):
        self._pattern = pattern
        self.regex = re.compile(self.pattern)

    @property
    def pattern(self):
        return self._pattern

    @pattern.setter
    def pattern(self, value):
        self._pattern = value
        self.regex = re.compile(self.pattern)

    @staticmethod
    def validate_regexp(regexp):
        try:
            re.compile(regexp)
            return True
        except re.error:
            return False

    def check(self, token):
        return not self.regex.match(token)

    def __str__(self):
        return '{} ({})'.format(self.name, self.pattern)


class FrequencyFilter(LexiconFilter):
    """Remove tokens with document frequency outside this range;
    use either absolute or relative frequency. """
    name = 'Document frequency'

    def __init__(self, min_df=0., max_df=1., keep_n=None):
        super().__init__()
        self._corpus_len = 0
        self.keep_n = keep_n
        self._max_df = max_df
        self._min_df = min_df

    def fit_filter(self, corpus):
        self._corpus_len = len(corpus)
        tokens = getattr(corpus, 'tokens', corpus)
        dictionary = corpora.Dictionary(tokens)
        dictionary.filter_extremes(self.min_df, self.max_df, self.keep_n)
        self.word_list = dictionary.token2id.keys()
        return self(tokens), dictionary

    @property
    def max_df(self):
        if isinstance(self._max_df, int):
            return self._max_df / self._corpus_len if self._corpus_len else 1.
        else:
            return self._max_df

    @max_df.setter
    def max_df(self, value):
        self._max_df = value

    @property
    def min_df(self):
        if isinstance(self._min_df, float):
            return int(self._corpus_len * self._min_df) or 1
        else:
            return self._min_df

    @min_df.setter
    def min_df(self, value):
        self._min_df = value

    def __str__(self):
        keep = ', keep {}'.format(self.keep_n) if self.keep_n else ''
        return "{} (range [{}, {}]{})".format(self.name, self._min_df,
                                              self._max_df, keep)

    def check(self, token):
        return token in self.lexicon
