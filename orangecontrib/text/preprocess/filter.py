from typing import List, Callable
import os
import re

from gensim import corpora
from nltk.corpus import stopwords

from Orange.data.io import detect_encoding
from Orange.util import wrap_callback, dummy_callback

from orangecontrib.text import Corpus
from orangecontrib.text.misc import wait_nltk_data
from orangecontrib.text.preprocess import TokenizedPreprocessor

__all__ = ['BaseTokenFilter', 'StopwordsFilter', 'LexiconFilter',
           'RegexpFilter', 'FrequencyFilter', 'MostFrequentTokensFilter']


class BaseTokenFilter(TokenizedPreprocessor):
    def __call__(self, corpus: Corpus, callback: Callable = None) -> Corpus:
        if callback is None:
            callback = dummy_callback
        corpus = super().__call__(corpus, wrap_callback(callback, end=0.2))
        return self._filter_tokens(corpus, wrap_callback(callback, start=0.2))

    def _filter_tokens(self, corpus: Corpus, callback: Callable) -> Corpus:
        callback(0, "Filtering...")
        tokens = [self._preprocess(tokens) for tokens in corpus.tokens]
        corpus.store_tokens(tokens)
        return corpus

    def _preprocess(self, tokens: List) -> List:
        return list(filter(self._check, tokens))

    def _check(self, token: str) -> bool:
        raise NotImplementedError


class FileWordListMixin:
    def __init__(self, path: str = None):
        self._lexicon = self.from_file(path)

    @staticmethod
    def from_file(path):
        if not path:
            return []

        for encoding in ('utf-8', None, detect_encoding(path)):
            try:
                with open(path, encoding=encoding) as f:
                    return set(line.strip() for line in f)
            except UnicodeDecodeError:
                continue
        # No encoding worked, raise
        raise UnicodeError("Couldn't determine file encoding")


class StopwordsFilter(BaseTokenFilter, FileWordListMixin):
    """ Remove tokens present in NLTK's language specific lists or a file. """
    name = 'Stopwords'

    @wait_nltk_data
    def __init__(self, language='English', path: str = None):
        super().__init__()
        FileWordListMixin.__init__(self, path)
        self.__stopwords = set(x.strip() for x in
                               stopwords.words(language.lower())) \
            if language else []

    @staticmethod
    @wait_nltk_data
    def supported_languages():
        # get NLTK list of stopwords
        stopwords_listdir = []
        try:
            stopwords_listdir = [file for file in
                                 os.listdir(stopwords._get_root())
                                 if file.islower()]
        except LookupError:  # when no NLTK data is available
            pass

        return sorted(file.capitalize() for file in stopwords_listdir)

    def _check(self, token):
        return token not in self.__stopwords and token not in self._lexicon


class LexiconFilter(BaseTokenFilter, FileWordListMixin):
    """ Keep only tokens present in a file. """
    name = 'Lexicon'

    def _check(self, token):
        return not self._lexicon or token in self._lexicon


class RegexpFilter(BaseTokenFilter):
    """ Remove tokens matching this regular expressions. """
    name = 'Regexp'

    def __init__(self, pattern=r'\.|,|:|!|\?'):
        self._pattern = pattern
        # Compiled Regexes are NOT deepcopy-able and hence to make Corpus deepcopy-able
        # we cannot store then (due to Corpus also storing used_preprocessor for BoW compute values).
        # To bypass the problem regex is compiled before every __call__ and discarded right after.
        self.regex = None

    def __call__(self, corpus: Corpus, callback: Callable = None) -> Corpus:
        self.regex = re.compile(self._pattern)
        corpus = super().__call__(corpus, callback)
        self.regex = None
        return corpus

    @staticmethod
    def validate_regexp(regexp):
        try:
            re.compile(regexp)
            return True
        except re.error:
            return False

    def _check(self, token):
        return not self.regex.match(token)


class FitDictionaryFilter(BaseTokenFilter):
    def __init__(self):
        self._lexicon = None
        self._dictionary = None

    def __call__(self, corpus: Corpus, callback: Callable = None) -> Corpus:
        if callback is None:
            callback = dummy_callback
        corpus = TokenizedPreprocessor.__call__(
            self, corpus, wrap_callback(callback, end=0.2))
        callback(0.2, "Fitting filter...")
        self._fit(corpus)
        return self._filter_tokens(corpus, wrap_callback(callback, start=0.6))

    def _fit(self, corpus: Corpus):
        raise NotImplemented

    def _filter_tokens(self, corpus: Corpus, callback: Callable) -> Corpus:
        callback(0, "Filtering...")
        tokens = [self._preprocess(tokens) for tokens in corpus.tokens]
        corpus.store_tokens(tokens, self._dictionary)
        return corpus

    def _check(self, token):
        assert self._lexicon is not None
        assert self._dictionary is not None
        return token in self._lexicon


class FrequencyFilter(FitDictionaryFilter):
    """Remove tokens with document frequency outside this range;
    use either absolute or relative frequency. """
    name = 'Document frequency'

    def __init__(self, min_df=0., max_df=1.):
        super().__init__()
        self._corpus_len = 0
        self._max_df = max_df
        self._min_df = min_df

    def _fit(self, corpus: Corpus):
        self._corpus_len = len(corpus)
        self._dictionary = corpora.Dictionary(corpus.tokens)
        self._dictionary.filter_extremes(self.min_df, self.max_df, None)
        self._lexicon = list(self._dictionary.token2id.keys())

    @property
    def max_df(self):
        if isinstance(self._max_df, int):
            return self._max_df / self._corpus_len if self._corpus_len else 1.
        else:
            return self._max_df

    @property
    def min_df(self):
        if isinstance(self._min_df, float):
            return int(self._corpus_len * self._min_df) or 1
        else:
            return self._min_df


class MostFrequentTokensFilter(FitDictionaryFilter):
    """Keep most frequent tokens."""
    name = 'Most frequent tokens'

    def __init__(self, keep_n=None):
        super().__init__()
        self._keep_n = keep_n

    def _fit(self, corpus: Corpus):
        self._dictionary = corpora.Dictionary(corpus.tokens)
        self._dictionary.filter_extremes(0, 1, self._keep_n)
        self._lexicon = list(self._dictionary.token2id.keys())
