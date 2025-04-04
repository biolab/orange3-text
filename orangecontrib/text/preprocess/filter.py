from itertools import compress
from typing import List, Callable, Optional, Set
import os
import re

import numpy as np
from gensim import corpora
from nltk.corpus import stopwords

from Orange.data.io import detect_encoding
from Orange.util import wrap_callback, dummy_callback

from orangecontrib.text import Corpus
from orangecontrib.text.language import ISO2LANG, LANG2ISO
from orangecontrib.text.misc import wait_nltk_data
from orangecontrib.text.preprocess import TokenizedPreprocessor

__all__ = ['BaseTokenFilter', 'StopwordsFilter', 'LexiconFilter',
           'RegexpFilter', 'FrequencyFilter', 'MostFrequentTokensFilter',
           'PosTagFilter', 'NumbersFilter', 'WithNumbersFilter']


class BaseTokenFilter(TokenizedPreprocessor):
    def __call__(self, corpus: Corpus, callback: Callable = None) -> Corpus:
        if callback is None:
            callback = dummy_callback
        corpus = super().__call__(corpus, wrap_callback(callback, end=0.2))
        return self._filter_tokens(corpus, wrap_callback(callback, start=0.2))

    def _filter_tokens(self, corpus: Corpus, callback: Callable) -> Corpus:
        callback(0, "Filtering...")
        filtered_tokens = []
        filtered_tags = []
        for i, tokens in enumerate(corpus.tokens):
            filter_map = self._preprocess(tokens)
            filtered_tokens.append(list(compress(tokens, filter_map)))
            if corpus.pos_tags is not None:
                filtered_tags.append(list(compress(corpus.pos_tags[i],
                                                   filter_map)))
        corpus.store_tokens(filtered_tokens)
        if filtered_tags:
            corpus.pos_tags = np.array(filtered_tags, dtype=object)
        return corpus

    def _preprocess(self, tokens: List) -> List:
        return [self._check(token) for token in tokens]

    def _check(self, token: str) -> bool:
        raise NotImplementedError


class FileWordListMixin:
    def __init__(self, path: str = None):
        self._lexicon = self.from_file(path)

    @staticmethod
    def from_file(path):
        if not path:
            return set()

        for encoding in ('utf-8-sig', None, detect_encoding(path)):
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

    # nltk uses different language nams for some languages
    LANG2NLTK = {"Slovenian": "Slovene"}
    NLTK2LANG = {v: k for k, v in LANG2NLTK.items()}

    def __init__(
        self,
        language: Optional[str] = "en",
        path: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        language
            The language code in ISO format for NLTK stopwords selection.
            If None, only words from file are used (NLTK stopwords are not used).
        path
            The path to the file with its stopwords will be used if present.
            The file must contain a newline-separated list of words.
        """
        super().__init__()
        FileWordListMixin.__init__(self, path)
        self.__stopwords = set()
        if language:
            # transform iso code to NLTK's language name
            language = ISO2LANG[language]
            language = self.LANG2NLTK.get(language, language).lower()
            self.__stopwords = set(x.strip() for x in stopwords.words(language))

    @staticmethod
    def lang_to_iso(language: str) -> str | None:
        """
        Returns the ISO language code for the NLTK language. NLTK have a different name
        for Slovenian. This function takes it into account while transforming to ISO.

        Parameters
        ----------
        language
            NLTK language name
        Returns
        -------
        ISO language code for input language, return None if language is not supported.
        """
        return LANG2ISO.get(StopwordsFilter.NLTK2LANG.get(language, language))

    @classmethod
    @property
    @wait_nltk_data
    def supported_languages(_) -> Set[str]:
        """
        List all languages supported by NLTK

        Returns
        -------
        Set of all languages supported by NLTK
        """
        try:
            return {
                StopwordsFilter.lang_to_iso(file.title())
                for file in os.listdir(stopwords._get_root())
                if file.islower()
            } - {None}
        except LookupError:  # when no NLTK data is available
            return set()

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


class NumbersFilter(BaseTokenFilter):
    """ Remove tokens that are numbers. """
    name = 'Numbers'

    def _check(self, token):
        try:
            float(token)
            return False
        except ValueError:
            return True


class WithNumbersFilter(RegexpFilter):
    """ Remove tokens with numbers. """
    name = 'Includes Numbers'

    def __init__(self):
        super().__init__(r'[0-9]')

    def _check(self, token):
        return not self.regex.findall(token)


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
        return super()._filter_tokens(corpus, callback)

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
        self._lexicon = set(self._dictionary.token2id.keys())

    @property
    def max_df(self):
        if isinstance(self._max_df, int):
            if self._max_df <= self._min_df:
                return 1.
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
        self._lexicon = set(self._dictionary.token2id.keys())


class PosTagFilter(BaseTokenFilter):
    """Keep selected POS tags."""
    name = 'POS tags'

    def __init__(self, tags=None):
        self._tags = set(i.strip().upper() for i in tags.split(","))

    @staticmethod
    def validate_tags(tags):
        # should we keep a dict of existing POS tags and compare them with
        # input?
        return len(tags.split(",")) > 0

    def _filter_tokens(self, corpus: Corpus, callback: Callable) -> Corpus:
        if corpus.pos_tags is None:
            return corpus
        callback(0, "Filtering...")
        filtered_tags = []
        filtered_tokens = []
        for tags, tokens in zip(corpus.pos_tags, corpus.tokens):
            tmp_tags = []
            tmp_tokens = []
            for tag, token in zip(tags, tokens):
                # should we consider partial matches, i.e. "NN" for "NNS"?
                if tag in self._tags:
                    tmp_tags.append(tag)
                    tmp_tokens.append(token)
            filtered_tags.append(tmp_tags)
            filtered_tokens.append(tmp_tokens)
        corpus.store_tokens(filtered_tokens)
        corpus.pos_tags = filtered_tags
        return corpus

    def _check(self, token: str) -> bool:
        pass
