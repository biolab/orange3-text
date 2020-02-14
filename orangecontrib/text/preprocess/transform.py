from typing import Callable
import re

from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import strip_accents_unicode

from Orange.util import wrap_callback, dummy_callback

from orangecontrib.text import Corpus
from orangecontrib.text.preprocess import Preprocessor

__all__ = ['BaseTransformer', 'HtmlTransformer', 'LowercaseTransformer',
           'StripAccentsTransformer', 'UrlRemover', 'BASE_TRANSFORMER']


class BaseTransformer(Preprocessor):
    def __call__(self, corpus: Corpus, callback: Callable = None) -> Corpus:
        corpus = super().__call__(corpus)
        if callback is None:
            callback = dummy_callback
        callback(0, "Transforming...")
        corpus = self._store_documents(corpus, wrap_callback(callback, end=0.5))
        return self._store_tokens(corpus, wrap_callback(callback, start=0.5)) \
            if corpus.has_tokens() else corpus


class LowercaseTransformer(BaseTransformer):
    """ Converts all characters to lowercase. """
    name = 'Lowercase'

    def _preprocess(self, string: str) -> str:
        return string.lower()


class StripAccentsTransformer(BaseTransformer):
    """ Removes accents. """
    name = "Remove accents"

    def _preprocess(self, string: str) -> str:
        return strip_accents_unicode(string)


class HtmlTransformer(BaseTransformer):
    """ Removes all html tags from string. """
    name = "Parse html"

    def _preprocess(self, string: str) -> str:
        return BeautifulSoup(string, 'html.parser').getText()


class UrlRemover(BaseTransformer):
    """ Removes hyperlinks. """
    name = "Remove urls"
    urlfinder = None

    def __call__(self, corpus: Corpus, callback: Callable = None) -> Corpus:
        self.urlfinder = re.compile(r"((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)")
        corpus = super().__call__(corpus, callback)
        self.urlfinder = None
        return corpus

    def _preprocess(self, string: str) -> str:
        assert self.urlfinder is not None
        return self.urlfinder.sub('', string)


BASE_TRANSFORMER = LowercaseTransformer()
