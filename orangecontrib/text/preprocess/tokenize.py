from typing import List, Callable
import re
from nltk import tokenize

from Orange.util import wrap_callback, dummy_callback

from orangecontrib.text import Corpus
from orangecontrib.text.misc import wait_nltk_data
from orangecontrib.text.preprocess import Preprocessor

__all__ = ['BaseTokenizer', 'WordPunctTokenizer', 'PunktSentenceTokenizer',
           'RegexpTokenizer', 'WhitespaceTokenizer', 'TweetTokenizer',
           'BASE_TOKENIZER']


class BaseTokenizer(Preprocessor):
    tokenizer = NotImplemented

    def __call__(self, corpus: Corpus, callback: Callable = None) -> Corpus:
        corpus = super().__call__(corpus)
        if callback is None:
            callback = dummy_callback
        callback(0, "Tokenizing...")
        return self._store_tokens_from_documents(corpus, callback)

    def _preprocess(self, string: str) -> List[str]:
        return list(filter(lambda x: x != '', self.tokenizer.tokenize(string)))


class WordPunctTokenizer(BaseTokenizer):
    """ Split by words and (keep) punctuation. """
    tokenizer = tokenize.WordPunctTokenizer()
    name = 'Word & Punctuation'


class PunktSentenceTokenizer(BaseTokenizer):
    """ Split by full-stop, keeping entire sentences. """
    tokenizer = tokenize.PunktSentenceTokenizer()
    name = 'Sentence'

    @wait_nltk_data
    def __init__(self):
        super().__init__()


class WhitespaceTokenizer(BaseTokenizer):
    """ Split only by whitespace. """
    tokenizer = tokenize.WhitespaceTokenizer()
    name = 'Whitespace'


class RegexpTokenizer(BaseTokenizer):
    """ Split by regular expression, default keeps only words. """
    tokenizer_cls = tokenize.RegexpTokenizer
    name = 'Regexp'

    def __init__(self, pattern=r'\w+'):
        super().__init__()
        self.tokenizer = None
        self.__pattern = pattern

    def __call__(self, corpus: Corpus, callback: Callable = None) -> Corpus:
        # Compiled Regexes are NOT deepcopy-able and hence to make Corpus deepcopy-able
        # we cannot store then (due to Corpus also storing used_preprocessor for BoW compute values).
        # To bypass the problem regex is compiled before every __call__ and discarded right after.
        self.tokenizer = self.tokenizer_cls(self.__pattern)
        corpus = Preprocessor.__call__(self, corpus)
        if callback is None:
            callback = dummy_callback
        callback(0, "Tokenizing...")
        corpus = self._store_tokens_from_documents(corpus, callback)
        self.tokenizer = None
        return corpus

    def _preprocess(self, string: str) -> List[str]:
        assert self.tokenizer is not None
        return super()._preprocess(string)

    @staticmethod
    def validate_regexp(regexp: str) -> bool:
        try:
            re.compile(regexp)
            return True
        except re.error:
            return False


class TweetTokenizer(BaseTokenizer):
    """ Pre-trained tokenizer for tweets. """
    tokenizer = tokenize.TweetTokenizer()
    name = 'Tweet'


BASE_TOKENIZER = WordPunctTokenizer()
