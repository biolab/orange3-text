import re
from nltk import tokenize


__all__ = ['BaseTokenizer', 'WordPunctTokenizer', 'PunktSentenceTokenizer',
           'RegexpTokenizer', 'WhitespaceTokenizer', 'TweetTokenizer']


class BaseTokenizer:
    """ Breaks a string into sequence of tokens. """
    tokenizer = NotImplemented
    name = 'Tokenizer'

    def __call__(self, sent):
        if isinstance(sent, str):
            return self.tokenize(sent)
        return self.tokenize_sents(sent)

    def tokenize(self, string):
        return list(filter(lambda x: x != '', self.tokenizer.tokenize(string)))

    def tokenize_sents(self, strings):
        return [self.tokenize(string) for string in strings]

    def __str__(self):
        return self.name

    def on_change(self):
        pass


class WordPunctTokenizer(BaseTokenizer):
    """ Split by words and (keep) punctuation. """
    tokenizer = tokenize.WordPunctTokenizer()
    name = 'Word & Punctuation'


class PunktSentenceTokenizer(BaseTokenizer):
    """ Split by full-stop, keeping entire sentences. """
    tokenizer = tokenize.PunktSentenceTokenizer()
    name = 'Sentence'


class WhitespaceTokenizer(BaseTokenizer):
    """ Split only by whitespace. """
    tokenizer = tokenize.WhitespaceTokenizer()
    name = 'Whitespace'


class RegexpTokenizer(BaseTokenizer):
    """ Split by regular expression, default keeps only words. """
    name = 'Regexp'

    def __init__(self, pattern=r'\w+'):
        self._pattern = pattern
        self.tokenizer = tokenize.RegexpTokenizer(pattern)

    @property
    def pattern(self):
        return self._pattern

    @pattern.setter
    def pattern(self, value):
        self._pattern = value
        self.tokenizer = tokenize.RegexpTokenizer(self.pattern)
        self.on_change()

    def __str__(self):
        return '{} ({})'.format(self.name, self.pattern)

    @staticmethod
    def validate_regexp(regexp):
        try:
            re.compile(regexp)
            return True
        except re.error:
            return False


class TweetTokenizer(BaseTokenizer):
    """ Pre-trained tokenizer for tweets. """
    tokenizer = tokenize.TweetTokenizer()
    name = 'Tweet'
