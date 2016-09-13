from nltk import stem

__all__ = ['BaseNormalizer', 'WordNetLemmatizer', 'PorterStemmer',
           'SnowballStemmer', 'DictionaryLookupNormalizer']


class BaseNormalizer:
    """ A generic normalizer class.
    You should either overwrite `normalize` method or provide a custom
    normalizer.

    Attributes:
        name(str): A short name for normalization method (will be used in OWPreprocessor)
        normalizer(Callable): An callabale object to be used for normalization.

    """
    name = NotImplemented
    normalizer = NotImplemented
    str_format = '{self.name}'

    def __call__(self, tokens):
        """ Normalizes tokens to canonical form. """
        if isinstance(tokens, str):
            return self.normalize(tokens)
        return [self.normalize(token) for token in tokens]

    def normalize(self, token):
        return self.normalizer(token)

    def __str__(self):
        return self.str_format.format(self=self)


class WordNetLemmatizer(BaseNormalizer):
    name = 'WordNet Lemmatizer'
    normalizer = stem.WordNetLemmatizer().lemmatize


class DictionaryLookupNormalizer(BaseNormalizer):
    """ Normalizes token with a <token: canonical_form> dictionary. """
    name = 'Dictionary Lookup'

    def __init__(self, dictionary):
        super().__init__()
        self.dictionary = dictionary

    def normalize(self, token):
        return self.dictionary.get(token, token)


class PorterStemmer(BaseNormalizer):
    name = 'Porter Stemmer'
    normalizer = stem.PorterStemmer().stem


class SnowballStemmer(BaseNormalizer):
    name = 'Snowball Stemmer'
    str_format = '{self.name} ({self.language})'
    supported_languages = [l.capitalize() for l in stem.SnowballStemmer.languages]

    def __init__(self, language='English'):
        self._language = language
        self.normalizer = stem.SnowballStemmer(self.language.lower())

    def normalize(self, token):
        return self.normalizer.stem(token)

    @property
    def language(self):
        return self._language

    @language.setter
    def language(self, value):
        self._language = value
        self.normalizer = stem.SnowballStemmer(self.language.lower())
