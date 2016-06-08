from nltk import stem

__all__ = ['BaseNormalizer', 'WordNetLemmatizer', 'PorterStemmer',
           'SnowballStemmer', 'DictionaryLookupNormalizer']


class BaseNormalizer:
    name = NotImplemented
    normalizer = NotImplemented

    def __call__(self, tokens):
        """ Normalizes tokens to canonical form. """
        if isinstance(tokens, str):
            return self.normalize(tokens)
        return [self.normalize(token) for token in tokens]

    def normalize(self, token):
        return self.normalizer(token)

    def on_change(self):
        pass

    def __str__(self):
        return self.name


class WordNetLemmatizer(BaseNormalizer):
    name = 'WordNet Lemmatizer'
    normalizer = stem.WordNetLemmatizer().lemmatize


class DictionaryLookupNormalizer(BaseNormalizer):
    """ Normalizes token with a <token: canonical_form> dictionary. """
    name = 'Dictionary Lookup'

    def __init__(self, dictionary):
        super().__init__()
        self._dictionary = dictionary

    @property
    def dictionary(self):
        return self._dictionary

    @dictionary.setter
    def dictionary(self, value):
        self._dictionary = value
        self.on_change()

    def normalize(self, token):
        return self.dictionary.get(token, token)


class PorterStemmer(BaseNormalizer):
    name = 'Porter Stemmer'
    normalizer = stem.PorterStemmer().stem


class SnowballStemmer(BaseNormalizer):
    name = 'Snowball Stemmer'

    supported_languages = stem.SnowballStemmer.languages

    def __init__(self, language='english'):
        self._language = language
        self.normalizer = stem.SnowballStemmer(self.language)

    def normalize(self, token):
        return self.normalizer.stem(token)

    @property
    def language(self):
        return self._language

    @language.setter
    def language(self, value):
        self._language = value
        self.normalizer = stem.SnowballStemmer(self.language)
        self.on_change()

    def __str__(self):
        return '{} ({})'.format(self.name, self.language)
