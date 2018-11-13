import os
import json
import ufal.udpipe as udpipe
import serverfiles
from nltk import stem
from requests.exceptions import ConnectionError
from Orange.misc.environ import data_dir


from orangecontrib.text.misc import wait_nltk_data

__all__ = ['BaseNormalizer', 'WordNetLemmatizer', 'PorterStemmer',
           'SnowballStemmer', 'DictionaryLookupNormalizer',
           'UDPipeLemmatizer']


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

    @wait_nltk_data
    def __init__(self):
        super().__init__()


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


def language_to_name(language):
    return language.lower().replace(' ', '') + 'ud'


def file_to_name(file):
    return file.replace('-', '').replace('_', '')


def file_to_language(file):
    return file[:file.find('ud')-1]\
        .replace('-', ' ').replace('_', ' ').capitalize()


class UDPipeModels:
    server_url = "http://file.biolab.si/files/udpipe/"

    def __init__(self):
        self.local_data = os.path.join(data_dir(versioned=False), 'udpipe/')
        self.serverfiles = serverfiles.ServerFiles(self.server_url)
        self.localfiles = serverfiles.LocalFiles(self.local_data,
                                                 serverfiles=self.serverfiles)
        self._supported_languages = []

    def __getitem__(self, language):
        file_name = self._find_file(language_to_name(language))
        return self.localfiles.localpath_download(file_name)

    @property
    def model_files(self):
        try:
            return self.serverfiles.listfiles()
        except ConnectionError:
            return self.localfiles.listfiles()

    def _find_file(self, language):
        return next(filter(lambda f: file_to_name(f).startswith(language),
                           map(lambda f: f[0], self.model_files)))

    @property
    def supported_languages(self):
        self._supported_languages = list(map(lambda f: file_to_language(f[0]),
                                             self.model_files))
        return self._supported_languages

    @property
    def online(self):
        try:
            self.serverfiles.listfiles()
            return True
        except ConnectionError:
            return False


class UDPipeLemmatizer(BaseNormalizer):
    name = 'UDPipe Lemmatizer'
    str_format = '{self.name} ({self.language})'

    def __init__(self, language='English'):
        self._language = language
        self.models = UDPipeModels()
        self.model = None
        self.output_format = udpipe.OutputFormat.newOutputFormat('epe')
        self.use_tokenizer = False

    def load_model(self):
        if self.model is None:
            self.model = udpipe.Model.load(self.models[self._language])

    def normalize(self, token):
        self.load_model()
        sentence = udpipe.Sentence()
        sentence.addWord(token)
        self.model.tag(sentence, self.model.DEFAULT)
        output = self.output_format.writeSentence(sentence)
        return json.loads(output)['nodes'][0]['properties']['lemma']

    def normalize_doc(self, document):
        self.load_model()
        tokens = []
        tokenizer = self.model.newTokenizer(self.model.DEFAULT)
        tokenizer.setText(document)
        error = udpipe.ProcessingError()
        sentence = udpipe.Sentence()
        while tokenizer.nextSentence(sentence, error):
            self.model.tag(sentence, self.model.DEFAULT)
            output = self.output_format.writeSentence(sentence)
            sentence = udpipe.Sentence()
            tokens.extend([t['properties']['lemma']
                           for t in json.loads(output)['nodes']])
        return tokens

    @property
    def language(self):
        return self._language

    @language.setter
    def language(self, value):
        self._language = value
        self.model = None
