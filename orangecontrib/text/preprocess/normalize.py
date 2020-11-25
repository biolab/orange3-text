from typing import List, Callable
import os
import json
import ufal.udpipe as udpipe
import serverfiles
from nltk import stem
from requests.exceptions import ConnectionError

from Orange.misc.environ import data_dir
from Orange.util import wrap_callback, dummy_callback

from orangecontrib.text import Corpus
from orangecontrib.text.misc import wait_nltk_data
from orangecontrib.text.preprocess import Preprocessor, TokenizedPreprocessor

__all__ = ['BaseNormalizer', 'WordNetLemmatizer', 'PorterStemmer',
           'SnowballStemmer', 'UDPipeLemmatizer']


class BaseNormalizer(TokenizedPreprocessor):
    """ A generic normalizer class.
    You should either overwrite `normalize` method or provide a custom
    normalizer.
    """
    normalizer = NotImplemented

    def __call__(self, corpus: Corpus, callback: Callable = None) -> Corpus:
        if callback is None:
            callback = dummy_callback
        corpus = super().__call__(corpus, wrap_callback(callback, end=0.2))
        callback(0.2, "Normalizing...")
        return self._store_tokens(corpus, wrap_callback(callback, start=0.2))

    def _preprocess(self, string: str) -> str:
        """ Normalizes token to canonical form. """
        return self.normalizer(string)


class WordNetLemmatizer(BaseNormalizer):
    name = 'WordNet Lemmatizer'
    normalizer = stem.WordNetLemmatizer().lemmatize

    @wait_nltk_data
    def __init__(self):
        super().__init__()


class PorterStemmer(BaseNormalizer):
    name = 'Porter Stemmer'
    normalizer = stem.PorterStemmer().stem


class SnowballStemmer(BaseNormalizer):
    name = 'Snowball Stemmer'
    supported_languages = [l.capitalize() for l in stem.SnowballStemmer.languages]

    def __init__(self, language='English'):
        self.normalizer = stem.SnowballStemmer(language.lower())

    def _preprocess(self, token):
        return self.normalizer.stem(token)


def language_to_name(language):
    return language.lower().replace(' ', '') + 'ud'


def file_to_name(file):
    return file.replace('-', '').replace('_', '')


def file_to_language(file):
    return file[:file.find('ud')-1]\
        .replace('-', ' ').replace('_', ' ').capitalize()


class UDPipeModels:
    server_url = "https://file.biolab.si/files/udpipe/"

    def __init__(self):
        self.local_data = os.path.join(data_dir(versioned=False), 'udpipe/')
        self.serverfiles = serverfiles.ServerFiles(self.server_url)
        self.localfiles = serverfiles.LocalFiles(self.local_data,
                                                 serverfiles=self.serverfiles)

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
        return list(map(lambda f: file_to_language(f[0]), self.model_files))

    @property
    def online(self):
        try:
            self.serverfiles.listfiles()
            return True
        except ConnectionError:
            return False


class UDPipeStopIteration(StopIteration):
    pass


class UDPipeLemmatizer(BaseNormalizer):
    name = 'UDPipe Lemmatizer'

    def __init__(self, language='English', use_tokenizer=False):
        self.__language = language
        self.__use_tokenizer = use_tokenizer
        self.models = UDPipeModels()
        self.__model = None
        self.__output_format = None

    @property
    def use_tokenizer(self):
        return self.__use_tokenizer

    @property
    def normalizer(self):
        return self.__normalize_document if self.__use_tokenizer \
            else self.__normalize_token

    def __call__(self, corpus: Corpus, callback: Callable = None) -> Corpus:
        try:
            self.__model = udpipe.Model.load(self.models[self.__language])
        except StopIteration:
            raise UDPipeStopIteration

        self.__output_format = udpipe.OutputFormat.newOutputFormat('epe')
        if self.__use_tokenizer:
            corpus = Preprocessor.__call__(self, corpus)
            if callback is None:
                callback = dummy_callback
            callback(0, "Normalizing...")
            return self._store_tokens_from_documents(corpus, callback)
        else:
            return super().__call__(corpus, callback)

    def __normalize_token(self, token: str) -> str:
        sentence = udpipe.Sentence()
        sentence.addWord(token)
        self.__model.tag(sentence, self.__model.DEFAULT)
        output = self.__output_format.writeSentence(sentence)
        return json.loads(output)['nodes'][0]['properties']['lemma']

    def __normalize_document(self, document: str) -> List[str]:
        tokens = []
        tokenizer = self.__model.newTokenizer(self.__model.DEFAULT)
        tokenizer.setText(document)
        error = udpipe.ProcessingError()
        sentence = udpipe.Sentence()
        while tokenizer.nextSentence(sentence, error):
            self.__model.tag(sentence, self.__model.DEFAULT)
            output = self.__output_format.writeSentence(sentence)
            sentence = udpipe.Sentence()
            tokens.extend([t['properties']['lemma']
                           for t in json.loads(output)['nodes']])
        return tokens
