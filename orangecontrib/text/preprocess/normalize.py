from typing import List, Callable
import os
import ufal.udpipe as udpipe
from lemmagen3 import Lemmatizer
import serverfiles
from nltk import stem
from requests.exceptions import ConnectionError

from Orange.misc.environ import data_dir
from Orange.util import wrap_callback, dummy_callback

from orangecontrib.text import Corpus
from orangecontrib.text.language import ISO2LANG
from orangecontrib.text.misc import wait_nltk_data
from orangecontrib.text.preprocess import Preprocessor, TokenizedPreprocessor

__all__ = ['BaseNormalizer', 'WordNetLemmatizer', 'PorterStemmer',
           'SnowballStemmer', 'UDPipeLemmatizer', 'LemmagenLemmatizer']


class BaseNormalizer(TokenizedPreprocessor):
    """ A generic normalizer class.
    You should either overwrite `normalize` method or provide a custom
    normalizer.
    """
    normalizer = NotImplemented

    def __init__(self):
        # cache already normalized string to speedup normalization
        self._normalization_cache = {}

    def __call__(self, corpus: Corpus, callback: Callable = None) -> Corpus:
        if callback is None:
            callback = dummy_callback
        if corpus.language is None or not self._language_supported(corpus.language):
            raise ValueError(
                f"{self.name} does not support the Corpus's language. "
            )
        self.load_model(corpus.language)
        corpus = super().__call__(corpus, wrap_callback(callback, end=0.2))
        callback(0.2, "Normalizing...")
        return self._store_tokens(corpus, wrap_callback(callback, start=0.2))

    def load_model(self, iso_language: str):
        """
        Some normalizers load model when called since language information is
        packed in Corpus
        """
        pass

    def _preprocess(self, string: str) -> str:
        """ Normalizes token to canonical form. """
        if string in self._normalization_cache:
            return self._normalization_cache[string]
        self._normalization_cache[string] = norm_string = self.normalizer(string)
        return norm_string

    def __getstate__(self):
        d = self.__dict__.copy()
        # since cache can be quite big, empty cache before pickling
        d["_normalization_cache"] = {}
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        # support old pickles (before caching was implemented) that are missing
        # _normalization_cache
        self._normalization_cache = {}

    def _language_supported(self, iso_language: str) -> bool:
        raise NotImplementedError


class WordNetLemmatizer(BaseNormalizer):
    name = 'WordNet Lemmatizer'
    normalizer = stem.WordNetLemmatizer().lemmatize

    @wait_nltk_data
    def __init__(self):
        super().__init__()

    def _language_supported(self, iso_language: str) -> bool:
        return iso_language == "en"


class PorterStemmer(BaseNormalizer):
    name = 'Porter Stemmer'
    normalizer = stem.PorterStemmer().stem

    def _language_supported(self, iso_language: str) -> bool:
        return iso_language == "en"


class SnowballStemmer(BaseNormalizer):
    name = 'Snowball Stemmer'

    def load_model(self, iso_language: str):
        self.normalizer = stem.SnowballStemmer(ISO2LANG[iso_language].lower()).stem

    def _language_supported(self, iso_language: str) -> bool:
        return ISO2LANG[iso_language].lower() in stem.SnowballStemmer.languages


def language_to_name(language):
    return language.lower().replace(' ', '') + 'ud'


def file_to_name(file):
    return file.replace('-', '').replace('_', '')


def file_to_language(file):
    return file[:file.find('ud') - 1] \
        .replace('-', ' ').replace('_', ' ').capitalize()


class UDPipeModels:
    server_url = "https://file.biolab.si/files/udpipe/"

    # some languages differ between udpipe and iso standard
    lang2upipe = {
        "Norwegian Bokmål": "Norwegian Bokmaal"
    }

    def __init__(self):
        self.local_data = os.path.join(data_dir(versioned=False), 'udpipe/')
        self.serverfiles = serverfiles.ServerFiles(self.server_url)
        self.localfiles = serverfiles.LocalFiles(self.local_data,
                                                 serverfiles=self.serverfiles)

    def __getitem__(self, language):
        file_name = self._find_file(
            language_to_name(self.lang2upipe.get(language, language))
        )
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

    def language_supported(self, iso_language):
        # capitalization in iso2lang and UDPIPE is different - compare lowercase
        sup_languages = {x.lower() for x in self.supported_languages}
        language = ISO2LANG[iso_language]
        return self.lang2upipe.get(language, language).lower() in sup_languages

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

    def __init__(self, use_tokenizer=False):
        super().__init__()
        self.__use_tokenizer = use_tokenizer
        self.models = UDPipeModels()
        self.__model = None

    @property
    def use_tokenizer(self):
        return self.__use_tokenizer

    @property
    def normalizer(self):
        return self.__normalize_document if self.__use_tokenizer \
            else self.__normalize_token

    def __call__(self, corpus: Corpus, callback: Callable = None) -> Corpus:
        if corpus.language is None or not self._language_supported(corpus.language):
            raise ValueError(
                f"{self.name} does not support the Corpus's language. "
            )
        try:
            self.__model = udpipe.Model.load(self.models[ISO2LANG[corpus.language]])
        except StopIteration:
            raise UDPipeStopIteration

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
        return sentence.words[1].lemma

    def __normalize_document(self, document: str) -> List[str]:
        tokens = []
        tokenizer = self.__model.newTokenizer(self.__model.DEFAULT)
        tokenizer.setText(document)
        error = udpipe.ProcessingError()
        sentence = udpipe.Sentence()
        while tokenizer.nextSentence(sentence, error):
            self.__model.tag(sentence, self.__model.DEFAULT)
            # 1: is used because words[0] is the root required by the dependency trees
            tokens.extend([w.lemma for w in sentence.words[1:]])
            sentence = udpipe.Sentence()
        return tokens

    def __getstate__(self):
        """
        This function remove udpipe.Model that cannot be pickled and models that
        include absolute paths on computer -- so it is not transferable between
        computers.
        """
        state = super().__getstate__()
        # Remove the nonpicklable Model.
        state['_UDPipeLemmatizer__model'] = None
        # models object together with serverfiles store absolute paths to models
        # on computers -- we will init it on when unpickling -- setstate
        state.pop('models')
        return state

    def __setstate__(self, state):
        """
        Called on unpickling the object. It init new models object which was
        deleted from the dictionary in __getstate__.

        Note: __model will be loaded on __call__
        """
        super().__setstate__(state)
        self.models = UDPipeModels()

    def _language_supported(self, iso_language: str) -> bool:
        return self.models.language_supported(iso_language)


class LemmagenLemmatizer(BaseNormalizer):
    name = 'Lemmagen Lemmatizer'

    def __init__(self):
        super().__init__()
        self.lemmatizer = None

    def load_model(self, iso_language: str):
        self.lemmatizer = Lemmatizer(iso_language)

    def normalizer(self, token):
        assert self.lemmatizer is not None
        t = self.lemmatizer.lemmatize(token)
        # sometimes Lemmagen returns an empty string, return original tokens
        # in this case
        return t if t else token

    def _language_supported(self, iso_language: str) -> bool:
        return iso_language in Lemmatizer.list_supported_languages()

    def __getstate__(self):
        """
        This function remove udpipe.Model that cannot be pickled and models that
        include absolute paths on computer -- so it is not transferable between
        computers.
        """
        state = super().__getstate__()
        # lemmagen3 lemmatizer is not picklable
        state['lemmatizer'] = None
        return state
