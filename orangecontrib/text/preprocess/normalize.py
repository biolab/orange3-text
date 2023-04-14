from typing import List, Callable, Dict, Tuple, Optional
import os
import ufal.udpipe as udpipe
from lemmagen3 import Lemmatizer
import serverfiles
from nltk import stem
from requests.exceptions import ConnectionError

from Orange.misc.environ import data_dir
from Orange.util import wrap_callback, dummy_callback

from orangecontrib.text import Corpus
from orangecontrib.text.language import LANG2ISO, ISO2LANG
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
        corpus = super().__call__(corpus, wrap_callback(callback, end=0.2))
        callback(0.2, "Normalizing...")
        return self._store_tokens(corpus, wrap_callback(callback, start=0.2))

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
    supported_languages = {
        LANG2ISO[l.capitalize()]
        for l in stem.SnowballStemmer.languages
        # skip porter since not language but porter stemmer that we implement separately
        if l != "porter"
    }

    def __init__(self, language='en'):
        super().__init__()
        self.normalizer = stem.SnowballStemmer(ISO2LANG[language].lower()).stem


class UDPipeModels:
    server_url = "https://file.biolab.si/files/udpipe/"

    # some languages differ between udpipe and iso standard
    UDPIPE2LANG = {"Norwegian Bokmaal": "Norwegian Bokmål"}

    def __init__(self):
        self.local_data = os.path.join(data_dir(versioned=False), 'udpipe/')
        self.serverfiles = serverfiles.ServerFiles(self.server_url)
        self.localfiles = serverfiles.LocalFiles(self.local_data,
                                                 serverfiles=self.serverfiles)

    def __getitem__(self, language: str) -> str:
        file_name = self._find_file(language)
        return self.localfiles.localpath_download(file_name)

    @property
    def model_files(self) -> Dict[str, Tuple[str, str]]:
        try:
            files = self.serverfiles.listfiles()
        except ConnectionError:
            files = self.localfiles.listfiles()
        return self.__files_to_dict(files)

    def _find_file(self, language: str) -> str:
        return self.model_files[language][1]

    def __files_to_dict(self, files: List[Tuple[str]]) -> Dict[str, Tuple[str, str]]:
        iso2lang = {}
        for f in files:
            langauge, iso = self.__file_to_language(f[0])
            iso2lang[iso] = (langauge, f[0])
        return iso2lang

    @property
    def supported_languages(self) -> List[Tuple[str, str]]:
        return [(name, iso) for iso, (name, _) in self.model_files.items()]

    @property
    def supported_languages_iso(self) -> List[Tuple[str, str]]:
        return {iso for _, iso in self.supported_languages}

    @property
    def online(self) -> bool:
        try:
            self.serverfiles.listfiles()
            return True
        except ConnectionError:
            return False

    def __file_to_language(self, file: str) -> Tuple[str, str]:
        """
        Transform filenames to langauge strings and iso codes.
        Language name has format "Language (Model)"
        ISO code consist of real iso code which we add the model variation to for
        example "en_lines" for lines english model.
        """
        # language and potential model variation are delimited with -
        name_split = file[: file.find("ud") - 1].split("-")
        # capitalize multi-word languages separated by _
        lg = name_split[0].replace("_", " ").title()
        # fix wrong spelling for Norwegian Bokmål
        lg = self.UDPIPE2LANG.get(lg, lg)

        if len(name_split) > 1:
            # languages with multiple models have model name as second item in split
            return f"{lg} ({name_split[1]})", self.__lang2iso(lg, name_split[1])
        return lg, self.__lang2iso(lg, None)

    @staticmethod
    def __lang2iso(language: str, model: Optional[str]) -> str:
        language = [LANG2ISO[language]]
        if model:
            language.append(model)
        return "_".join(language)

    def language_to_iso(self, language: str) -> str:
        """This method is used to migrate from old widget's language settings to ISO"""
        # UDPIPE language changes when migrating from language words to ISO
        # previously the second word of two-word languages started with lowercase
        # also different models for same language were written just with space between
        # the language and model name, now we use parenthesis
        migration = {
            "Ancient greek proiel": "Ancient Greek (proiel)",
            "Ancient greek": "Ancient Greek",
            "Czech cac": "Czech (cac)",
            "Czech cltt": "Czech (cltt)",
            "Dutch lassysmall": "Dutch (lassysmall)",
            "English lines": "English (lines)",
            "English partut": "English (partut)",
            "Finnish ftb": "Finnish (ftb)",
            "French partut": "French (partut)",
            "French sequoia": "French (sequoia)",
            "Galician treegal": "Galician (treegal)",
            "Latin ittb": "Latin (ittb)",
            "Latin proiel": "Latin (proiel)",
            "Norwegian bokmaal": "Norwegian Bokmål",
            "Norwegian nynorsk": "Norwegian Nynorsk",
            "Old church slavonic": "Old Church Slavonic",
            "Portuguese br": "Portuguese (br)",
            "Russian syntagrus": "Russian (syntagrus)",
            "Slovenian sst": "Slovenian (sst)",
            "Spanish ancora": "Spanish (ancora)",
            "Swedish lines": "Swedish (lines)",
        }
        return dict(self.supported_languages).get(migration.get(language, language))


class UDPipeStopIteration(StopIteration):
    pass


class UDPipeLemmatizer(BaseNormalizer):
    name = 'UDPipe Lemmatizer'

    def __init__(self, language="en", use_tokenizer=False):
        super().__init__()
        self.__language = language
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
        try:
            self.__model = udpipe.Model.load(self.models[self.__language])
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


class LemmagenLemmatizer(BaseNormalizer):
    name = 'Lemmagen Lemmatizer'
    supported_languages = set(Lemmatizer.list_supported_languages())

    def __init__(self, language="en"):
        super().__init__()
        self.language = language  # used only for unpicking
        self.lemmatizer = Lemmatizer(language)

    def normalizer(self, token):
        assert self.lemmatizer is not None
        t = self.lemmatizer.lemmatize(token)
        # sometimes Lemmagen returns an empty string, return original tokens
        # in this case
        return t if t else token

    def __getstate__(self):
        """Remove model that cannot be pickled"""
        state = super().__getstate__()
        state["lemmatizer"] = None
        return state

    def __setstate__(self, state):
        """Reinstate the model when upickled"""
        super().__setstate__(state)
        self.lemmatizer = Lemmatizer(self.language)
