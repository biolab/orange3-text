from typing import List, Callable, Tuple

import nltk
import spacy
from spacy.cli import info, download
from spacy.tokens import Doc
import numpy as np
from Orange.util import wrap_callback, dummy_callback

from orangecontrib.text import Corpus
from orangecontrib.text.misc import wait_nltk_data
from orangecontrib.text.preprocess import TokenizedPreprocessor
from orangecontrib.text.util import chunkable


__all__ = ["POSTagger", "AveragedPerceptronTagger", "MaxEntTagger",
           "SpacyPOSTagger"]


SPACY_MODELS = {
    "ca": {"language": "Catalan", "package": "ca_core_news_sm", "dependency": "None"},
    "zh": {"language": "Chinese", "package": "zh_core_web_sm", "dependency": "Jieba"},
    "hr": {"language": "Croatian", "package": "hr_core_news_sm", "dependency": "None"},
    "da": {"language": "Danish", "package": "da_core_news_sm", "dependency": "None"},
    "nl": {"language": "Dutch", "package": "nl_core_news_sm", "dependency": "None"},
    "en": {"language": "English", "package": "en_core_web_sm", "dependency": "None"},
    "fi": {"language": "Finnish", "package": "fi_core_news_sm", "dependency": "None"},
    "fr": {"language": "French", "package": "fr_core_news_sm", "dependency": "None"},
    "de": {"language": "German", "package": "de_core_news_sm", "dependency": "None"},
    "el": {"language": "Greek", "package": "el_core_news_sm", "dependency": "None"},
    "it": {"language": "Italian", "package": "it_core_news_sm", "dependency": "None"},
    "ja": {"language": "Japanese", "package": "ja_core_news_sm", "dependency": "SudachiPy"},
    "ko": {"language": "Korean", "package": "ko_core_news_sm", "dependency": "None"},
    "lt": {"language": "Lithuanian", "package": "lt_core_news_sm", "dependency": "None"},
    "mk": {"language": "Macedonian", "package": "mk_core_news_sm", "dependency": "None"},
    "xx": {"language": "Multi-language", "package": "xx_ent_wiki_sm", "dependency": "None"},
    "nb": {"language": "Norwegian Bokmål", "package": "nb_core_news_sm", "dependency": "None"},
    "pl": {"language": "Polish", "package": "pl_core_news_sm", "dependency": "None"},
    "pt": {"language": "Portuguese", "package": "pt_core_news_sm", "dependency": "None"},
    "ro": {"language": "Romanian", "package": "ro_core_news_sm", "dependency": "None"},
    "ru": {"language": "Russian", "package": "ru_core_news_sm", "dependency": "pymorphy3"},
    "sl": {"language": "Slovenian", "package": "sl_core_news_sm", "dependency": "None"},
    "es": {"language": "Spanish", "package": "es_core_news_sm", "dependency": "None"},
    "sv": {"language": "Swedish", "package": "sv_core_news_sm", "dependency": "None"},
    "uk": {"language": "Ukrainian", "package": "uk_core_news_sm", "dependency":
        "pymorphy3, pymorphy3-dicts-uk"}
}


class POSTagger(TokenizedPreprocessor):
    """A class that wraps `nltk.TaggerI` and performs Corpus tagging. """
    def __init__(self, tagger):
        self.tagger = tagger.tag_sents

    def __call__(self, corpus: Corpus, callback: Callable = None,
                 **kw) -> Corpus:
        """ Marks tokens of a corpus with POS tags. """
        if callback is None:
            callback = dummy_callback
        corpus = super().__call__(corpus, wrap_callback(callback, end=0.2))

        assert corpus.has_tokens()
        callback(0.2, "POS Tagging...")
        tags = np.array(self._preprocess(corpus.tokens, **kw), dtype=object)
        corpus.pos_tags = tags
        return corpus

    @chunkable
    def _preprocess(self, tokens: List[List[str]]) -> List[List[str]]:
        return list(map(lambda sent: list(map(lambda x: x[1], sent)),
                        self.tagger(tokens)))


class AveragedPerceptronTagger(POSTagger):
    name = 'Averaged Perceptron Tagger'

    @wait_nltk_data
    def __init__(self):
        super().__init__(nltk.PerceptronTagger())


class MaxEntTagger(POSTagger):
    name = 'Treebank POS Tagger (MaxEnt)'

    @wait_nltk_data
    def __init__(self):
        tagger = nltk.data.load('taggers/maxent_treebank_pos_tagger/english.pickle')
        super().__init__(tagger)


def find_model(language: str) -> str:
    return SPACY_MODELS[language]["package"]


class SpacyModels:
    installed_models_info = info()

    def __init__(self):
        self.installed_models = self.installed_models_info['pipelines']

    def __getitem__(self, language: str) -> str:
        model = find_model(language)
        if model not in self.installed_models:
            download(model)
        return model


class SpacyPOSTagger(TokenizedPreprocessor):
    name = 'Spacy POS Tagger'
    supported_languages = set(SPACY_MODELS.keys())

    def __init__(self, language: str = "en"):
        self.__language = language
        self.models = SpacyModels()
        self.__model = None

    def __call__(self, corpus: Corpus, callback: Callable = None,
                 **kw) -> Corpus:
        """ Marks tokens of a corpus with POS tags. """
        if callback is None:
            callback = dummy_callback
        corpus = super().__call__(corpus, wrap_callback(callback, end=0.2))

        assert corpus.has_tokens()
        callback(0.2, "POS Tagging...")
        self.__model = spacy.load(self.models[self.__language])
        tags = np.array(self.tag(corpus.tokens), dtype=object)
        corpus.pos_tags = tags
        return corpus

    def tag(self, tokens):
        out_tokens = []
        for token_list in tokens:
            # required for Spacy to work with pre-tokenized texts
            doc = Doc(self.__model.vocab, words=token_list)
            out_tokens.append([token.pos_ for token in self.__model(doc)])
        return out_tokens
