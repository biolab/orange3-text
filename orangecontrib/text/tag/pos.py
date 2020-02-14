from typing import List, Callable

import nltk
import numpy as np

from Orange.util import wrap_callback, dummy_callback

from orangecontrib.text import Corpus
from orangecontrib.text.misc import wait_nltk_data
from orangecontrib.text.preprocess import TokenizedPreprocessor
from orangecontrib.text.util import chunkable


__all__ = ['POSTagger', 'StanfordPOSTagger', 'AveragedPerceptronTagger', 'MaxEntTagger']


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


class StanfordPOSTaggerError(Exception):
    pass


class StanfordPOSTagger(nltk.StanfordPOSTagger, POSTagger):
    name = 'Stanford POS Tagger'

    @classmethod
    def check(cls, path_to_model, path_to_jar):
        """ Checks whether provided `path_to_model` and `path_to_jar` are valid.

        Raises:
            ValueError: in case at least one of the paths is invalid.

        Notes:
            Can raise an exception if Java Development Kit is not installed or not properly configured.

        Examples:
            >>> try:
            ...    StanfordPOSTagger.check('path/to/model', 'path/to/stanford.jar')
            ... except ValueError as e:
            ...    print(e)
            Could not find stanford-postagger.jar jar file at path/to/stanford.jar

        """
        try:
            cls(path_to_model, path_to_jar).tag(())
        except OSError as e:
            raise StanfordPOSTaggerError(
                'Either Java SDK not installed or some of the '
                'files are invalid.\n' + str(e))
        except LookupError as e:
            raise StanfordPOSTaggerError(str(e).strip(' =\n'))


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
