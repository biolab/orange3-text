from typing import List, Callable

import nltk
import numpy as np
from Orange.util import wrap_callback, dummy_callback

from orangecontrib.text import Corpus
from orangecontrib.text.misc import wait_nltk_data
from orangecontrib.text.preprocess import TokenizedPreprocessor
from orangecontrib.text.util import chunkable


__all__ = ["POSTagger", "AveragedPerceptronTagger", "MaxEntTagger"]


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
