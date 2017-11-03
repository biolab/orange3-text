import nltk
import numpy as np

from orangecontrib.text.util import chunkable
from orangecontrib.text.misc import wait_nltk_data


__all__ = ['POSTagger', 'StanfordPOSTagger', 'AveragedPerceptronTagger', 'MaxEntTagger']


class POSTagger:
    """A class that wraps `nltk.TaggerI` and performs Corpus tagging. """
    def __init__(self, tagger, name='POS Tagger'):
        self.tag_sents = tagger.tag_sents
        self.name = name

    def tag_corpus(self, corpus, **kwargs):
        """ Marks tokens of a corpus with POS tags.

        Args:
            corpus (orangecontrib.text.corpus.Corpus): A corpus instance.

        """
        corpus.pos_tags = np.array(self._tag_sents(corpus.tokens, **kwargs), dtype=object)
        return corpus

    @chunkable
    def _tag_sents(self, documents):
        return list(map(lambda sent: list(map(lambda x: x[1], sent)), self.tag_sents(documents)))

    def __str__(self):
        return self.name


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
            raise ValueError('Either Java SDK not installed or some of the files are invalid.\n' + str(e))
        except LookupError as e:
            raise ValueError(str(e).strip(' =\n'))

    def __str__(self):
        return "{} (model: {})".format(self.name, self._stanford_model)


class AveragedPerceptronTagger(POSTagger):
    name = 'Averaged Perceptron Tagger'

    @wait_nltk_data
    def __init__(self):
        super().__init__(nltk.PerceptronTagger(), self.name)


class MaxEntTagger(POSTagger):
    name = 'Treebank POS Tagger (MaxEnt)'

    @wait_nltk_data
    def __init__(self):
        tagger = nltk.data.load('taggers/maxent_treebank_pos_tagger/english.pickle')
        super().__init__(tagger, self.name)
