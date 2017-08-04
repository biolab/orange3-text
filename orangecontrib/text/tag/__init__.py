"""

A module for tagging :ref:`Corpus` instances.

This module provides a default `pos_tagger` that can be used for POSTagging an English corpus::

    >>> from orangecontrib.text.corpus import Corpus
    >>> from orangecontrib.text.tag import AveragedPerceptronTagger
    >>> corpus = Corpus.from_file('deerwester.tab')
    >>> tagger = AveragedPerceptronTagger()
    >>> tagged_corpus = tagger.tag_corpus(corpus)
    >>> tagged_corpus.pos_tags[0]  # you can use `pos_tags` attribute to access tags directly
    ['JJ', 'NN', 'NN', 'IN', 'NN', 'NN', 'NN', 'NNS']
    >>> next(tagged_corpus.ngrams_iterator(include_postags=True))  # or `ngrams_iterator` to iterate over documents
    ['human_JJ', 'machine_NN', 'interface_NN', 'for_IN', 'lab_NN', 'abc_NN', 'computer_NN', 'applications_NNS']

"""

from .pos import *
