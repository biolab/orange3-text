""" This module provides basic functions to process :ref:`Corpus` and extract tokens from documents.

To use preprocessing you should create a corpus::

    >>> from orangecontrib.text import Corpus
    >>> corpus = Corpus.from_file('book-excerpts')

And create an instance of an arbitrary preprocessor:

    >>> from orangecontrib.text import preprocess
    >>> p = preprocess.LowercaseTransformer()
    >>> corpus = p(corpus)
    >>> corpus.tokens[0][:10]
    ['the', 'house', 'jim', 'says', 'he', 'rum', ';', 'and', 'as', 'he']


You can also create a :class:`PreprocessorList` objects with preprocessors you want:

    >>> from orangecontrib.text.preprocess import PreprocessorList
    >>> pp_list = [preprocess.LowercaseTransformer(),
    ...            preprocess.WordPunctTokenizer(),
    ...            preprocess.SnowballStemmer(),
    ...            preprocess.StopwordsFilter(),
    ...            preprocess.FrequencyFilter(min_df=.1)]
    >>> p = PreprocessorList(pp_list)

Then you can apply you preprocessors to the corpus and access tokens via ``tokens`` attribute:

    >>> corpus = Corpus.from_file('book-excerpts')
    >>> corpus = p(corpus)
    >>> corpus.tokens[0][:10]
    ['hous', 'say', ';', 'spoke', 'littl', 'one', 'hand', 'wall', 'hurt', '?']


This module defines ``default_preprocessor`` that will be used to extract tokens from a :ref:`Corpus`
if no preprocessing was applied yet::

    >>> from orangecontrib.text import Corpus
    >>> corpus = Corpus.from_file('deerwester')
    >>> corpus.tokens[0]
    ['human', 'machine', 'interface', 'for', 'lab', 'abc', 'computer', 'applications']

"""
from .preprocess import *
from .tokenize import *
from .filter import *
from .normalize import *
from .transform import *
