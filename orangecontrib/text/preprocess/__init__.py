""" This module provides basic functions to process :ref:`Corpus` and extract tokens from documents.

To use preprocessing you should create a corpus::

    >>> from orangecontrib.text import Corpus
    >>> corpus = Corpus.from_file('bookexcerpts')

And create a :class:`Preprocessor` objects with methods you want:

    >>> from orangecontrib.text import preprocess
    >>> p = preprocess.Preprocessor(transformers=[preprocess.LowercaseTransformer()],
    ...                             tokenizer=preprocess.WordPunctTokenizer(),
    ...                             normalizer=preprocess.SnowballStemmer('english'),
    ...                             filters=[preprocess.StopwordsFilter('english'),
    ...                                      preprocess.FrequencyFilter(min_df=.1)])

Then you can apply you preprocessor to the corpus and access tokens via ``tokens`` attribute:

    >>> new_corpus = p(corpus)
    >>> new_corpus.tokens[0][:10]
    ['hous', 'say', ';', 'spoke', 'littl', 'one', 'hand', 'wall', 'hurt', '?']


This module defines ``default_preprocessor`` that will be used to extract tokens from a :ref:`Corpus`
if no preprocessing was applied yet::

    >>> from orangecontrib.text import Corpus
    >>> corpus = Corpus.from_file('deerwester')
    >>> corpus.tokens[0]
    ['human', 'machine', 'interface', 'for', 'lab', 'abc', 'computer', 'applications']

"""
import nltk
from orangecontrib.text.misc import nltk_data_dir

# Ensure NLTK data is present
NLTK_DATA = ["wordnet", "stopwords", "punkt", "opinion_lexicon", 'averaged_perceptron_tagger', 'maxent_treebank_pos_tagger']
nltk.download(NLTK_DATA, download_dir=nltk_data_dir(), quiet=False)

from .filter import *
from .normalize import *
from .tokenize import *
from .transform import *
from .preprocess import *

base_preprocessor = Preprocessor(transformers=[LowercaseTransformer()],
                                 tokenizer=WordPunctTokenizer())
