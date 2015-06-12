import os
import unittest
from orangecontrib.text.corpus import Corpus


DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'datasets')


class CorpusTests(unittest.TestCase):
    def test_corpus_from_file(self):
        c = Corpus.from_file(os.path.join(DATASET_PATH, 'bookexcerpts.txt'))
