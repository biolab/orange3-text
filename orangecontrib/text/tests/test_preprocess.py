import os
import unittest
from orangecontrib.text.preprocess import Preprocessor, Stemmer, Lemmatizer


DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'datasets')


class PreprocessTests(unittest.TestCase):
    documents = ["Human machine interface for lab abc computer applications.",
                 "A survey of user opinion of computer system response time.",
                 "The EPS user interface management system",
                 "System and human system engineering testing of EPS",
                 "Relation of user perceived response time to error measurement",
                 "The generation of random binary unordered trees",
                 "The intersection graph of paths in trees",
                 "Graph minors IV Widths of trees and well quasi ordering",
                 "Graph minors A survey",
                 ]
    # TODO test preprocessing and combinations of trans and other settings since trans overrides some of them

    def test_preprocess(self):
        p = Preprocessor(incl_punct=False, lowercase=True, stop_words=None, trans=Lemmatizer)
        corpus = p(self.documents)
        for l in corpus:
            print(l)
        self.assertEqual(len(corpus), 9)
