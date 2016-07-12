import unittest

from orangecontrib.text.corpus import Corpus
from orangecontrib.text.vectorization import SimhashVectorizer


class TestSimhash(unittest.TestCase):
    def setUp(self):
        self.corpus = Corpus.from_file('deerwester')

    def test_transform(self):
        vect = SimhashVectorizer(shingle_len=10, f=64)
        result = vect.transform(self.corpus)
        self.assertIsInstance(result, Corpus)
        self.assertEqual(len(result), len(self.corpus))
        self.assertEqual(result.X.shape, (len(self.corpus), 64))

    def test_report(self):
        vect = SimhashVectorizer()
        self.assertGreater(len(vect.report()), 0)
