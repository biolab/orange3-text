import unittest
from unittest.mock import MagicMock, call

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

    def test_callback(self):
        vect = SimhashVectorizer(shingle_len=10, f=64)
        callback = MagicMock()
        result = vect.transform(self.corpus, callback=callback)

        self.assertIsInstance(result, Corpus)
        self.assertEqual(len(result), len(self.corpus))
        self.assertEqual(result.X.shape, (len(self.corpus), 64))
        callback.assert_has_calls([call(i / len(self.corpus)) for i in range(9)])


if __name__ == "__main__":
    unittest.main()
