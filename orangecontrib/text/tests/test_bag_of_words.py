import unittest

from orangecontrib.text.bagofowords import BagOfWords
from orangecontrib.text.corpus import Corpus


class BagOfWordsTests(unittest.TestCase):

    def progress_callback(self):
        self.progress_callbacks += 1
        return

    def error_callback(self, *args):
        self.error_callbacks += 1
        return

    def setUp(self):
        self.progress_callbacks = 0
        self.error_callbacks = 0

        self.bow = BagOfWords(
                progress_callback=self.progress_callback,
                error_callback=self.error_callback,
        )

    def test_create_bow(self):
        corpus = Corpus.from_file('deerwester')
        bag_of_words = self.bow(corpus, use_tfidf=True)

        self.assertIsNotNone(bag_of_words.X)
        self.assertEqual(9, bag_of_words.X.shape[0])
        self.assertEqual(42, bag_of_words.X.shape[1])
        self.assertEqual(self.progress_callbacks, 4)
        self.assertEqual(self.error_callbacks, 0)

    def test_empty_tokens(self):
        corpus = Corpus.from_file('deerwester')
        corpus.text_features = []
        bag_of_words = self.bow(corpus)

        self.assertIs(corpus, bag_of_words)

    def test_bow_exceptions(self):
        self.assertRaises(
            ValueError,
            self.bow,
            None
        )
