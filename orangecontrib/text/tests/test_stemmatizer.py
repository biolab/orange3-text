import unittest
from orangecontrib.text.preprocess import Stemmatizer
from nltk import PorterStemmer


class StemmatizerTests(unittest.TestCase):

    def setUp(self):
        self.stemmer = PorterStemmer().stem

    def test_init_without_stemmer(self):
        with self.assertRaises(ValueError):
            Stemmatizer('not a function')

    def test_str(self):
        stemmatizer = Stemmatizer(self.stemmer, name='porter')
        self.assertIn('porter', str(stemmatizer))

    def test_call(self):
        stemmatizer = Stemmatizer(self.stemmer, name='porter')

        word = 'Testing'
        self.assertEqual(stemmatizer(word), self.stemmer(word))

        tokens = [
            'Testing',
            'tokenized',
            'Sentence'
        ]
        self.assertEqual(
                stemmatizer(tokens),
                [self.stemmer(token) for token in tokens]
        )

    def test_call_with_bad_input(self):
        stemmatizer = Stemmatizer(self.stemmer, name='porter')

        with self.assertRaises(ValueError):
            stemmatizer(10)
