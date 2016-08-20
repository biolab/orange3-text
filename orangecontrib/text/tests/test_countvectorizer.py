import unittest

from orangecontrib.text.vectorization import CountVectorizer
from orangecontrib.text.corpus import Corpus
from orangecontrib.text import preprocess


class CountVectorizerTests(unittest.TestCase):
    def test_ngrams(self):
        vect = CountVectorizer()
        corpus = Corpus.from_file('deerwester')
        pr = preprocess.Preprocessor(tokenizer=preprocess.RegexpTokenizer('\w+'),
                                     ngrams_range=(1, 3))
        pr(corpus, inplace=True)
        result = vect.transform(corpus)
        attrs = [attr.name for attr in result.domain.attributes]
        self.assertIn(corpus.tokens[0][1], attrs)
        self.assertIn(' '.join(corpus.tokens[0][:2]), attrs)
        self.assertIn(' '.join(corpus.tokens[0][:3]), attrs)

    def test_domain(self):
        vect = CountVectorizer()
        corpus = Corpus.from_file('deerwester')

        result = vect.transform(corpus)
        attrs = [attr.name for attr in result.domain.attributes]
        self.assertEqual(attrs, sorted(attrs))

        X = result.X.toarray()
        for i in range(len(corpus)):
            for contains, attr in zip(X[i], attrs):
                if contains > .001:
                    self.assertIn(attr, corpus.tokens[i])

    def test_transform(self):
        vect = CountVectorizer()
        corpus = Corpus.from_file('deerwester')
        result = vect.transform(corpus)
        self.assertIsInstance(result, Corpus)
        self.assertEqual(len(result.domain), 43)

    def test_binary(self):
        vect = CountVectorizer(binary=True)
        corpus = Corpus.from_file('deerwester')
        result = vect.transform(corpus)
        self.assertEqual(result.X.max(), 1.)

    def test_report(self):
        vect = CountVectorizer()
        self.assertGreater(len(vect.report()), 0)

    def test_empty_tokens(self):
        corpus = Corpus.from_file('deerwester')
        corpus.text_features = []
        bag_of_words = CountVectorizer().transform(corpus, copy=False)

        self.assertIs(corpus, bag_of_words)
