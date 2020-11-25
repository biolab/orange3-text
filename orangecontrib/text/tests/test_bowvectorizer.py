import unittest

import numpy as np

from orangecontrib.text import preprocess
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.vectorization import BowVectorizer


class BowVectorizationTest(unittest.TestCase):
    def test_transform(self):
        vect = BowVectorizer()
        corpus = Corpus.from_file('deerwester')

        result = vect.transform(corpus)
        self.assertIsInstance(result, Corpus)
        self.assertEqual(len(result.domain), 43)

    def test_binary(self):
        vect = BowVectorizer(wlocal=BowVectorizer.BINARY)
        corpus = Corpus.from_file('deerwester')
        result = vect.transform(corpus)
        self.assertEqual(result.X.max(), 1.)

    def test_empty_tokens(self):
        corpus = Corpus.from_file('deerwester')
        corpus.text_features = []
        bag_of_words = BowVectorizer().transform(corpus, copy=False)

        self.assertIs(corpus, bag_of_words)

    def test_domain(self):
        vect = BowVectorizer()
        corpus = Corpus.from_file('deerwester')

        result = vect.transform(corpus)
        attrs = [attr.name for attr in result.domain.attributes]
        self.assertEqual(attrs, sorted(attrs))

        X = result.X.toarray()
        for i in range(len(corpus)):
            for contains, attr in zip(X[i], attrs):
                if contains > .001:
                    self.assertIn(attr, corpus.tokens[i])

    def test_ngrams(self):
        vect = BowVectorizer()
        corpus = Corpus.from_file('deerwester')
        corpus = preprocess.RegexpTokenizer('\w+')(corpus)
        corpus = preprocess.NGrams(ngrams_range=(1, 3))(corpus)
        result = vect.transform(corpus)
        attrs = [attr.name for attr in result.domain.attributes]
        self.assertIn(corpus.tokens[0][1], attrs)
        self.assertIn(' '.join(corpus.tokens[0][:2]), attrs)
        self.assertIn(' '.join(corpus.tokens[0][:3]), attrs)

    def test_report(self):
        vect = BowVectorizer()
        self.assertGreater(len(vect.report()), 0)

    def test_args(self):
        corpus = Corpus.from_file('deerwester')

        BowVectorizer.wglobals['const'] = lambda df, N: 1

        vect = BowVectorizer(norm=BowVectorizer.NONE,
                             wlocal=BowVectorizer.COUNT,
                             wglobal='const')

        self.assertEqualCorpus(vect.transform(corpus),
                               BowVectorizer(wlocal=BowVectorizer.COUNT).transform(corpus))

        vect = BowVectorizer(norm=BowVectorizer.NONE,
                             wlocal=BowVectorizer.BINARY,
                             wglobal='const')
        self.assertEqualCorpus(vect.transform(corpus),
                               BowVectorizer(wlocal=BowVectorizer.BINARY).transform(corpus))

        vect = BowVectorizer(norm=BowVectorizer.L1,
                             wlocal=BowVectorizer.COUNT,
                             wglobal='const')
        x = vect.transform(corpus).X
        self.assertAlmostEqual(abs(x.sum(axis=1) - 1).sum(), 0)

    def test_compute_values(self):
        corpus = Corpus.from_file('deerwester')
        vect = BowVectorizer()

        bow = vect.transform(corpus)
        computed = Corpus.from_table(bow.domain, corpus)

        self.assertEqual(bow.domain, computed.domain)
        self.assertEqual((bow.X != computed.X).nnz, 0)

    def test_compute_values_to_different_domain(self):
        source = Corpus.from_file('deerwester')
        destination = Corpus.from_file('book-excerpts')

        self.assertFalse(source.domain.attributes)
        self.assertFalse(destination.domain.attributes)

        bow = BowVectorizer().transform(source)
        computed = destination.transform(bow.domain)

        self.assertEqual(bow.domain.attributes, computed.domain.attributes)

    def assertEqualCorpus(self, first, second, msg=None):
        np.testing.assert_allclose(first.X.todense(), second.X.todense(), err_msg=msg)

    def test_empty_corpus(self):
        """
        Empty data.
        GH-247
        """
        corpus = Corpus.from_file("deerwester")[:0]
        vect = BowVectorizer(norm=BowVectorizer.L1)
        out = vect.transform(corpus)
        self.assertEqual(out, corpus)

    def tests_duplicated_names(self):
        """
        BOW adds words to the domain and if same attribute name already appear
        in the domain it renames it and add number to the existing attribute
        name
        """
        corpus = Corpus.from_file("deerwester")
        corpus = corpus.extend_attributes(np.ones((len(corpus), 1)), ["human"])
        corpus = corpus.extend_attributes(np.ones((len(corpus), 1)), ["testtest"])
        vect = BowVectorizer()
        out = vect.transform(corpus)
        # first attribute is in the dataset before bow and should be renamed
        self.assertEqual("human (1)", out.domain[0].name)
        self.assertEqual("testtest", out.domain[1].name)
        # all attributes from [1:] are are bow attributes and should include
        # human
        self.assertIn("human", [v.name for v in out.domain.attributes[1:]])


if __name__ == "__main__":
    unittest.main()
