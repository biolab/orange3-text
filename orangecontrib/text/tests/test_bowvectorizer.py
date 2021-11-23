import unittest

import numpy as np
from Orange.data import Domain, StringVariable

from orangecontrib.text import preprocess
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.vectorization import BowVectorizer


class BowVectorizationTest(unittest.TestCase):
    def test_transform(self):
        vect = BowVectorizer()
        corpus = Corpus.from_file('deerwester')

        result = vect.transform(corpus)
        self.assertIsInstance(result, Corpus)
        self.assertEqual(len(result.domain.variables), 43)

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

    def test_compute_values_same_tfidf_regardless_num_documents(self):
        """
        When computing TF-IDF from compute values TF-IDF should give same
        results regardless of length of new corpus - IDF weighting should consider
        only counts from original corpus.
        """
        corpus = Corpus.from_file('deerwester')
        train_corpus = corpus[:5]
        test_corpus = corpus[5:]
        vect = BowVectorizer(wglobal=BowVectorizer.IDF)

        bow = vect.transform(train_corpus)
        computed1 = Corpus.from_table(bow.domain, test_corpus[1:])
        computed2 = Corpus.from_table(bow.domain, test_corpus)

        self.assertEqual(computed1.domain, computed2.domain)
        self.assertEqual(bow.domain, computed2.domain)
        self.assertEqual((computed1.X != computed2.X[1:]).nnz, 0)

    # fmt: off
    domain = Domain([], metas=[StringVariable("text")])
    small_corpus_train = Corpus(
        domain,
        np.empty((4, 0)),
        metas=np.array([
            ["this is a nice day day"],
            ["the day is nice"],
            ["i love a beautiful day"],
            ["this apple is mine"]
        ])
    )
    terms = [
        "this", "is", "a", "nice", "day", "the", "i", "love", "beautiful",
        "apple", "mine"
    ]
    train_counts = np.array([
        [1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    ])
    small_corpus_test = Corpus(
        domain,
        np.empty((3, 0)),
        metas=np.array([
            ["this is a nice day day"],
            ["day nice summer mine"],
            ["apple is cool"],
        ])
    )
    test_counts = np.array([
        [1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    ])
    # fmt: on

    def assert_bow_same(self, corpus, values, terms):
        self.assertSetEqual(set(terms), set(a.name for a in corpus.domain.attributes))
        for i, a in enumerate(terms):
            self.assertListEqual(
                corpus.get_column_view(a)[0].tolist(),
                values[:, i].tolist(),
                f"BOW differ for term {a}",
            )

    def test_count_correctness(self):
        """Test if computed counts are correct for train and test dataset"""
        bow = BowVectorizer().transform(self.small_corpus_train)
        self.assert_bow_same(bow, self.train_counts, self.terms)

        # computed from compute_values - result contains only terms from train dataset
        bow_test = Corpus.from_table(bow.domain, self.small_corpus_test)
        self.assert_bow_same(bow_test, self.test_counts, self.terms)

    def test_tfidf_correctness(self):
        """
        Test if computed tf-ids are correct for train and test dataset
        When computing tf-idf on the training dataset (from compute values)
        weights (idf) must be computed based on numbers on training dataset
        """
        bow = BowVectorizer(wglobal=BowVectorizer.IDF).transform(
            self.small_corpus_train
        )

        document_appearance = (self.train_counts != 0).sum(0)
        n = len(self.train_counts)
        idfs_train = self.train_counts * np.log(n / document_appearance)
        self.assert_bow_same(bow, idfs_train, self.terms)

        bow_test = Corpus.from_table(bow.domain, self.small_corpus_test)
        # weights computed based on numbers from training dataset
        idfs_test = self.test_counts * np.log(n / document_appearance)
        self.assert_bow_same(bow_test, idfs_test, self.terms)


if __name__ == "__main__":
    unittest.main()
