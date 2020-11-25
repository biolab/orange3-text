import unittest

from Orange.data import Table

from orangecontrib.text import Corpus
try:
    from orangecontrib.text.corpus_to_network import CorpusToNetwork
    from orangecontrib.network import Network
    SKIP = False
except Exception:
    SKIP = True


@unittest.skipIf(SKIP, "Network add-on is not installed.")
class CorpusToNetworkTest(unittest.TestCase):
    def test_init(self):
        corpus = Corpus.from_file('deerwester')
        c2n = CorpusToNetwork(corpus)
        self.assertIsInstance(c2n.corpus, Corpus)
        self.assertEqual(len(c2n.corpus), 9)
        with self.assertRaises(Exception):
            c2n = CorpusToNetwork(corpus.domain)

    def test_call_document(self):
        corpus = Corpus.from_file('deerwester')
        c2n = CorpusToNetwork(corpus)
        result = c2n(document_nodes=True,
                     window_size=1,
                     threshold=1,
                     freq_threshold=1)
        items = c2n.get_current_items(True)
        self.assertIsInstance(result, Network)
        self.assertIsInstance(items, Table)
        self.assertEqual(len(items), result.number_of_nodes())
        self.assertEqual(result.number_of_nodes(), len(corpus))

    def test_call_word(self):
        corpus = Corpus.from_file('deerwester')
        c2n = CorpusToNetwork(corpus)
        result = c2n(document_nodes=False,
                     window_size=1,
                     threshold=1,
                     freq_threshold=1)
        items = c2n.get_current_items(False)
        self.assertIsInstance(result, Network)
        self.assertIsInstance(items, Table)
        self.assertEqual(len(items), result.number_of_nodes())
        self.assertGreater(result.number_of_nodes(), len(corpus))

    def test_params(self):
        corpus = Corpus.from_file('deerwester')
        c2n = CorpusToNetwork(corpus)
        result1 = c2n(document_nodes=False,
                      window_size=1,
                      threshold=1,
                      freq_threshold=1)
        result2 = c2n(document_nodes=False,
                      window_size=1,
                      threshold=100,
                      freq_threshold=1)
        self.assertGreater(result1.number_of_edges(),
                           result2.number_of_edges())
        result2 = c2n(document_nodes=False,
                      window_size=10,
                      threshold=1,
                      freq_threshold=1)
        self.assertLess(result1.number_of_edges(),
                        result2.number_of_edges())
        result2 = c2n(document_nodes=False,
                      window_size=1,
                      threshold=1,
                      freq_threshold=100)
        self.assertGreater(result1.number_of_nodes(),
                           result2.number_of_nodes())

    def test_cache(self):
        corpus = Corpus.from_file('deerwester')
        c2n = CorpusToNetwork(corpus)
        result1 = c2n(document_nodes=True,
                      window_size=1,
                      threshold=1,
                      freq_threshold=1)
        result2 = c2n(document_nodes=True,
                      window_size=1,
                      threshold=1,
                      freq_threshold=1)
        self.assertIs(result1, result2)

    def test_empty(self):
        corpus = Corpus.from_file('deerwester')[:0]
        c2n = CorpusToNetwork(corpus)
        result = c2n(document_nodes=True,
                     window_size=1,
                     threshold=1,
                     freq_threshold=1)
        self.assertEqual(result.number_of_nodes(), 0)
        self.assertEqual(result.number_of_edges(), 0)


if __name__ == '__main__':
    unittest.main()
