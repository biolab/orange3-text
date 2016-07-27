import unittest

from orangecontrib.text.wikipedia import WikipediaAPI
from orangecontrib.text.corpus import Corpus


class WikipediaTests(unittest.TestCase):
    def test_search(self):
        result = WikipediaAPI.search('en', ['Barack Obama'], ['pageid'])
        self.assertIsInstance(result, Corpus)
        self.assertEquals(len(result.domain.attributes), 1)
        self.assertGreater(len(result), 0)
