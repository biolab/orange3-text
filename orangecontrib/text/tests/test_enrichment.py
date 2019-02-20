import unittest
from Orange.data import Table
from orangecontrib.text.enrichment import enrichment


class EnrichmentTests(unittest.TestCase):
    data = Table('iris')[:10]
    subset = data[:5]

    def test_enrichment(self):
        enrich = enrichment(self.data, self.subset)
        self.assertEqual(len(enrich), 3)

