import unittest
import os
import csv
from orangecontrib.text.scraper import _get_info
from unittest.mock import patch
from contextlib import contextmanager
import tempfile

class MockUrlOpen:
    def __init__(self, filepath):

        self.data = []
        with open(filepath, 'r') as f:
            reader=csv.reader(f,delimiter='\t')
            for row in reader:
                self.data=row
                self.url=row[4]

                try:
                    next(f)
                except StopIteration: # Last empty line is sometimes missing
                    pass

    def __call__(self, url):
        
        @contextmanager
        def cm():
            yield self

        return cm()

filename='article_cache.csv'
filepath=os.path.join(os.path.dirname(__file__), filename)
mock_urllib = MockUrlOpen(filepath)

@patch('urllib.request.urlopen', mock_urllib)
class NewspaperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(delete=False)
        os.remove(self.tmp.name)
        self.cache_path = self.tmp.name

    def test_get_info(self):   #checks whether article, title, date, author, url are same
        scraped_data, is_cached = _get_info(mock_urllib.url)
        self.assertEqual(scraped_data , mock_urllib.data)
        self.assertTrue(is_cached)
