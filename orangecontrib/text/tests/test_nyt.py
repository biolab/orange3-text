import os
import tempfile
import unittest
import datetime
from unittest.mock import patch
from contextlib import contextmanager
from orangecontrib.text.nyt import NYT, NYT_TEXT_FIELDS

from Orange.data import StringVariable


class MockUrlOpen:
    def __init__(self, cache):
        self.data = {}
        with open(cache, 'r') as f:
            for url in f:
                data = next(f)
                self.data[url.strip()] = data.strip().encode('utf-8')
                try:
                    next(f)
                except StopIteration:
                    # Last empty line is sometimes missing
                    pass

    def __call__(self, url):
        self.current_url = url
        self.response_code = 200 if url in self.data else 404

        @contextmanager
        def cm():
            yield self

        return cm()

    def read(self):
        return self.data[self.current_url]

    def getcode(self):
        return self.response_code


CACHE = os.path.join(os.path.dirname(__file__), 'nyt-cache.txt')
mock_urllib = MockUrlOpen(CACHE)


@patch('urllib.request.urlopen', mock_urllib)
class NYTTests(unittest.TestCase):
    API_KEY = 'api_key'

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(delete=False)
        os.remove(self.tmp.name)
        self.nyt = NYT(self.API_KEY)
        self.nyt.cache_path = self.tmp.name

    def tearDown(self):
        if os.path.exists(self.tmp.name):
            os.remove(self.tmp.name)

    def test_nyt_key(self):
        self.assertTrue(self.nyt.check_api_key())

    def test_nyt_query_keywords(self):
        records = self.nyt.run_query('slovenia')
        self.assertEqual(len(records), 10)

    def test_nyt_query_date_range(self):
        corpus = self.nyt.run_query('slovenia', datetime.date(2013, 1, 1), datetime.date(2014, 1, 1))
        self.assertEqual(len(corpus.documents), 10)

    def test_nyt_query_max_records(self):
        records = self.nyt.run_query('slovenia', max_records=30)
        self.assertEqual(len(records), 30)

    def test_nyt_corpus_domain_generation(self):
        corpus = self.nyt.run_query('slovenia')
        meta_vars = [StringVariable.make(field) for field in NYT_TEXT_FIELDS] + \
                    [StringVariable.make('pub_date'), StringVariable.make('country')]

        self.assertEqual(len(meta_vars), len(corpus.domain.metas))
        self.assertEqual(len(corpus), 10)

    def test_nyt_result_caching(self):
        # Run a query to create a cache entry first.
        self.nyt.run_query('slovenia')
        data, is_cached, error = self.nyt._execute_query(0)
        # Check if the response was cached.
        self.assertTrue(is_cached)
