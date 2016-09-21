import datetime
import os
import tempfile
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from Orange.data import TimeVariable
from orangecontrib.text import Corpus
from orangecontrib.text.nyt import NYT


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
        self.assertTrue(self.nyt.api_key_valid())

    def test_nyt_query_keywords(self):
        c = self.nyt.search('slovenia', max_docs=10)
        self.assertIsInstance(c, Corpus)
        self.assertEqual(len(c), 10)

    def test_nyt_query_date_range(self):
        from_date = datetime.date(2013, 1, 1)
        to_date = datetime.date(2014, 1, 1)
        corpus = self.nyt.search('slovenia', from_date, to_date, max_docs=10)
        self.assertEqual(len(corpus), 10)

        time_index = next(i for i, (var, _) in enumerate(NYT.metas) if isinstance(var, TimeVariable))
        tv = corpus.domain.metas[time_index]
        for doc in corpus:
            date = tv.repr_val(doc.metas[time_index])
            date = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S').date()
            self.assertGreaterEqual(date, from_date)
            self.assertLessEqual(date, to_date)

    def test_nyt_query_max_records(self):
        c = self.nyt.search('slovenia', max_docs=25)
        self.assertEqual(len(c), 25)

    def test_nyt_corpus_domain_generation(self):
        corpus = self.nyt.search('slovenia', max_docs=10)
        for var, _ in NYT.attributes:
            self.assertIn(var, corpus.domain.attributes)
        for var, _ in NYT.class_vars:
            self.assertIn(var, corpus.domain.class_vars)
        for var, _ in NYT.metas:
            self.assertIn(var, corpus.domain.metas)

    def test_nyt_result_caching(self):
        self.nyt._fetch_page('slovenia', None, None, 0)     # assure in cache
        _, is_cached = self.nyt._fetch_page('slovenia', None, None, 0)
        self.assertTrue(is_cached)
