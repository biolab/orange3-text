import datetime
import os
import tempfile
import unittest
from contextlib import contextmanager
from unittest.mock import patch, Mock, MagicMock
from http.client import HTTPException
from urllib.error import HTTPError, URLError

from Orange.data import TimeVariable
from orangecontrib.text import Corpus
from orangecontrib.text.nyt import NYT, BATCH_SIZE


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

    def __call__(self, url, **kwargs):
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


class MockHTTPErrors(Mock):
    def __call__(self, url, *args, **kwargs):
        if 'test' in url:               # mock api valid check as usual
            return mock_urllib(url)
        else:                           # raise HTTP errors for _fetch_page
            raise HTTPError(None, 429, None, None, None)


class MockURLErrors(Mock):
    def __call__(self, url, *args, **kwargs):
        if 'test' in url:               # mock api valid check as usual
            return mock_urllib(url)
        else:                           # raise URLError errors for _fetch_page
            raise URLError(None, None)


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
        _, go_sleep = self.nyt._fetch_page('slovenia', None, None, 0)
        self.assertFalse(go_sleep)

    def test_nyt_sleep(self):
        self.nyt.search('slovenia', max_docs=25)
        with patch('time.sleep', Mock()) as sleep:
            self.nyt.search('slovenia', max_docs=25)
            self.assertEqual(sleep.call_count, 0)

    def test_on_progress(self):
        n_calls = 0

        def on_progress(progress, max_docs):
            nonlocal n_calls
            n_calls += 1
            self.assertEqual(max_docs, 25)

        self.nyt.search('slovenia', max_docs=25, on_progress=on_progress)
        self.assertEqual(n_calls, 3)

    def test_break(self):
        def should_break():
            return True

        c = self.nyt.search('slovenia', max_docs=25, should_break=should_break)
        self.assertEqual(len(c), BATCH_SIZE)


@patch('urllib.request.urlopen', MockHTTPErrors())
class NYTTestsErrorRaising(unittest.TestCase):
    API_KEY = 'api_key'

    def test_error_callbacks(self):
        n_calls_rate_limit = 0
        n_calls_error = 0

        def on_rate_limit():
            nonlocal n_calls_rate_limit
            n_calls_rate_limit += 1

        def on_error(e):
            nonlocal n_calls_error
            n_calls_error += 1

        nyt = NYT(self.API_KEY)
        nyt.on_rate_limit = on_rate_limit
        nyt.on_error = on_error

        # both callback, should call rate limit
        nyt.search('slovenia')
        self.assertEqual(n_calls_rate_limit, 1)
        self.assertEqual(n_calls_error, 0)

        # only error callback, should call it
        n_calls_rate_limit = 0
        n_calls_error = 0
        nyt.on_rate_limit = None
        nyt.search('slovenia')
        self.assertEqual(n_calls_rate_limit, 0)
        self.assertEqual(n_calls_error, 1)

        # no callback
        n_calls_rate_limit = 0
        n_calls_error = 0
        nyt.on_error = None
        nyt.on_rate_limit = None
        nyt.search('slovenia')
        self.assertEqual(n_calls_rate_limit, 0)
        self.assertEqual(n_calls_error, 0)

    @patch('orangecontrib.text.nyt.NYT._fetch_page', Mock(return_value=(None, False)))
    def test_error_empty_result(self):
        nyt = NYT(self.API_KEY)
        c = nyt.search('slovenia', max_docs=25)
        self.assertIsNone(c)


@patch('urllib.request.urlopen', MockURLErrors())
class NYTTestsErrorRaising(unittest.TestCase):
    API_KEY = 'api_key'

    def test_url_errors(self):
        nyt = NYT(self.API_KEY)
        nyt.on_no_connection = MagicMock()
        c = nyt.search('slovenia')
        self.assertIsNone(c)
        self.assertEqual(nyt.on_no_connection.call_count, 1)

        nyt.on_no_connection = None
        with self.assertRaises(URLError):
            nyt.search('slovenia')


class NYTTestsApiValidErrorRaising(unittest.TestCase):
    API_KEY = 'api_key'

    def setUp(self):
        self.nyt = NYT(self.API_KEY)

    def test_api_key_valid_errors(self):
        errors = [
            HTTPError(None, 429, None, None, None),
            URLError(''),
            HTTPException(),
        ]

        for e in errors:
            with patch('urllib.request.urlopen', Mock(side_effect=e)):
                self.assertFalse(self.nyt.api_key_valid())


class Test403(unittest.TestCase):
    API_KEY = 'api_key'

    def setUp(self):
        self.nyt = NYT(self.API_KEY)

    @patch('urllib.request.urlopen',
           Mock(side_effect=HTTPError('', 403, None, None, None)))
    def test_nyt_http_error_403(self):
        data, go_sleep = self.nyt._fetch_page('slovenia', None, None, 1)
        self.assertEqual(len(data['response']['docs']), 0)
        self.assertTrue(go_sleep)
