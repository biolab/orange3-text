import unittest
from unittest import mock

import wikipedia

from orangecontrib.text.wikipedia import WikipediaAPI
from orangecontrib.text.corpus import Corpus


class StopingMock(mock.Mock):
    def __init__(self, allow_calls=0):
        super().__init__()
        self.allow_calls = allow_calls
        self.call_count = 0

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        if self.call_count >= self.allow_calls:
            return True
        else:
            return False


class WikipediaTests(unittest.TestCase):
    def test_search(self):
        on_progress = mock.MagicMock()

        api = WikipediaAPI()

        result = api.search('en', ['Clinton'], articles_per_query=2, on_progress=on_progress)
        self.assertIsInstance(result, Corpus)
        self.assertEquals(len(result.domain.attributes), 0)
        self.assertEquals(len(result.domain.metas), 7)
        self.assertEquals(len(result), 2)

        self.assertEquals(on_progress.call_count, 2)
        progress = 0
        for arg in on_progress.call_args_list:
            self.assertGreater(arg[0][0], progress)
            progress = arg[0][0]

    def test_search_disambiguation(self):
        api = WikipediaAPI()
        result = api.search('en', ['Scarf'], articles_per_query=3)

        self.assertIsInstance(result, Corpus)
        self.assertGreater(len(result), 3)

    def test_search_break(self):
        api = WikipediaAPI()

        # stop immediately
        result = api.search('en', ['Clinton'], articles_per_query=2, should_break=mock.Mock(return_value=True))
        self.assertEqual(len(result), 0)

        # stop inside recursion
        result = api.search('en', ['Scarf'], articles_per_query=3, should_break=StopingMock(4))
        self.assertEqual(len(result), 2)

    def page(*args, **kwargs):
        raise wikipedia.exceptions.PageError('1')

    @mock.patch('wikipedia.page', page)
    def test_page_error(self):
        on_error = mock.MagicMock()
        api = WikipediaAPI(on_error=on_error)
        api.search('en', ['Barack Obama'])
        self.assertEqual(on_error.call_count, 0)

    def search(*args, **kwargs):
        raise IOError('Network error')

    @mock.patch('wikipedia.search', search)
    def test_network_errors(self):
        on_error = mock.MagicMock()
        api = WikipediaAPI(on_error=on_error)
        api.search('en', ['Barack Obama'])
        self.assertEqual(on_error.call_count, 1)
