import unittest
from unittest import mock

import wikipedia

from orangecontrib.text.wikipedia import WikipediaAPI
from orangecontrib.text.corpus import Corpus


class WikipediaTests(unittest.TestCase):
    def test_search(self):
        on_finish = mock.MagicMock()
        on_progress = mock.MagicMock()

        api = WikipediaAPI(on_finish=on_finish, on_progress=on_progress)

        result = api.search('en', ['Clinton'], ['pageid'])
        self.assertIsInstance(result, Corpus)
        self.assertEquals(len(result.domain.attributes), 1)
        self.assertGreater(len(result), 0)

        self.assertEqual(on_finish.call_count, 1)
        self.assertEqual(on_finish.call_args[0][0], result)

        self.assertGreater(on_progress.call_count, 5)
        progress = 0
        for arg in on_progress.call_args_list:
            self.assertGreater(arg[0][0], progress)
            progress = arg[0][0]

    def test_disconnect(self):
        on_finish = mock.MagicMock()
        api = WikipediaAPI(on_finish=on_finish)

        api.search('en', ['Barack Obama'], attributes=['pageid'], async=True)
        api.disconnect()
        self.assertEqual(on_finish.call_count, 1)
        self.assertIsNone(on_finish.call_args[0][0])

    def test_several_threads(self):
        api = WikipediaAPI()
        api.search('en', ['Barack Obama'], attributes=['pageid'], async=True)
        with self.assertRaises(RuntimeError):
            api.search('en', ['Barack Obama'], attributes=['pageid'], async=True)

    def page(*args, **kwargs):
        raise wikipedia.exceptions.PageError('1')

    @mock.patch('wikipedia.page', page)
    def test_page_error(self):
        on_error = mock.MagicMock()
        api = WikipediaAPI(on_error=on_error)
        api.search('en', ['Barack Obama'], attributes=['pageid'])
        self.assertEqual(on_error.call_count, 0)

    def search(*args, **kwargs):
        raise IOError('Network error')

    @mock.patch('wikipedia.search', search)
    def test_network_errors(self):
        on_error = mock.MagicMock()
        api = WikipediaAPI(on_error=on_error)
        api.search('en', ['Barack Obama'], attributes=['pageid'])
        self.assertEqual(on_error.call_count, 1)