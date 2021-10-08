import unittest
from unittest.mock import patch

import numpy as np
from Orange.data import Domain, StringVariable
from Orange.widgets.tests.base import WidgetTest
from orangecontrib.text import twitter, Corpus
from orangecontrib.text.tests.test_twitter import Response
from orangecontrib.text.widgets.owtwitter import OWTwitter
from tweepy import TweepyException, TooManyRequests


# it is not possible to test real API because API key cannot be shared for
# tests so we use dummy_fetch instead
def dummy_fetch(self, cursors, max_tweets, search_author, callback):
    return (
        Corpus(
            Domain([], metas=[StringVariable("Content")]),
            metas=np.array([["Abc"], ["Cde"], ["Gf"]]),
        ),
        3,
    )


class TestTwitterWidget(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWTwitter)
        # give some key to api - to allow start the search
        self.widget.update_api(twitter.Credentials("testkey", "testsecret"))

    def test_no_error(self):
        self.widget.search()
        self.assertFalse(self.widget.Error.empty_authors.is_shown())

    def test_empty_author_list(self):
        self.widget.mode = 1
        self.widget.mode_toggle()
        self.widget.search_button.click()
        self.wait_until_finished()
        self.assertTrue(self.widget.Error.empty_authors.is_shown())
        self.assertIsNone(self.get_output(self.widget.Outputs.corpus))

    @patch("orangecontrib.text.twitter.TwitterAPI.fetch", dummy_fetch)
    def test_content_search(self):
        self.widget.word_list = ["orange"]
        self.widget.search_button.click()
        output = self.get_output(self.widget.Outputs.corpus)
        self.assertEqual(3, len(output))
        self.assertGreater(len(str(output[0, "Content"])), 0)

    @patch("orangecontrib.text.twitter.TwitterAPI.fetch", dummy_fetch)
    def test_author(self):
        self.widget.mode = 1
        self.widget.word_list = ["@OrangeDataMiner"]
        self.widget.mode_toggle()
        self.widget.search_button.click()

        output = self.get_output(self.widget.Outputs.corpus)
        self.assertEqual(3, len(output))
        self.assertGreater(len(str(output[0, "Content"])), 0)

    @patch("tweepy.Cursor.items")
    def test_rate_limit(self, mock_items):
        mock_items.side_effect = TooManyRequests(Response(492))
        self.widget.word_list = ["orange"]
        self.widget.search_button.click()
        self.wait_until_finished()
        self.assertTrue(self.widget.Error.rate_limit.is_shown())
        self.assertEqual("Search", self.widget.search_button.text())

    @patch("tweepy.Cursor.items")
    def test_error(self, mock_items):
        mock_items.side_effect = TweepyException("Other errors", Response(400))
        self.widget.word_list = ["orange"]
        self.widget.search_button.click()
        self.wait_until_finished()
        self.assertTrue(self.widget.Error.api_error.is_shown())
        self.assertEqual("Search", self.widget.search_button.text())


if __name__ == "__main__":
    unittest.main()
