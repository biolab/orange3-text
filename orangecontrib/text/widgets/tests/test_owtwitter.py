import unittest
from unittest.mock import patch, MagicMock

from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate
from tweepy import TooManyRequests, TweepyException

from orangecontrib.text.tests.test_twitter import (
    DummyPaginator,
    tweets,
    users,
    places,
    TestTwitterAPI,
)
from orangecontrib.text.widgets.owtwitter import OWTwitter


@patch("tweepy.Client.get_user", MagicMock())
class TestTwitterWidget(WidgetTest):
    def setUp(self):
        self.widget: OWTwitter = self.create_widget(OWTwitter)
        # give some key to api - to allow start the search
        self.widget.update_api("test_key")

    def test_empty_query_error(self):
        self.widget.search_button.click()
        self.assertTrue(self.widget.Error.empty_query.is_shown())
        self.assertTrue(str(self.widget.Error.empty_query).endswith("keywords."))
        self.assertIsNone(self.get_output(self.widget.Outputs.corpus))

        simulate.combobox_activate_item(self.widget.controls.mode, "Author")
        self.widget.search_button.click()
        self.assertTrue(self.widget.Error.empty_query.is_shown())
        self.assertTrue(str(self.widget.Error.empty_query).endswith("authors."))
        self.assertIsNone(self.get_output(self.widget.Outputs.corpus))

    @patch("tweepy.Paginator", DummyPaginator(tweets, users, places))
    def test_author(self):
        simulate.combobox_activate_item(self.widget.controls.mode, "Author")
        self.widget.word_list = ["@OrangeDataMiner"]
        self.widget.search_button.click()

        output = self.get_output(self.widget.Outputs.corpus)
        self.assertEqual(4, len(output))

        self.widget.word_list = ["OrangeDataMiner", "test"]
        self.widget.search_button.click()

        output = self.get_output(self.widget.Outputs.corpus)
        self.assertEqual(4, len(output))

        self.widget.word_list = []
        self.widget.search_button.click()

        output = self.get_output(self.widget.Outputs.corpus)
        self.assertIsNone(output)

    @patch("tweepy.Paginator", DummyPaginator(tweets, users, places))
    def test_content(self):
        simulate.combobox_activate_item(self.widget.controls.mode, "Content")
        self.widget.word_list = ["OrangeDataMiner"]
        self.widget.search_button.click()

        output = self.get_output(self.widget.Outputs.corpus)
        self.assertEqual(4, len(output))

        self.widget.word_list = ["OrangeDataMiner", "test"]
        self.widget.search_button.click()

        output = self.get_output(self.widget.Outputs.corpus)
        self.assertEqual(4, len(output))

        self.widget.word_list = []
        self.widget.search_button.click()

        output = self.get_output(self.widget.Outputs.corpus)
        self.assertIsNone(output)

    @patch("tweepy.Paginator")
    def test_rate_limit(self, mock_items):
        mock_items.__iter__.side_effect = TooManyRequests(MagicMock())
        self.widget.word_list = ["orange"]
        self.widget.search_button.click()
        self.wait_until_finished()
        self.assertTrue(self.widget.Info.nut_enough_tweets.is_shown())
        # since rate error happen at beginning no tweets are download so far
        self.assertIsNone(self.get_output(self.widget.Outputs.corpus))
        self.assertEqual("Search", self.widget.search_button.text())

    @patch("tweepy.Paginator", side_effect=TweepyException("Other"))
    def test_tweepy_error(self, _):
        self.widget.word_list = ["orange"]
        self.widget.search_button.click()
        self.wait_until_finished()
        self.assertTrue(self.widget.Error.api_error.is_shown())
        self.assertTrue(str(self.widget.Error.api_error).startswith("Api error: Other"))
        self.assertEqual("Search", self.widget.search_button.text())

    def test_author_not_existing(self):
        with patch("tweepy.Client.get_user") as m:
            m.return_value = MagicMock(data=None)
            simulate.combobox_activate_item(self.widget.controls.mode, "Author")
            self.widget.word_list = ["orange"]
            self.widget.search_button.click()
            self.wait_until_finished()
            self.assertTrue(self.widget.Error.wrong_author.is_shown())
            self.assertEqual(
                "Author 'orange' does not exist.", str(self.widget.Error.wrong_author)
            )
            self.assertEqual("Search", self.widget.search_button.text())

    @patch("tweepy.Paginator")
    def test_language(self, mock):
        simulate.combobox_activate_item(self.widget.controls.mode, "Content")
        simulate.combobox_activate_item(self.widget.language_combo, "English")
        self.widget.word_list = ["OrangeDataMiner"]
        self.widget.search_button.click()
        self.wait_until_finished()

        TestTwitterAPI.assert_query(mock, '"OrangeDataMiner" -is:retweet lang:en')
        mock.reset_mock()

        simulate.combobox_activate_item(self.widget.language_combo, "Slovenian")
        self.widget.search_button.click()
        self.wait_until_finished()

        TestTwitterAPI.assert_query(mock, '"OrangeDataMiner" -is:retweet lang:sl')
        mock.reset_mock()

        simulate.combobox_activate_item(self.widget.language_combo, "German")
        self.widget.search_button.click()
        self.wait_until_finished()

        TestTwitterAPI.assert_query(mock, '"OrangeDataMiner" -is:retweet lang:de')

    @patch("tweepy.Paginator")
    def test_is_retweet(self, mock):
        self.widget.retweets_checkbox.setChecked(False)
        self.widget.word_list = ["OrangeDataMiner"]
        self.widget.search_button.click()
        self.wait_until_finished()

        TestTwitterAPI.assert_query(mock, '"OrangeDataMiner" -is:retweet')
        mock.reset_mock()

        self.widget.retweets_checkbox.setChecked(True)
        self.widget.search_button.click()
        self.wait_until_finished()

        TestTwitterAPI.assert_query(mock, '"OrangeDataMiner"')
        mock.reset_mock()

        self.widget.retweets_checkbox.setChecked(False)
        self.widget.search_button.click()
        self.wait_until_finished()

        TestTwitterAPI.assert_query(mock, '"OrangeDataMiner" -is:retweet')
        mock.reset_mock()

    @patch("tweepy.Paginator", DummyPaginator(tweets, users, places))
    def test_max_tweets(self):
        simulate.combobox_activate_item(self.widget.controls.mode, "Content")
        self.widget.controls.max_tweets.setValue(2)
        self.widget.word_list = ["OrangeDataMiner"]
        self.widget.search_button.click()

        output = self.get_output(self.widget.Outputs.corpus)
        self.assertEqual(2, len(output))

        self.widget.controls.max_tweets.setValue(3)
        self.widget.word_list = ["OrangeDataMiner"]
        self.widget.search_button.click()

        output = self.get_output(self.widget.Outputs.corpus)
        self.assertEqual(3, len(output))

    @patch("tweepy.Paginator")
    def test_send_report(self, _):
        simulate.combobox_activate_item(self.widget.controls.mode, "Content")
        self.widget.controls.max_tweets.setValue(2)
        self.widget.word_list = ["OrangeDataMiner"]
        self.widget.search_button.click()
        self.wait_until_finished()

        self.widget.send_report()

    @patch("tweepy.Paginator", DummyPaginator(tweets, users, places))
    def test_interrupted(self):
        self.widget.word_list = ["OrangeDataMiner"]
        self.widget.search_button.click()
        self.assertEqual("Stop", self.widget.search_button.text())
        self.widget.search_button.click()
        self.wait_until_finished()
        self.assertEqual("Search", self.widget.search_button.text())


if __name__ == "__main__":
    unittest.main()
