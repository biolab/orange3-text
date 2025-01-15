import unittest
from collections import namedtuple
from datetime import date
from unittest.mock import patch, ANY, MagicMock, call

import numpy as np
import pandas as pd
from Orange.data.pandas_compat import table_to_frame

from orangecontrib.text import twitter
from orangecontrib.text.twitter import NoAuthorError

DummyTweet = namedtuple(
    "DummyTweet",
    [
        "id",
        "text",
        "author_id",
        "created_at",
        "lang",
        "public_metrics",
        "in_reply_to_user_id",
        "geo",
    ],
)
DummyUser = namedtuple(
    "DummyUser", ["id", "username", "name", "public_metrics", "verified", "description"]
)
DummyPlace = namedtuple("DummyUser", ["id", "country_code"])
DummyResponse = namedtuple("DummyRespons", ["data", "includes"])

EXPECTED_RESULTS = {
    "Content": ["my first tweet", "good start", "it is boring afternoon", "test"],
    "Author": ["@anaana", "@bert", "@bert", "@cc"],
    "Date": pd.to_datetime([date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 4)]),
    "Language": ["en", "it", "en", np.nan],
    "Location": ["sl", np.nan, np.nan, np.nan],
    "Number of Likes": [1, 100, 100, 1],
    "Number of Retweets": [0, 1000, 1000, 0],
    "In Reply To": [np.nan, "@anaana", np.nan, np.nan],
    "Author Name": ["Ana", "Berta", "Berta", "Cilka"],
    "Author Description": ["I am the best", "", "", "Haha"],
    "Author Tweets Count": [1, 100, 100, 1000],
    "Author Following Count": [200, 100, 100, 10],
    "Author Followers Count": [3, 22, 22, 44],
    "Author Listed Count": [1, 4, 4, 2],
    "Author Verified": ["True", "False", "False", "False"],
    "Longitude": [10, np.nan, np.nan, np.nan],
    "Latitude": [11, np.nan, np.nan, np.nan],
}
ER = pd.DataFrame(EXPECTED_RESULTS)

geo1 = {
    "place_id": "place1",
    "coordinates": {"coordinates": (ER.loc[0, "Longitude"], ER.loc[0, "Latitude"])},
}
pm1 = {
    "like_count": ER.loc[0, "Number of Likes"],
    "retweet_count": ER.loc[0, "Number of Retweets"],
}
pm2 = {
    "like_count": ER.loc[1, "Number of Likes"],
    "retweet_count": ER.loc[1, "Number of Retweets"],
}
tweets = [
    DummyTweet(
        "tweet1",
        ER.loc[0, "Content"],
        "user0",
        ER.loc[0, "Date"],
        ER.loc[0, "Language"],
        pm1,
        None,
        geo1,
    ),
    DummyTweet(
        "tweet2",
        ER.loc[1, "Content"],
        "user1",
        ER.loc[1, "Date"],
        ER.loc[1, "Language"],
        pm2,
        "user0",
        None,
    ),
    DummyTweet(
        "tweet3",
        ER.loc[2, "Content"],
        "user1",
        ER.loc[2, "Date"],
        ER.loc[2, "Language"],
        pm2,
        None,
        None,
    ),
    DummyTweet(
        "tweet4",
        ER.loc[3, "Content"],
        "user2",
        ER.loc[3, "Date"],
        None,
        pm1,
        None,
        None,
    ),
]
upm0 = {
    "tweet_count": ER.loc[0, "Author Tweets Count"],
    "following_count": ER.loc[0, "Author Following Count"],
    "followers_count": ER.loc[0, "Author Followers Count"],
    "listed_count": ER.loc[0, "Author Listed Count"],
}
upm1 = {
    "tweet_count": ER.loc[1, "Author Tweets Count"],
    "following_count": ER.loc[1, "Author Following Count"],
    "followers_count": ER.loc[1, "Author Followers Count"],
    "listed_count": ER.loc[1, "Author Listed Count"],
}
upm2 = {
    "tweet_count": ER.loc[3, "Author Tweets Count"],
    "following_count": ER.loc[3, "Author Following Count"],
    "followers_count": ER.loc[3, "Author Followers Count"],
    "listed_count": ER.loc[3, "Author Listed Count"],
}
users = [
    DummyUser(
        "user0",
        ER.loc[0, "Author"][1:],
        ER.loc[0, "Author Name"],
        upm0,
        ER.loc[0, "Author Verified"],
        ER.loc[0, "Author Description"],
    ),
    DummyUser(
        "user1",
        ER.loc[1, "Author"][1:],
        ER.loc[1, "Author Name"],
        upm1,
        ER.loc[1, "Author Verified"],
        ER.loc[1, "Author Description"],
    ),
    DummyUser(
        "user2",
        ER.loc[3, "Author"][1:],
        ER.loc[3, "Author Name"],
        upm2,
        ER.loc[3, "Author Verified"],
        ER.loc[3, "Author Description"],
    ),
]
places = [DummyPlace("place1", ER.loc[0, "Location"])]


class DummyPaginator:
    call_count = 0
    n = 2
    tweets = tweets
    users = users
    places = places

    def __init__(self, tweets, users, places):
        self.tweets = tweets
        self.users = users
        self.places = places

    def __call__(self, *args, **kwargs):
        self.call_count = 0
        return self

    def __iter__(self):
        return self

    def __next__(self):
        from_ = self.call_count * self.n
        self.call_count += 1
        if from_ < len(self.tweets):
            return DummyResponse(
                self.tweets[from_ : from_ + self.n],
                {"users": self.users, "places": self.places},
            )
        else:
            raise StopIteration


class TestTwitterAPI(unittest.TestCase):
    def setUp(self):
        self.client = twitter.TwitterAPI("key")

    @staticmethod
    def assert_query(mock, query):
        mock.assert_called_with(
            ANY,
            query,
            tweet_fields=ANY,
            user_fields=ANY,
            place_fields=ANY,
            expansions=ANY,
            max_results=100,
        )

    @staticmethod
    def assert_author_query(mock, user_id):
        mock.assert_called_with(
            ANY,
            user_id,
            tweet_fields=ANY,
            user_fields=ANY,
            place_fields=ANY,
            expansions=ANY,
            max_results=100,
        )

    @patch("tweepy.Paginator")
    def test_query_content(self, mock):
        self.client.search_content(["orange"])
        self.assert_query(mock, '"orange"')

        self.client.search_content(["orange", "day", "test"])
        self.assert_query(mock, '"orange" OR "day" OR "test"')

        self.client.search_content(["orange", "day"], lang="en")
        self.assert_query(mock, '"orange" OR "day" lang:en')

        self.client.search_content(["orange", "day"], allow_retweets=False)
        self.assert_query(mock, '"orange" OR "day" -is:retweet')

        self.client.search_content(["orange", "day"], lang="en", allow_retweets=False)
        self.assert_query(mock, '"orange" OR "day" -is:retweet lang:en')

    @patch("tweepy.Client.get_user")
    @patch("tweepy.Paginator")
    def test_query_authors(self, mock, user_mock):
        user_mock.return_value = MagicMock(data=MagicMock(id=1))

        self.client.search_authors(["orange"])
        user_mock.assert_called_with(username="orange")
        self.assert_author_query(mock, 1)
        user_mock.reset_mock()

        self.client.search_authors(["orange", "day", "test"])
        user_mock.assert_has_calls(
            [call(username="orange"), call(username="day"), call(username="test")]
        )
        self.assert_author_query(mock, 1)

    @patch("tweepy.Paginator", DummyPaginator(tweets, users, places))
    def test_load_data(self):
        corpus = self.client.search_content(["orange"])
        self.assertEqual(4, len(corpus))
        self.assertTupleEqual(tuple(m[0]() for m in twitter.METAS), corpus.domain.metas)

        df = table_to_frame(corpus, include_metas=True)
        pd.testing.assert_frame_equal(
            df.reset_index(drop=True), ER, check_dtype=False, check_categorical=False
        )

    @patch("tweepy.Client.get_user")
    @patch("tweepy.Paginator", DummyPaginator(tweets, users, places))
    def test_load_authors(self, user_mock):
        user_mock.return_value = MagicMock(data=MagicMock(id=1))

        corpus = self.client.search_authors(["orange"])
        self.assertEqual(4, len(corpus))
        self.assertTupleEqual(tuple(m[0]() for m in twitter.METAS), corpus.domain.metas)

        df = table_to_frame(corpus, include_metas=True)
        pd.testing.assert_frame_equal(
            df.reset_index(drop=True), ER, check_dtype=False, check_categorical=False
        )

    @patch("tweepy.Client.get_user")
    @patch("tweepy.Paginator", DummyPaginator(tweets, users, places))
    def test_history(self, user_mock):
        user_mock.return_value = MagicMock(data=MagicMock(id=1))

        self.client.search_authors(["orange"])
        self.assertListEqual(
            [
                (
                    ("Query", ["orange"]),
                    ("Search by", "Author"),
                    ("Language", None),
                    ("Allow retweets", "None"),
                    ("Tweets count", 4),
                )
            ],
            self.client.search_history,
        )
        self.assertEqual(4, len(self.client.tweets))

        self.client.reset()
        self.assertEqual(0, len(self.client.search_history))
        self.assertEqual(0, len(self.client.tweets))

        self.client.search_content(["orange"])
        self.assertListEqual(
            [
                (
                    ("Query", ["orange"]),
                    ("Search by", "Content"),
                    ("Language", "Any"),
                    ("Allow retweets", "True"),
                    ("Tweets count", 4),
                )
            ],
            self.client.search_history,
        )
        self.assertEqual(4, len(self.client.tweets))

        self.client.reset()
        self.client.search_content(["orange1"], lang="en", allow_retweets=False)
        self.assertListEqual(
            [
                (
                    ("Query", ["orange1"]),
                    ("Search by", "Content"),
                    ("Language", "English"),
                    ("Allow retweets", "False"),
                    ("Tweets count", 4),
                )
            ],
            self.client.search_history,
        )
        self.assertEqual(4, len(self.client.tweets))

    @patch("tweepy.Paginator", DummyPaginator(tweets, users, places))
    def test_tweet_history(self):
        """Test that tweets save in history only once and max tweets"""
        self.client.search_content(["orange"], max_tweets=2)
        self.assertEqual(2, len(self.client.tweets))
        self.client.search_content(["orange"], max_tweets=3)
        self.assertEqual(3, len(self.client.tweets))
        self.client.search_content(["orange"], max_tweets=4)
        self.assertEqual(4, len(self.client.tweets))

    @patch("tweepy.Client.get_user")
    @patch("tweepy.Paginator", DummyPaginator(tweets, users, places))
    def test_collecting(self, user_mock):
        user_mock.return_value = MagicMock(data=MagicMock(id=1))
        self.client.search_content(["orange"])
        self.assertEqual(4, self.client.search_history[-1][4][1])

        # when query repeated and collecting=False count only new tweets
        # (0 from dummy tweets)
        self.client.search_content(["orange"], collecting=True)
        self.assertEqual(0, self.client.search_history[-1][4][1])

        # when collecting=True tweets are recollected (previous tweets removed)
        self.client.search_content(["orange"], collecting=False)
        self.assertEqual(4, self.client.search_history[-1][4][1])

        # test for authors
        self.client.reset()
        self.client.search_authors(["orange"])
        self.assertEqual(4, self.client.search_history[-1][4][1])

        # when query repeated and collecting=False count only new tweets
        # (0 from dummy tweets)
        self.client.search_authors(["orange"], collecting=True)
        self.assertEqual(0, self.client.search_history[-1][4][1])

        # when collecting=True tweets are recollected (previous tweets removed)
        self.client.search_authors(["orange"], collecting=False)
        self.assertEqual(4, self.client.search_history[-1][4][1])

    @patch("tweepy.Client.get_user")
    def test_author_not_existing(self, user_patch):
        user_patch.return_value = MagicMock(data=None)
        with self.assertRaises(NoAuthorError):
            self.client.search_authors(["orange"], collecting=True)

    @patch("tweepy.Client.get_user")
    def test_tweets_language(self, user_mock):
        user_mock.return_value = MagicMock(data=MagicMock(id=1))

        with patch("tweepy.Paginator", DummyPaginator(tweets, users, places)):
            # language should be None returned tweets have different languages
            corpus = self.client.search_content(["orange"])
            self.assertIsNone(corpus.language)

            # corpus language should be same than language in the request
            corpus = self.client.search_content(["orange"], lang="en")
            self.assertEqual("en", corpus.language)

            # language should be None returned tweets have different languages
            corpus = self.client.search_content(["orange"])
            self.assertIsNone(corpus.language)

        with patch(
            "tweepy.Paginator", DummyPaginator([tweets[0], tweets[2]], users, places)
        ):
            # corpus language should be same than language in the request
            corpus = self.client.search_content(["orange"])
            self.assertEqual("en", corpus.language)

            corpus = self.client.search_content(["orange"])
            self.assertEqual("en", corpus.language)


if __name__ == "__main__":
    unittest.main()
