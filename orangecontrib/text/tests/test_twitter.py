import json
import os
import pickle
import time
import unittest
from unittest import mock

import tweepy

from orangecontrib.text import twitter
from orangecontrib.text.corpus import Corpus


def get_credentials():
    key = os.environ.get('TWITTER_KEY', '')
    secret = os.environ.get('TWITTER_SECRET', '')
    if key and secret:
        return twitter.Credentials(key, secret)
    return twitter.Credentials('key', 'secret')


NO_CREDENTIALS = os.environ.get('TWITTER_KEY', None) is None
CREDENTIALS_MSG = "No twitter api credentials have been found."


class TestCredentials(unittest.TestCase):

    @unittest.skipIf(NO_CREDENTIALS, CREDENTIALS_MSG)
    def test_valid(self):
        credentials = get_credentials()
        self.assertTrue(credentials.valid)

    def test_check_bad(self):
        key = twitter.Credentials('bad key', 'wrong secret')
        self.assertFalse(key.valid)

    def test_equal(self):
        key1 = twitter.Credentials('key1', 'secret1')
        key2 = twitter.Credentials('key1', 'secret1')
        self.assertEqual(key1, key2)

        key2.consumer_secret = 'key2'
        self.assertNotEqual(key1, key2)

        self.assertNotEqual(key1, None)

    def test_pickle(self):
        key1 = twitter.Credentials('key1', 'secret1')
        pickled = pickle.dumps(key1)
        key2 = pickle.loads(pickled)
        self.assertEqual(key1, key2)


class MyCursor:
    def __init__(self, *args, **kwargs):
        time.sleep(.05)
        self.statuses = tweepy.Status.parse_list(
            None, json.load(open(os.path.join(os.path.dirname(__file__), 'tweets.json'))))
        self.kwargs = kwargs
        self.args = args

    def items(self, count):
        count = min(count, 20)
        return self.statuses[:count]


class MyCursor2(MyCursor):  # go from back forward
    def items(self, count):
        return super().items(20)[::-1][:count]


@mock.patch('tweepy.Cursor', MyCursor)
class TestTwitterAPI(unittest.TestCase):

    def setUp(self):
        self.credentials = get_credentials()
        self.api = twitter.TwitterAPI(self.credentials)

    def test_search_callbacks(self):
        self.checker = 0

        def on_progress(total, current):
            self.assertTrue(total % 20 == 0)
            self.checker += 1

        api = twitter.TwitterAPI(self.credentials, on_progress=on_progress)
        api.search_content('hello', max_tweets=20, lang='en')
        self.assertEqual(self.checker, 1)

    def test_create_corpus(self):
        self.api.search_content('hello', max_tweets=5)
        corpus = self.api.create_corpus()
        self.assertIsInstance(corpus, Corpus)
        self.assertEqual(len(corpus), 5)

    def test_clear(self):
        self.api.search_content(content=['hello'], max_tweets=5)
        self.assertEqual(len(self.api.container), 5)
        self.api.reset()
        self.assertEqual(len(self.api.container), 0)

    def test_report(self):
        api = twitter.TwitterAPI(self.credentials)
        api.search_content('hello', max_tweets=5, collecting=True)
        self.assertEqual(len(api.report()), 1)
        api.search_content('world', collecting=True)
        self.assertEqual(len(api.report()), 2)

    def test_empty_content(self):
        api = twitter.TwitterAPI(self.credentials)
        corpus = api.search_content('', max_tweets=10, allow_retweets=False)
        self.assertEqual(len(corpus), 10)

    def test_search_author(self):
        api = twitter.TwitterAPI(self.credentials)
        corpus = api.search_authors('hello', max_tweets=5)
        self.assertEqual(len(corpus), 5)

    def test_search_author_collecting(self):
        api = twitter.TwitterAPI(self.credentials)
        with unittest.mock.patch('tweepy.Cursor', MyCursor) as mock:
            corpus = api.search_authors('hello', max_tweets=5)
            self.assertEqual(len(corpus), 5)
        # MyCursor2 so we get different tweets
        with unittest.mock.patch('tweepy.Cursor', MyCursor2) as mock:
            corpus = api.search_authors('world', max_tweets=5, collecting=True)
            self.assertEqual(len(corpus), 10)

    def test_max_tweets_zero(self):
        api = twitter.TwitterAPI(self.credentials)
        corpus = api.search_content('hello', max_tweets=0)
        self.assertEqual(len(corpus), 20)   # 20 is the #tweets in cache

        corpus = api.search_authors('hello', max_tweets=0)
        self.assertEqual(len(corpus), 20)  # 20 is the #tweets in cache

    def test_geo_util(self):
        point = twitter.coordinates_geoJSON({})
        self.assertIsNone(point[0])
        self.assertIsNone(point[1])

        point = twitter.coordinates_geoJSON({'coordinates': [10, 10]})
        self.assertEqual(point[0], 10)
        self.assertEqual(point[1], 10)

    def test_breaking(self):
        count = 0

        def should_break():
            nonlocal count
            if count == 1:
                return True
            count += 1
            return False

        api = twitter.TwitterAPI(self.credentials, should_break=should_break)
        corpus = api.search_content('hello', max_tweets=10)
        self.assertEqual(len(corpus), 1)


class Response:
    def __init__(self, code):
        self.status_code = code


class TestTwitterAPIErrorRaising(unittest.TestCase):
    def setUp(self):
        self.credentials = get_credentials()

    def test_error_reporting(self):
        with unittest.mock.patch('tweepy.Cursor.items') as mock:
            mock.side_effect = tweepy.TweepError('', Response(500))
            error_callback = unittest.mock.Mock()
            api = twitter.TwitterAPI(self.credentials, on_error=error_callback)
            api.search_authors('hello', max_tweets=5)
            self.assertEqual(error_callback.call_count, 1)

    def test_rate_limit_reporting(self):
        with unittest.mock.patch('tweepy.Cursor.items') as mock:
            mock.side_effect = tweepy.TweepError('', Response(429))
            rate_callback = unittest.mock.Mock()
            api = twitter.TwitterAPI(self.credentials,
                                     on_rate_limit=rate_callback)
            api.search_authors('hello', max_tweets=5)
            self.assertEqual(rate_callback.call_count, 1)
