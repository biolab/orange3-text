import unittest
from unittest import mock
import os

import pickle

import json

import tweepy
from datetime import date
import time

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
        return self.statuses[:count]


@mock.patch('tweepy.Cursor', MyCursor)
class TestTwitterAPI(unittest.TestCase):

    def setUp(self):
        self.credentials = get_credentials()
        self.api = twitter.TwitterAPI(self.credentials)

    def test_search_callbacks(self):
        self.checker = 0

        def on_start():
            self.assertEqual(self.checker, 0)
            self.checker += 1

        def on_progress(progress):
            self.assertEqual(self.checker, progress)
            self.checker += 1

        def on_finish():
            self.assertEqual(self.checker, 3)
            self.checker += 1

        api = twitter.TwitterAPI(self.credentials, on_start=on_start,
                                 on_progress=on_progress, on_finish=on_finish)
        api.search(word_list=['hello'], max_tweets=2, lang='en')
        api.join()
        self.assertEqual(self.checker, 4)

    def test_create_corpus(self):
        self.api.search(word_list=['hello'], max_tweets=5)
        self.api.join()
        corpus = self.api.create_corpus()
        self.assertIsInstance(corpus, Corpus)
        self.assertEqual(len(corpus), 5)

    def test_clear(self):
        self.api.search(word_list=['hello'], max_tweets=5)
        self.api.join()
        self.assertEqual(len(self.api.container), 5)

        self.api.reset()
        self.assertEqual(len(self.api.container), 0)

    def test_report(self):
        self.api.search(word_list=['hello'], max_tweets=5)
        self.api.join()
        self.assertEqual(len(self.api.history), 1)
        self.assertIsNotNone(self.api.task.report())

    def test_geo_util(self):
        point = twitter.coordinates_geoJSON({})
        self.assertIsNone(point[0])
        self.assertIsNone(point[1])

        point = twitter.coordinates_geoJSON({'coordinates': [10, 10]})
        self.assertEqual(point[0], 10)
        self.assertEqual(point[1], 10)

    def test_build_query(self):
        # https://dev.twitter.com/rest/public/search

        query = self.api.build_query(word_list=['hello', 'world'])
        self.assertIn('hello', query)
        self.assertIn('world', query)

        query = self.api.build_query(authors=['johndoe'])
        self.assertIn('from:johndoe', query)

        query = self.api.build_query(since=date(2016, 10, 9))
        self.assertIn('since:2016-10-09', query)

        query = self.api.build_query(until=date(2016, 10, 9))
        self.assertIn('until:2016-10-09', query)

        query = self.api.build_query(word_list=['hello', 'world'], allow_retweets=False)
        self.assertIn(' -filter:retweets', query)


@mock.patch('tweepy.Cursor', MyCursor)
class TestSearch(unittest.TestCase):
    def setUp(self):
        self.credentials = get_credentials()
        self.api = twitter.TwitterAPI(self.credentials)

    def test_running(self):
        self.assertFalse(self.api.running)
        self.api.search(word_list=['hello'], max_tweets=5)
        self.assertTrue(self.api.running)
        self.api.disconnect()
        self.assertFalse(self.api.running)

    def test_search_disconnect(self):
        self.api.search(word_list=['hello'], max_tweets=20, lang='en')
        self.api.disconnect()
        self.api.join()
        self.assertLess(len(self.api.container), 10)
