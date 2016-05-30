import unittest
import os

import pickle
from itertools import chain

from orangecontrib.text import twitter
from orangecontrib.text.corpus import Corpus


def get_valid_credentials():
    key = os.environ.get('TWITTER_KEY', '')
    secret = os.environ.get('TWITTER_SECRET', '')
    if key and secret:
        return twitter.Credentials(key, secret)


class TestCredentials(unittest.TestCase):

    def test_valid(self):
        credentials = get_valid_credentials()
        if credentials is None:
            self.skipTest("Credentials have not been provided.")
        else:
            self.assertTrue(credentials.valid)

    def test_check_bad(self):
        key = twitter.Credentials('bad key', 'wrong secret')
        self.assertFalse(key.check())

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


class TestTwitterAPI(unittest.TestCase):

    def setUp(self):
        self.credentials = get_valid_credentials()
        if self.credentials is None:
            self.skipTest("Valid credentials has not provided.")

        self.api = twitter.TwitterAPI(self.credentials)

    def test_search(self):
        self.api.search(word_list=['hello'], max_tweets=5, lang='en')
        self.api.join()
        self.assertEqual(len(self.api.container), 5)
        for tweet in self.api.tweets:
            self.assertIn('hello', tweet['text'].lower())
            self.assertEqual('en', tweet['lang'])

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

        def on_error(error):
            self.assertEqual(self.checker, 4)
            self.checker += 1

        api = twitter.TwitterAPI(twitter.Credentials('bad', 'credentials'),
                                 on_error=on_error)

        api.search(word_list=['hello'], max_tweets=2, lang='en')
        api.join()
        self.assertEqual(self.checker, 5)

    def test_search_disconnect(self):
        self.api.search(word_list=['hello'], max_tweets=50, lang='en')
        self.api.disconnect()
        self.api.join()
        self.assertLess(len(self.api.container), 20)

    def test_create_corpus(self):
        self.api.search(word_list=['hello'], max_tweets=5)
        self.api.join()
        corpus = self.api.create_corpus()
        self.assertIsInstance(corpus, Corpus)
        self.assertEqual(len(corpus), 5)

    def test_crate_corpus_attr_selection(self):
        self.api.search(word_list=['hello'], max_tweets=5)
        self.api.join()
        attributes = ['text', 'created_at', 'author_id']
        corpus = self.api.create_corpus(included_attributes=attributes)
        domain_attributes = [attr.name for attr in chain(corpus.domain.attributes, corpus.domain.metas)]
        self.assertEqual(len(domain_attributes), 3)
        for attr in attributes:
            self.assertIn(attr, domain_attributes)

    def test_clear(self):
        self.api.search(word_list=['hello'], max_tweets=5)
        self.api.join()
        self.assertEqual(len(self.api.container), 5)

        self.api.reset()
        self.assertEqual(len(self.api.container), 0)

    def test_running(self):
        self.assertFalse(self.api.running)
        self.api.search(word_list=['hello'], max_tweets=5)
        self.assertTrue(self.api.running)
        self.api.disconnect()
        self.assertFalse(self.api.running)

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
