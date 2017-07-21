import unittest
import requests
import numpy as np
from unittest.mock import patch, Mock

from orangecontrib.text.corpus import Corpus
from orangecontrib.text.tweet_profiler import TweetProfiler

SERVER_CALL = 'orangecontrib.text.tweet_profiler.TweetProfiler.server_call'
CHECK_ALIVE = 'orangecontrib.text.tweet_profiler.TweetProfiler.check_server_alive'

MODELS = ['model-mc', 'model-ml']
MODES = ['Embeddings', 'Probabilities', 'Classes']
PROFILE_CLASSES = ['c1', 'c2', 'c3']
EMBEDDINGS_NUM = 100


class MockServerCall(unittest.TestCase):
    def __call__(self, *args, **kwargs):
        # print('Got called with:', args, kwargs)
        url = args[0]
        json = kwargs.get('json', None)

        if url == 'get_configurations':
            return {'models': MODELS, 'output_modes': MODES}
        elif url == 'tweet_profiler':
            self.assertIn('tweets', json)
            self.assertIn('model_name', json)
            self.assertIn('output_mode', json)

            n_tweets = len(json['tweets'])
            output_mode = json['output_mode']
            mcml = json['model_name'].split('-')[1]

            if output_mode == 'Embeddings':
                return {
                    'classes': ['Feature {}'.format(i)
                                for i in range(EMBEDDINGS_NUM)],
                    'profile': np.zeros((n_tweets, EMBEDDINGS_NUM)),
                    'target_mode': mcml,
                }
            else:
                n_profiles = len(PROFILE_CLASSES) if mcml == 'ml' else 1
                return {
                    'classes': PROFILE_CLASSES,
                    'profile': np.zeros((n_tweets, n_profiles)),
                    'target_mode': mcml,
                }


class MockServerCallSuddenlyUnavailable(MockServerCall):
    # Mock server that becomes unavailable during execution
    def __call__(self, *args, **kwargs):
        url = args[0]
        if url == 'tweet_profiler':
            return None
        else:
            return super().__call__(*args, **kwargs)


class TestTweetProfiler(unittest.TestCase):
    @patch(SERVER_CALL, MockServerCall())
    @patch(CHECK_ALIVE, Mock(return_value=True))
    def setUp(self):
        self.data = Corpus.from_file('election-tweets-2016.tab')[:100]
        self.profiler = TweetProfiler()

    @patch(CHECK_ALIVE, Mock(return_value=True))
    def test_get_server_address(self):
        server = self.profiler.get_server_address()
        self.assertTrue(server.startswith('http'))

    @patch(SERVER_CALL, MockServerCall())
    def test_get_configuration(self):
        self.assertEqual(self.profiler.model_names, MODELS)
        self.assertEqual(self.profiler.output_modes, MODES)

    @patch(SERVER_CALL, MockServerCall())
    @patch(CHECK_ALIVE, Mock(return_value=True))
    def test_transform_embeddings(self):
        advance_call_mock = Mock()
        text_var = self.data.domain.metas[1]
        corp = self.profiler.transform(self.data, text_var,
                                       'model-mc', 'Embeddings',
                                       on_advance=advance_call_mock)
        self.assertIsInstance(corp, Corpus)
        self.assertEqual(advance_call_mock.call_count, 2)
        self.assertEqual(len(corp.domain.attributes) -
                         len(self.data.domain.attributes),
                         EMBEDDINGS_NUM)
        self.assertEqual(corp.X.shape[1] - self.data.X.shape[1],
                         EMBEDDINGS_NUM)
        self.assertEqual(np.sum(corp.X[:, -EMBEDDINGS_NUM]), 0)

    @patch(SERVER_CALL, MockServerCall())
    @patch(CHECK_ALIVE, Mock(return_value=True))
    def test_transform_probabilities_and_ml_classes(self):
        text_var = self.data.domain.metas[1]
        for mode in ['Probabilities', 'Classes']:
            corp = self.profiler.transform(self.data, text_var,
                                           'model-ml', mode)
            self.assertIsInstance(corp, Corpus)
            self.assertEqual(len(corp.domain.attributes) -
                             len(self.data.domain.attributes),
                             len(PROFILE_CLASSES))
            self.assertEqual(corp.X.shape[1] - self.data.X.shape[1],
                             len(PROFILE_CLASSES))
            self.assertEqual(np.sum(corp.X[:, -len(PROFILE_CLASSES)]), 0)

    @patch(SERVER_CALL, MockServerCall())
    @patch(CHECK_ALIVE, Mock(return_value=True))
    def test_transform_mc_classes(self):
        text_var = self.data.domain.metas[1]
        corp = self.profiler.transform(self.data, text_var,
                                       'model-mc', 'Classes')
        self.assertIsInstance(corp, Corpus)
        self.assertEqual(len(corp.domain.attributes) -
                         len(self.data.domain.attributes),
                         1)
        self.assertEqual(corp.X.shape[1] - self.data.X.shape[1], 1)
        self.assertEqual(np.sum(corp.X[:, -1]), 0)

    @patch(CHECK_ALIVE, Mock(return_value=False))
    def test_transform_probabilities(self):
        text_var = self.data.domain.metas[1]
        corp = self.profiler.transform(self.data, text_var,
                                       MODELS[0], 'Classes')
        self.assertIsNone(corp)

    @patch(SERVER_CALL, MockServerCall())
    @patch(CHECK_ALIVE, Mock(return_value=True))
    def test_transform_empty_corpus(self):
        text_var = self.data.domain.metas[1]
        corp = self.profiler.transform(self.data[:0], text_var,
                                       'model-mc', 'Classes')
        self.assertIsNone(corp)

    @patch(SERVER_CALL, MockServerCallSuddenlyUnavailable())
    @patch(CHECK_ALIVE, Mock(return_value=True))
    def test_transform_server_call_error(self):
        text_var = self.data.domain.metas[1]
        corp = self.profiler.transform(self.data, text_var,
                                       'model-mc', 'Classes')
        self.assertIsNone(corp)

response_mock = Mock()
response_mock.status_code = 200
response_mock.json = lambda: {}


class TestErrorsRaising(unittest.TestCase):
    @patch(SERVER_CALL, MockServerCall())
    @patch(CHECK_ALIVE, Mock(return_value=True))
    def setUp(self):
        self.profiler = TweetProfiler()

    @patch('requests.get', Mock(side_effect=requests.exceptions.ConnectionError))
    def test_get_server_address_connection_error(self):
        address = self.profiler.get_server_address()
        self.assertIsNone(address)

    @patch(CHECK_ALIVE, Mock(return_value=False))
    def test_get_server_address_server_down(self):
        address = self.profiler.get_server_address()
        self.assertIsNone(address)

    @patch('requests.head', Mock(return_value=response_mock))
    def test_server_alive(self):
        self.assertTrue(TweetProfiler.check_server_alive(''))

    @patch('requests.head', Mock(side_effect=requests.exceptions.ConnectionError))
    def test_server_raise_error(self):
        self.assertFalse(TweetProfiler.check_server_alive(''))

    @patch(CHECK_ALIVE, Mock(return_value=False))
    def test_assure_server_and_tokens_server_down(self):
        self.profiler.on_server_down = Mock()
        r = self.profiler.assure_server()
        self.assertFalse(r)
        self.assertEqual(self.profiler.on_server_down.call_count, 1)

    @patch(SERVER_CALL, MockServerCall())
    def test_assure_server_and_tokens_get_server(self):
        self.profiler.server = None
        r = self.profiler.assure_server()
        self.assertTrue(r)

    def test_server_call_server_missing(self):
        self.profiler.server = None
        r = self.profiler.server_call('', {})
        self.assertIsNone(r)

    @patch('requests.post', Mock(return_value=response_mock))
    def test_server_call_server_ok(self):
        r = self.profiler.server_call('', {})
        self.assertEqual(r, {})

    @patch('requests.post', Mock(side_effect=requests.exceptions.RequestException))
    def test_server_call_server_ok(self):
        self.profiler.on_server_down = Mock()
        r = self.profiler.server_call('', {})
        self.assertIsNone(r)
        self.assertEqual(self.profiler.on_server_down.call_count, 1)
