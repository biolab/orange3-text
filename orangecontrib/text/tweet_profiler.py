import logging

import numpy as np
import requests

log = logging.getLogger(__name__)


class TweetProfiler:
    BATCH_SIZE = 50
    TIMEOUT = 15    # how many seconds to wait for a server to respond
    SERVER_LIST = 'https://raw.githubusercontent.com/biolab/' \
                  'orange3-text/master/SERVERS.txt'

    def __init__(self, server=None, on_server_down=None):
        self.server = server or self.get_server_address()
        self.on_server_down = on_server_down
        self.model_names = []
        self.output_modes = []

        self.assure_server()
        self.set_configuration()

    def transform(self, corpus, meta_var, model_name, output_mode,
                  on_advance=None):
        if not self.assure_server():
            return None

        corpus = corpus.copy()
        metas_ind = corpus.domain.metas.index(meta_var)
        tweets = corpus.metas[:, metas_ind].tolist()

        results = []
        class_vars = []
        target_mode = None
        for i in range(0, len(corpus), self.BATCH_SIZE):
            json = {'tweets': tweets[i: i+self.BATCH_SIZE],
                    'model_name': model_name,
                    'output_mode': output_mode,
            }

            json = self.server_call('tweet_profiler', json=json)

            if json is None:    # something went wrong
                return None

            class_vars = json['classes']
            profile = np.array(json['profile'])
            target_mode = json['target_mode']
            results.append(profile)

            if callable(on_advance):
                on_advance(self.BATCH_SIZE)

        if results:
            results = np.vstack(results)
            feature_names = None
            feature_values = None
            var_attrs = None

            if output_mode == 'Embeddings':
                feature_names = class_vars
                var_attrs = {'hidden': True}
            elif output_mode == 'Probabilities':
                feature_names = class_vars
            elif output_mode == 'Classes' and target_mode == 'mc':
                feature_names = ['Emotion']
                feature_values = [class_vars]
            elif output_mode == 'Classes' and target_mode == 'ml':
                feature_names = class_vars
                feature_values = tuple(['no', 'yes'] for _ in class_vars)

            corpus.extend_attributes(results,
                                     feature_names=feature_names,
                                     feature_values=feature_values,
                                     var_attrs=var_attrs)
            return corpus
        else:
            return None

    def get_server_address(self):
        # return 'http://127.0.0.1:8081/'
        try:
            res = requests.get(self.SERVER_LIST)
        except requests.exceptions.ConnectionError:
            log.error("Could not access server list at: {}\n.".format(
                self.SERVER_LIST))
            return None

        for server in res.text.split('\n'):
            server = server.strip()
            if server and self.check_server_alive(server):
                return server
        return None

    @staticmethod
    def check_server_alive(server):
        try:
            r = requests.head(server)
            status = r.status_code
            alive = status == 200
            return alive
        except requests.exceptions.ConnectionError:
            log.error('Server {} is not responding.'.format(server))
            return False

    def assure_server(self):
        if not self.server:     # try to obtain server address
            self.server = self.get_server_address()

        if not self.server or not self.check_server_alive(self.server):
            if callable(self.on_server_down):
                self.on_server_down()
            return False
        return True

    def server_call(self, url, json=None):
        if not self.server:
            return None

        try:
            res = requests.post(self.server + '/' + url, json=json,
                                timeout=self.TIMEOUT)
            return res.json()
        except requests.exceptions.RequestException as e:
            if callable(self.on_server_down):
                self.on_server_down(str(e))
            return None

    def set_configuration(self):
        json = self.server_call('get_configurations')
        if json:
            self.model_names = json['models']
            self.output_modes = json['output_modes']
