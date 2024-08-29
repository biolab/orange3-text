import logging
import os
import sys
import time
from functools import wraps
from threading import Thread
from urllib.parse import urlparse, ParseResult

import nltk
from Orange.misc.environ import data_dir_base
from Orange.misc.utils.embedder_utils import get_proxies


__all__ = ['wait_nltk_data', 'nltk_data_dir']

log = logging.getLogger(__name__)


NLTK_DATA = [
    'wordnet',
    'stopwords',
    'punkt',
    'opinion_lexicon',
    'vader_lexicon',
    'averaged_perceptron_tagger_eng',
    'maxent_treebank_pos_tagger_tab',
    'omw-1.4',
]


def nltk_data_dir():
    """ Location where the NLTK data is stored. """
    dir_ = os.path.join(data_dir_base(), 'Orange', 'nltk_data')
    # make sure folder exists for ReadTheDocs
    os.makedirs(dir_, exist_ok=True)
    return dir_


is_done_loading = False


# for any other potential scheme, it should be provided by user
DEFAULT_PORTS = {
    "http": "80",
    "https": "443",
    "socks4": "1080",
    "socks": "1080",
    "quic": "443",
}


def _get_proxy_address():
    """
    Set proxy addresses for NLTK since NLTK do not use proxy addresses from
    https_proxy environment variable
    """
    proxies = get_proxies() or {}
    # nltk uses https to download data
    if "https://" in proxies:
        proxy = urlparse(proxies['https://'])
        log.debug(f"Using proxy for NLTK: {proxy}")
        port = proxy.port or DEFAULT_PORTS.get(proxy.scheme)
        url = ParseResult(
            scheme=proxy.scheme,
            netloc="{}:{}".format(proxy.hostname, port) if port else proxy.netloc,
            path=proxy.path,
            params=proxy.params,
            query=proxy.query,
            fragment=proxy.fragment
        ).geturl()
        return url


def _download_nltk_data():
    global is_done_loading

    proxy_address = _get_proxy_address()
    if proxy_address:
        nltk.set_proxy(proxy_address)
    nltk.download(NLTK_DATA, download_dir=nltk_data_dir(), quiet=True)
    is_done_loading = True
    sys.stdout.flush()


Thread(target=_download_nltk_data).start()


def wait_nltk_data(func):
    """ Decorator that waits until all NLTK data is downloaded. """
    dir_ = nltk_data_dir()
    if dir_ not in nltk.data.path:  # assure NLTK knows where the data is
        nltk.data.path.append(dir_)

    @wraps(func)
    def wrapper(*args, **kwargs):
        global is_done_loading
        while not is_done_loading:
            time.sleep(.1)
        return func(*args, **kwargs)
    return wrapper
