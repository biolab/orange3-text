import os
import sys
import time
from functools import wraps
from threading import Thread

import nltk
from Orange.misc.environ import data_dir_base

__all__ = ['wait_nltk_data', 'nltk_data_dir']

NLTK_DATA = [
    'wordnet',
    'stopwords',
    'punkt',
    'opinion_lexicon',
    'vader_lexicon',
    'averaged_perceptron_tagger',
    'maxent_treebank_pos_tagger',
]


def nltk_data_dir():
    """ Location where the NLTK data is stored. """
    dir_ = os.path.join(data_dir_base(), 'Orange', 'nltk_data')
    # make sure folder exists for ReadTheDocs
    os.makedirs(dir_, exist_ok=True)
    return dir_


is_done_loading = False


def _download_nltk_data():
    global is_done_loading
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
