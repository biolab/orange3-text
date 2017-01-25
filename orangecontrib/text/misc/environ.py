import os

from Orange.misc.environ import data_dir_base


def nltk_data_dir():
    """ Location where the NLTK data is stored. """
    return os.path.join(data_dir_base(), 'Orange', 'nltk_data')
