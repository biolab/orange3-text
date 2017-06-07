import os

from Orange.misc.environ import data_dir_base


def nltk_data_dir():
    """ Location where the NLTK data is stored. """
    dir_ = os.path.join(data_dir_base(), 'Orange', 'nltk_data')
    dir_ = "F:\\Orange"
    # make sure folder exists for ReadTheDocs
    os.makedirs(dir_, exist_ok=True)
    # return dir_
    return dir_
