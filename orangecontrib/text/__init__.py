# Set where NLTK data is downloaded
import os
from orangecontrib.text.misc import nltk_data_dir
os.environ['NLTK_DATA'] = nltk_data_dir()

from .corpus import Corpus

from .version import git_revision as __git_revision__
from .version import version as __version__
