# Ensure NLTK data is present
import nltk
NLTK_DATA = ["wordnet",  "stopwords",  "punkt"]
nltk.download(NLTK_DATA, quiet=True)

from .filter import *
from .normalize import *
from .tokenize import *
from .transform import *
from .preprocess import *



base_preprocessor = Preprocessor(transformers=[LowercaseTransformer()],
                                 tokenizer=WordPunctTokenizer())
