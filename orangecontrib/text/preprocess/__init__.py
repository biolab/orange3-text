from .filter import *
from .normalize import *
from .tokenize import *
from .transform import *
from .preprocess import *


base_preprocessor = Preprocessor(transformers=[LowercaseTransformer()],
                                 tokenizer=WordPunctTokenizer())
