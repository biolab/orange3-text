import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

from orangecontrib.text.corpus import Corpus

# Ensure NLTK data is present
NLTK_DATA = ["wordnet",  "stopwords",  "punkt"]
nltk.download(NLTK_DATA)


class Preprocessor:
    """A pre-processing class for the Orange text mining extension.

    The class is capable of tokenizing, lowercasing, stemming and lemmatizing
    the input. Returns a list of preprocessed tokens, sorted by order of
    appearance in text.
    """
    def __init__(self, lowercase=True, stop_words=None,
                 min_df=None, max_df=None, transformation=None,
                 use_twitter_tokenizer=False, callback=None):
        """
        Args:
            lowercase (Optional[bool]): If set, transform the tokens to lower
                case, before returning them.
            stop_words (Optional[str, list]): Determines whether and what
                stop words should be removed. Can remove language specific stop
                words, supported by the NLTK or stop words provided in a custom
                list.
            min_df (Optional[int, float]): Tokens that appear in
                less than 'min_df' documents, will be removed. Can be
                specified either as an actual count of documents or a
                proportion of the corpus. Assigning this parameter when
                pre-processing a single document, will have no effect.
            max_df (Optional[int, float]): Tokens that appear in
                more than 'max_df' documents, will be removed. Can be
                specified either as an actual count of documents or a
                proportion of the corpus. Assigning this parameter when
                pre-processing a single document, will have no effect.
            transformation (Optional[`orangecontrib.text
                .preprocess.Stemmatizer`]): Name of the morphological
                transformation method to be performed on the tokens.
            use_twitter_tokenizer (Optional[bool]): Determines the use of
                either the Twitter or default word tokenizer.
            callback (Callable (function or method)): The callback that should
                be performed when a single document is finished pre-processing.

        Returns:
            list: Holds a list of lists, that correspond to documents in the
                input corpus.

        Raises:
            ValueError: If the stop words or transformation arguments are not
                of expected values/types.
        """
        # Tokenizer.
        self.tokenizer = word_tokenize
        if use_twitter_tokenizer:
            self.tokenizer = None   # TODO: Change when twitter is available.

        # Lowercase.
        self.lowercase = lowercase

        # Stop words.
        if isinstance(stop_words, str):
            try:
                self.stop_words = set(stopwords.words(stop_words))
            except Exception:
                raise ValueError('Could not find a stop word set for "{}".'
                                 .format(stop_words))
        elif isinstance(stop_words, list):
            # Checking custom stop words for faulty parameters (non str).
            for sw in stop_words:
                assert isinstance(sw, str)
            self.stop_words = set(stop_words)
        elif stop_words is None:
            self.stop_words = set()
        else:
            raise ValueError('The stop words parameter should be either a string,'
                             'a list containing the stop words or None.')

        # Min/Max df.
        # Constant checks whether or not we should use 'df', are quite painful.
        self.use_df_sw = False  # This flag summarizes this check.
        self._check_df_constraints(min_df, max_df)
        self.min_df = min_df
        self.max_df = max_df
        self.df_stop_words = set()

        # Transformation.
        if transformation is not None and not isinstance(transformation,
                                                         Stemmatizer):
            raise ValueError('Type "{}" not supported.'.format(transformation))
        self.transformation = transformation

        # Callback.
        self.callback = callback

    def __call__(self, input_data):
        """Pre-process the input using the settings specified.

        Args:
            input_data (str, list, `orangecontrib.text.corpus.Corpus`): The
                input we want to process.

        Returns:
            str, `orangecontrib.text.corpus.Corpus`: The pre-processed input in
                the form of either a string or a corpus, depending on the
                input.

        Raises:
            ValueError: If the input is not of expected type.
        """
        # ---- INPUT CHECKS ---- #
        is_corpus = isinstance(input_data, Corpus)
        if isinstance(input_data, str):
            data = [input_data]
        elif is_corpus:
            data = input_data.documents
        elif isinstance(input_data, list):
            data = input_data
        else:
            raise ValueError(
                    'Type "{}" not supported.'.format(type(input_data))
            )

        # ---- TOKENIZE ---- #
        tokens = []
        for document in data:
            tokens.append(self.tokenizer(document))
            if self.callback:
                self.callback()

        # ---- DETERMINE DF STOP WORDS ---- #
        if self.use_df_sw:  # DF parameters were specified.
            self._get_corpus_specific_stopwords(tokens, self.lowercase)

        # ---- PRE-PROCESS DOCUMENTS ---- #
        pp_tokens = []
        stop_words = self.stop_words | self.df_stop_words
        assert isinstance(stop_words, set)
        for doc in tokens:
            if self.lowercase:  # Lowercase.
                doc = [token.lower() for token in doc]

            if stop_words:  # Remove stop words.
                doc = [word for word in doc if word not in stop_words]

            if self.transformation is not None:  # Transform.
                doc = self.transformation(doc)

            pp_tokens.append(doc)
            if self.callback:
                self.callback()

        if is_corpus:
            input_data.store_tokens(pp_tokens)
            return input_data
        return pp_tokens

    def _get_corpus_specific_stopwords(self, corpus_tokens, lowercase):
        # Calculates the corpus specific stop words.
        document_frequencies = {}
        # 1.) Calculate the document frequency for individual tokens.
        for document in corpus_tokens:
            for token in set(document):
                if lowercase:
                    token = token.lower()
                document_frequencies[token] = document_frequencies.get(
                        token, 0
                ) + 1

        # 2.) SW based on minimum document frequency.
        if self.min_df is not None:
            min_df = self.min_df  # If int.
            if isinstance(self.min_df, float):  # If float.
                min_df = round(len(corpus_tokens) * self.min_df)
        else:
            min_df = 0

        # 3.) SW based on maximum document frequency.
        if self.max_df is not None:
            max_df = self.max_df  # If int.
            if isinstance(self.max_df, float):  # If float.
                max_df = round(len(corpus_tokens) * self.max_df)
        else:
            max_df = max(document_frequencies.values())

        self.df_stop_words = {
            token for token, freq
            in document_frequencies.items()
            if freq < min_df or freq > max_df
        }

    def _check_df_constraints(self, min_df, max_df):
        # Checks for appropriate data types and parameter ranges.
        if min_df is not None:
            self.use_df_sw = True
            is_float = isinstance(min_df, float)
            is_int = isinstance(min_df, int)
            if is_float:
                in_range = 0.0 <= min_df <= 1.0
                if not in_range:
                    raise ValueError('Parameter min_df is out of range ({}).'
                                     .format(min_df))
            elif not is_int:
                raise ValueError('Type "{}" not supported.'.format(min_df))

        if max_df is not None:
            self.use_df_sw = True
            is_float = isinstance(max_df, float)
            is_int = isinstance(max_df, int)
            if is_float:
                in_range = 0.0 <= max_df <= 1.0
                if not in_range:
                    raise ValueError('Parameter max_df is out of range ({}).'
                                     .format(max_df))
            elif not is_int:
                raise ValueError('Type "{}" not supported.'.format(max_df))


class Stemmatizer:
    """A common class for stemming and lemmatization."""
    def __init__(self, transformation, name='Stemmatizer'):
        """
        Args:
            transformation (Callable (function or method)): The function that
                will perform transformation on the tokens.
            name (Optional[str]): The name of the transformation object.

        Returns:
            `orangecontrib.text.preprocess.Stemmatizer`: The instance of a
                Stemmatizer of choice.
        """
        if not callable(transformation):
            raise ValueError('The transformation must be callable.')

        self.transformation = transformation
        self.name = name

    def __str__(self):
        return self.name

    def __call__(self, data):
        """Calls the transformation function of this instance.

        Args:
            data (str, list): The input that we wish to transform.

        Returns:
            str, list: The transformed input in the form of a string or list,
                depending on the input.

        Raises:
            ValueError: If the input type is not supported.
        """
        if isinstance(data, str):
            return self.transformation(data)
        elif isinstance(data, list):
            return [self.transformation(word) for word in data]
        else:
            raise ValueError('Type {} not supported.'.format(type(data)))

PorterStemmer = Stemmatizer(PorterStemmer().stem, name='PorterStemmer')
SnowballStemmer = Stemmatizer(
        SnowballStemmer(language='english').stem,
        name='SnowballStemmer'
)
Lemmatizer = Stemmatizer(WordNetLemmatizer().lemmatize, name='Lemmatizer')
