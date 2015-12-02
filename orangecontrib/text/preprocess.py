from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

from orangecontrib.text.corpus import Corpus


class Preprocessor:
    """
        A pre-processing class for the Orange text mining extension is capable of tokenizing,
        lowercasing, stemming and lemmatizing the input. Returns a list of preprocessed tokens,
        sorted by order of appearance in text.
    """
    def __init__(self, lowercase=True, stop_words=None, min_df=None, max_df=None, transformation=None,
                 use_twitter_tokenizer=False):
        """
        :param lowercase: If set, transform the tokens to lower case, before returning them.
        :type lowercase: boolean
        :param stop_words: Determines whether and what stop words should be removed. Can remove
            default nltk stop words for the English language or stop words provided in a custom list.
        :type stop_words: 'english', a list or None
        :param min_df: Tokens that appear in exactly or less than 'min_df' documents, will be removed.
            Can be specified either as an actual count of documents or a proportion of the corpus. Assigning
            this parameter when pre-processing a single document, will have no effect.
        :type min_df: int or float
        :param max_df: Tokens that appear in exactly or more than 'max_df' documents, will be removed.
            Can be specified either as an actual count of documents or a proportion of the corpus. Assigning
            this parameter when pre-processing a single document, will have no effect.
        :type max_df: int or float
        :param transformation: Name of the morphological transformation method to be performed on the tokens.
        :type transformation: `orangecontrib.text.preprocess.Stemmatizer`
        :param use_twitter_tokenizer: Determines the use of either the Twitter or default word tokenizer.
        :type use_twitter_tokenizer: bool
        :return: list
        """
        # Tokenizer.
        self.tokenizer = word_tokenize
        if use_twitter_tokenizer:
            self.tokenizer = None  # TODO: Change when twitter tokenizer is available.

        # Lowercase.
        self.lowercase = lowercase

        # Stop words.
        if stop_words == "english":
            self.stop_words = stopwords.words("english")
        elif isinstance(stop_words, list):
            self.stop_words = stop_words
        elif stop_words is None:
            self.stop_words = None
        else:
            raise ValueError('The stop words parameter should be either "english\", '
                             'a list containing the stop words or None.')

        # Min/Max df.
        # Constant checks whether or not we should use 'df', are quite painful.
        self.use_df_sw = False  # This flag summarizes this check.
        self._check_df_constraints(min_df, max_df)
        self.min_df = min_df
        self.max_df = max_df
        self.df_stop_words = None

        # Transformation.
        if transformation is not None and not isinstance(transformation, Stemmatizer):
            raise ValueError("Type '{}' not supported.".format(transformation))
        self.transformation = transformation

    def __call__(self, data):
        if isinstance(data, Corpus):
            data = data.documents

        if isinstance(data, str):
            pp_corpus = self._preprocess_document(data)
            return pp_corpus
        elif isinstance(data, list):
            if self.use_df_sw:  # DF parameters were specified.
                self._get_corpus_specific_stopwords(data)
            pp_corpus = [self._preprocess_document(doc) for doc in data]
            return pp_corpus
        else:
            raise ValueError("Type '{}' not supported.".format(type(data)))

    def _preprocess_document(self, data):
        # Tokenize.
        data = self._tokenize(data)
        # Lowercase.
        if self.lowercase:
            data = [token.lower() for token in data]
        # Remove stop words.
        if self.stop_words is not None:
            data = self._remove_stop_words(data, self.stop_words)
        if self.df_stop_words is not None:
            data = self._remove_stop_words(data, self.df_stop_words)
        # Transform.
        if self.transformation is not None:
            data = self._transform(data)
        return data

    def _tokenize(self, data):
        """
        Splits the text of the input into tokens by whitespaces and punctuation.
        :param data: The input holding the text to be tokenized.
        :type data: str
        :return: list
        """
        if isinstance(data, str):
            return self.tokenizer(data)
        else:
            raise ValueError("Type {} not supported.".format(type(data)))

    def _transform(self, data):
        """
        Performs on the input, whatever transformation was specified at class instance creation.
        :param data: The input holding the text to be transformed. Can be provided
            in form of a string or list of string tokens.
        :type data: string, list
        :return: string, list
        """
        return self.transformation(data)

    @staticmethod
    def _remove_stop_words(data, stop_words):
        """
        Removes the stop words from the corpus. Removes either the stop words specified
        at class instance creation, or stop words calculated from the min/max_df parameters.
        """
        return [word for word in data if word not in stop_words]

    def _get_corpus_specific_stopwords(self, corpus_documents):
        """
        Calculates the corpus specific stop words, using the minimum and maximum document
        frequency parameters.
        """
        # 1.) First, we need to tokenize the documents in the corpus.
        corpus_documents = [self._tokenize(doc) for doc in corpus_documents]

        document_frequencies = {}
        # 2.) Calculate the document frequency for individual tokens.
        for document in corpus_documents:
            for token in set(document):
                document_frequencies[token] = document_frequencies.get(token, 0) + 1

        # 3.) SW based on minimum document frequency.
        sw_min = []
        if self.min_df is not None:
            min_df = self.min_df  # If int.
            if isinstance(self.min_df, float):  # If float.
                min_df = round(len(corpus_documents) * self.min_df)
            sw_min = [token for token, freq in document_frequencies.items() if freq <= min_df]
            print ("")

        # 4.) SW based on maximum document frequency.
        sw_max = []
        if self.max_df is not None:
            max_df = self.max_df  # If int.
            if isinstance(self.max_df, float):  # If float.
                max_df = round(len(corpus_documents) * self.max_df)
            sw_max = [token for token, freq in document_frequencies.items() if freq >= max_df]

        self.df_stop_words = sw_min+sw_max

    def _check_df_constraints(self, min_df, max_df):
        """
        Checks for appropriate data types and parameter ranges.
        """
        if min_df is not None:
            self.use_df_sw = True
            is_float = isinstance(min_df, float)
            is_int = isinstance(min_df, int)
            if is_float:
                in_range = 1.0 >= min_df >= 0.0
                if not in_range:
                    raise ValueError('Parameter min_df is out of range ({}).'.format(min_df))
            elif not is_int:
                raise ValueError('Type "{}" not supported.'.format(min_df))

        if max_df is not None:
            self.use_df_sw = True
            is_float = isinstance(max_df, float)
            is_int = isinstance(max_df, int)
            if is_float:
                in_range = 1.0 >= max_df >= 0.0
                if not in_range:
                    raise ValueError('Parameter max_df is out of range ({}).'.format(max_df))
            elif not is_int:
                raise ValueError('Type "{}" not supported.'.format(max_df))


class Stemmatizer():
    """
        A common class for stemming and lemmatization.
    """
    def __init__(self, transformation, name='Stemmatizer'):
        """
            :param transformation: The method that will perform transformation on the tokens.
            :param name: The name of the transformation object.
            :return: :class: `orangecontrib.text.preprocess.Stemmatizer`
        """
        self.transformation = transformation
        self.name = name

    def __call__(self, data):
        """
            :param data: The input that we wish to transform.
            :type data: string, list
            :return: string, list
        """
        if isinstance(data, str):
            return self.transformation(str)
        elif isinstance(data, list):
            return [self.transformation(word) for word in data]
        else:
            raise ValueError("Type {} not supported.".format(type(data)))

PorterStemmer = Stemmatizer(PorterStemmer().stem, 'PorterStemmer')
SnowballStemmer = Stemmatizer(SnowballStemmer(language="english").stem, 'SnowballStemmer')
Lemmatizer = Stemmatizer(WordNetLemmatizer().lemmatize, 'Lemmatizer')
