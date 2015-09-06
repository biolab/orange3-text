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
    def __init__(self, lowercase=True, stop_words=None, transformation=None, use_twitter_tokenizer=False):
        """
        :param lowercase: If set, transform the tokens to lower case, before returning them.
        :type lowercase: boolean
        :param stop_words: Determines whether and what stop words should be removed. Can remove
            default nltk stop words for the English language or stop words provided in a custom list.
        :type stop_words: 'default', a list or None
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
        if stop_words == "default":
            self.stop_words = stopwords.words("english")
        elif isinstance(stop_words, list):
            self.stop_words = stop_words
        elif stop_words is None:
            self.stop_words = None
        else:
            raise ValueError("The stop words parameter should be either \'default\', "
                             "a list containing the stop words or None.")

        # Transformation.
        if not isinstance(transformation, Stemmatizer):
            raise ValueError("Type '{}' not supported.".format(transformation))
        self.transformation = transformation

    def __call__(self, data):
        if isinstance(data, Corpus):
            data = data.documents

        if isinstance(data, str):
            return self.preprocess_document(data)
        elif isinstance(data, list):
            return [self.preprocess_document(doc) for doc in data]
        else:
            raise ValueError("Type '{}' not supported.".format(type(data)))

    def preprocess_document(self, data):
        # Tokenize.
        data = self._tokenize(data)
        # Remove stop words.
        if self.stop_words is not None:
            data = self._remove_stop_words(data)
        # Lowercase.
        if self.lowercase:
            data = [token.lower() for token in data]
        # Transform.
        if self.transformation is not None:
            data = self._transform(data)
        return data

    def _tokenize(self, data):
        """
        Splits the text of the input into tokens by whitespaces and punctuation.
        :param data: The input holding the text to be tokenized. Can be provided
            in form of a string or list of strings.
        :type data: string, list
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

    def _remove_stop_words(self, data):
        """
        Removes the stop words specified at class instance creation, from the list of tokens.
        :param data: The list of string tokens, from where the stop words are to be removed.
        :type data: list
        :return: list
        """
        return [word for word in data if word not in self.stop_words]

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
