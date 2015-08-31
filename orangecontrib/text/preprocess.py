from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

from orangecontrib.text.corpus import Corpus

import Orange


class Preprocessor:
    """
        A pre-processing class for the Orange text mining extension. An instance of the class
        is capable of tokenizing, lowercasing, stemming and lemmatizing the input. Returns a list
        of preprocessed tokens, sorted by order of appearance in text.
    """
    def __init__(self, tokenizer="word_tokenizer", lowercase=True, stop_words=None, transformation=None):
        """
        :param tokenization: A mandatory argument that defines the type of tokenizer we wish to use.
        :type tokenization: 'word_tokenizer' or 'twitter_tokenizer'
        :param lowercase: If set, transform the tokens to lower case, before returning them.
        :type lowercase: boolean
        :param stop_words: Determines whether and what stop words should be removed. Can remove
            default nltk stop words for the English language or stop words provided in a custom list.
        :type stop_words: 'default', a list or None
        :param trans: Name of the morphological transformation method to be performed on the tokens.
        :type trans: 'porter_stemmer', 'snowball_stemmer', 'lemmatizer' or None
        :return: list
        """
        # Tokenizer.
        if tokenizer is None:
            raise ValueError("The pre-processor must specify a tokenizer "
                             "('word_tokenizer' or 'twitter_tokenizer').")
        elif tokenizer == "word_tokenizer":
            self.tokenizer = word_tokenize
        elif tokenizer == "twitter_tokenizer":
            self.tokenizer = word_tokenize
        else:
            raise ValueError("Tokenizer of type '{}', is not recognized.".format(tokenizer))

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
        self.transformation = transformation
        if transformation == 'porter_stemmer':
            self.transformation = PorterStemmer
        elif transformation == 'snowball_stemmer':
            self.transformation = SnowballStemmer
        elif transformation == 'lemmatizer':
            self.transformation = Lemmatizer
        elif transformation is not None:    # If not None and not supported.
            raise ValueError("Transformation type '{}', is not recognized.".format(transformation))

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
        data = self.tokenize(data)
        # Remove stop words.
        if self.stop_words is not None:
            data = self.remove_stop_words(data)
        # Lowercase.
        if self.lowercase:
            data = [token.lower() for token in data]
        # Transform.
        if self.transformation is not None:
            data = self.transform(data)
        return data

    def tokenize(self, data):
        """
        Splits the text of the input into tokens by whitespaces and punctuation.
        :param data: The input holding the text to be tokenized. Can be provided
            in form of a string or list of strings.
        :type data: string, list
        :return: list
        """
        if isinstance(data, str):
            return self.tokenizer(data)
        elif isinstance(data, Orange.data.Table):
            # TODO: Handle this input.
            return None
        else:
            raise ValueError("Cannot tokenize. Type {} not supported.".format(type(data)))

    def transform(self, data):
        """
        Performs on the input, whatever transformation was specified at class instance creation.
        :param data: The input holding the text to be transformed. Can be provided
            in form of a string or list of string tokens.
        :type data: string, list
        :return: string, list
        """
        if self.transformation is None:
            raise ValueError("Cannot perform transformation, none was specified.")

        return self.transformation(data)

    def remove_stop_words(self, data):
        """
        Removes the stop words specified at class instance creation, from the list of tokens.
        :param data: The list of string tokens, from where the stop words are to be removed.
        :type data: list
        :return: list
        """
        if self.stop_words is None:
            raise ValueError("Cannot remove stop word, no set of such was specified.")

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
            raise ValueError("Cannot transform. Type {} not supported.".format(type(data)))

PorterStemmer = Stemmatizer(PorterStemmer().stem, 'PorterStemmer')
SnowballStemmer = Stemmatizer(SnowballStemmer(language="english").stem, 'SnowballStemmer')
Lemmatizer = Stemmatizer(WordNetLemmatizer().lemmatize, 'Lemmatizer')
