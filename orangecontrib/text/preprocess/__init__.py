from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


class Preprocessor():
    """
        Holds pre-processing flags and other information, about stop word
        removal, lowercasing, text morphing etc.(the options are set via
        the Preprocess widget).
    """
    def __init__(self, incl_punct=True, lowercase=True, stop_words=None, trans=None):
        """
        :param incl_punct: Determines whether the tokenizer should include punctuation in the tokens.
        :type incl_punct: boolean
        :param lowercase: If set, transform the tokens to lower case, before returning them.
        :type lowercase: boolean
        :param stop_words: Determines whether stop words should("english"), or should not(None) be removed.
        :type stop_words: string
        :param trans: An optional pre-processor object to perform the morphological
            transformation on the tokens before returning them.
        :type trans: :class: `orangecontrib.text.preprocess.Lemmatizer`
            or :class: `orangecontrib.text.preprocess.Stemmer`
        :return: :class: `orangecontrib.text.preprocess.Preprocessor`
        """
        # TODO Needs more elaborate check on list contents.
        if stop_words != 'english' or isinstance(stop_words, list):
            raise ValueError("The stop words parameter should be either \'english\' or a list.")
        self.stop_words = stop_words
        self.incl_punct = incl_punct
        self.lowercase = lowercase
        self.transformation = trans


class Stemmatizer():
    """
        A common class for stemming and lemmatization.
    """
    def __init__(self, trans, name='Stemmatizer'):
        """
            :param trans: The method that will perform transformation on the tokens.
            :param name: The name of the transformation object.
            :return: :class: `orangecontrib.text.preprocess.Stemmatizer`
        """
        self.trans = trans
        self.name = name

    def __call__(self, data):
        """
            :param data: The input that we wish to transform.
            :type data: string, list or :class: `Orange.data.Table`
            :return: string, list or :class: `Orange.data.Table`
        """
        if isinstance(data, str):
            return self.trans(data)
        elif isinstance(data, list):
            return [self.trans(word) for word in data]
        else:
            raise ValueError("Type {} not supported.".format(type(data)))

Stemmer = Stemmatizer(PorterStemmer().stem, 'Stemmer')
Lemmatizer = Stemmatizer(WordNetLemmatizer().lemmatize, 'Lemmatizer')
