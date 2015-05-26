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
        self.trans_name = None
        self.transformation = None

        if trans == "Stemmer":
            self.trans_name = trans
            self.transformation = Stemmer
        elif trans == "Lemmatizer":
            self.trans_name = trans
            self.transformation = Lemmatizer

        sw = None
        if stop_words:
            sw = "english"

        self.pp_info = {"incl_punct": incl_punct, "lowercase": lowercase,
                        "stop_words": sw, "transformation": self}

    def __call__(self, data):
        if isinstance(data, str):
            output = data
            if self.pp_info["lowercase"]:
                output = output.lower()
            if self.transformation:
                output = self.transformation(output)
            return output
        elif isinstance(data, list):
            output = data
            if self.pp_info["lowercase"] and self.transformation:
                output = [self.transformation(word.lower()) for word in output]
            elif self.transformation:
                output = [self.transformation(word) for word in output]
            elif self.pp_info["lowercase"]:
                output = [word.lower() for word in output]
            return output
        else:
            raise ValueError("Type {} not supported.".format(type(data)))

class Stemmatizer():
    """
        A common class for stemming and lemmatization.
    """
    def __init__(self, trans):
        """
            :param trans: The method that will perform transformation on the tokens.
            :return: :class: `orangecontrib.text.preprocess.Stemmatizer`
        """
        self.trans = trans

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

Stemmer = Stemmatizer(PorterStemmer().stem)
Lemmatizer = Stemmatizer(WordNetLemmatizer().lemmatize)