from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from orangecontrib.text.corpus import Corpus


class Preprocessor():
    """
        A class holding the implementation of a text tokenizer.
        The input text may be tokenized by multiple delimiters,
        the default being whitespace. Can also recognize punctuation
        as tokens if specified so by the 'incl_punct' parameter. Several
        transformation objects and flags can be set, including whether the
        input should be stemmed, lemmatized, lowercased etc.
    """
    def __init__(self, incl_punct=True, trans=None, low_case=True, stop_wrds=None):
        """
        :param incl_punct: Determines whether the tokenizer should include punctuation as valid tokens.
        :type incl_punct: boolean
        :param trans: An optional pre-processor object to perform the corresponding
            tranformation on the tokens before returning them.
        :type trans: :class: `orangecontrib.text.preprocess.Lemmatizer`
            or :class: `orangecontrib.text.preprocess.Stemmer`
        :param low_case: If set, transform the tokens to lower case, before returning the list.
        :type low_case: boolean
        :param stop_wrds: Determines what stop words should be removed. If "english", remove default
            english stop words, to remove custom stopwords provide them in a list.
        :type stop_wrds: string or list
        :return: :class: `orangecontrib.text.preprocess.Preprocessor`
        """
        self.incl_punct = incl_punct
        self.trans = trans
        self.low_case = low_case
        self.stop_wrds = stop_wrds

    def __call__(self, data):
        """
            :param data: The input that we wish to tokenize.
            :type data: :class: `orangecontrib.text.corpus.Corpus`
            :return: list or :class: `Orange.data.Table`
        """
        if not isinstance(data, Corpus):
            raise ValueError("Type {} not supported.".format(type(data)))

        for doc in data.documents:
            if self.incl_punct:
                tokens = word_tokenize(doc.text)
            else:
                tokenizer = RegexpTokenizer(r'\w+')
                tokens = tokenizer.tokenize(doc.text)

            # Perform transformations.
            if self.trans:
                tokens = self.trans(tokens)

            # Lowercase.
            if self.low_case:
                tokens = [t.lower() for t in tokens]

            # Stop-word removal.
            if self.stop_wrds is not None:
                if self.stop_wrds == 'english':
                    sw = set(stopwords.words('english'))
                else:
                    sw = self.stop_wrds
                tokens = [w for w in tokens if w not in sw]
            doc.tokens = tokens
        return data


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
