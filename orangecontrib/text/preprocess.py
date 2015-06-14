from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer


class Preprocessor:
    """
        Holds pre-processing flags and other information, about stop word
        removal, lowercasing, text morphing etc.(the options are set via
        the Preprocess widget).
    """
    def __init__(self, incl_punct=False, lowercase=True, stop_words='english', trans=None, min_df=1):
        """
        :param incl_punct: Determines whether the tokenizer should include punctuation in the tokens.
        :type incl_punct: boolean
        :param lowercase: If set, transform the tokens to lower case, before returning them.
        :type lowercase: boolean
        :param stop_words: Determines whether stop words should("english"), or should not(None) be removed.
            If this is list, it should contain stopwords.
        :type stop_words: 'english' or list or None
        :param trans: An optional pre-processor object to perform the morphological
            transformation on the tokens before returning them.
        :type trans: :class: `orangecontrib.text.preprocess.Lemmatizer`
            or :class: `orangecontrib.text.preprocess.Stemmer`
        :return: :class: `orangecontrib.text.preprocess.Preprocessor`
        """
        # TODO Needs more elaborate check on list contents.
        if not (stop_words == 'english' or isinstance(stop_words, list) or stop_words is None):
            raise ValueError("The stop words parameter should be either \'english\', a list or None.")
        if isinstance(trans, Stemmatizer):  # hack, since scikit does not obey lowercase when preprocesor given
            trans.lowercase = lowercase
        self.stop_words = stop_words
        # TODO does including punctuation even work?
        self.incl_punct = incl_punct
        self.lowercase = lowercase
        self.transformation = trans
        self.min_df = min_df
        self.cv = CountVectorizer(
            lowercase=lowercase,
            stop_words=stop_words,
            preprocessor=trans,
            min_df=min_df,
        )

    def __call__(self, data):
        if isinstance(data, str):
            vec = self.cv.fit([data])
            return vec.get_feature_names()
        if isinstance(data, list):
            vec = self.cv.fit(data)
            features = vec.get_feature_names()
            docs = [[] for _ in range(len(data))]
            for (line, column), count in vec.transform(data).todok().items():
                docs[line].extend([features[column]] * count)
            return docs
        else:
            raise ValueError("Type '{}' not supported.".format(type(data)))


class Stemmatizer():
    """
        A common class for stemming and lemmatization.
    """
    def __init__(self, trans, lowercase=True, name='Stemmatizer'):
        """
            :param trans: The method that will perform transformation on the tokens.
            :param name: The name of the transformation object.
            :return: :class: `orangecontrib.text.preprocess.Stemmatizer`
        """
        self.trans = trans
        self.name = name
        self.lowercase = lowercase

    def __call__(self, data):
        """
            :param data: The input that we wish to transform.
            :type data: string, list or :class: `Orange.data.Table`
            :return: string, list or :class: `Orange.data.Table`
        """
        if isinstance(data, str):
            if self.lowercase:
                return self.trans(data).lower()
            else:
                return self.trans(data)
        elif isinstance(data, list):
            if self.lowercase:
                return [self.trans(word).lower() for word in data]
            else:
                return [self.trans(word) for word in data]
        else:
            raise ValueError("Type {} not supported.".format(type(data)))

Stemmer = Stemmatizer(PorterStemmer().stem, 'Stemmer')
Lemmatizer = Stemmatizer(WordNetLemmatizer().lemmatize, 'Lemmatizer')
