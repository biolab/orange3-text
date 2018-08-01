import numpy as np
from nltk.corpus import opinion_lexicon
from nltk.sentiment import SentimentIntensityAnalyzer

from orangecontrib.text import Corpus
from orangecontrib.text.misc import wait_nltk_data
from orangecontrib.text.preprocess import WordPunctTokenizer
from orangecontrib.text.vectorization.base import SharedTransform, \
    VectorizationComputeValue
from orangecontrib.text.sentiment.opinion_lexicon_slo import opinion_lexicon_slo


class Liu_Hu_Sentiment:
    sentiments = ('sentiment',)
    name = 'Liu Hu'

    methods = {'English': opinion_lexicon,
               'Slovenian': opinion_lexicon_slo}

    @wait_nltk_data
    def __init__(self, language):
        self.language = language
        self.positive = set(self.methods[language].positive())
        self.negative = set(self.methods[language].negative())

    def transform(self, corpus, copy=True):
        scores = []
        tokenizer = WordPunctTokenizer()
        tokens = tokenizer(corpus.documents)

        for doc in tokens:
            pos_words = sum(word in self.positive for word in doc)
            neg_words = sum(word in self.negative for word in doc)
            scores.append([100*(pos_words - neg_words)/max(len(doc), 1)])
        X = np.array(scores).reshape((-1, len(self.sentiments)))

        # set  compute values
        shared_cv = SharedTransform(self)
        cv = [VectorizationComputeValue(shared_cv, col)
              for col in self.sentiments]

        if copy:
            corpus = corpus.copy()
        corpus.extend_attributes(X, self.sentiments, compute_values=cv)
        return corpus


class Vader_Sentiment:
    sentiments = ('pos', 'neg', 'neu', 'compound')
    name = 'Vader'

    @wait_nltk_data
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()

    def transform(self, corpus, copy=True):
        scores = []
        for text in corpus.documents:
            pol_sc = self.vader.polarity_scores(text)
            scores.append([pol_sc[x] for x in self.sentiments])
        X = np.array(scores).reshape((-1, len(self.sentiments)))

        # set  compute values
        shared_cv = SharedTransform(self)
        cv = [VectorizationComputeValue(shared_cv, col)
              for col in self.sentiments]

        if copy:
            corpus = corpus.copy()
        corpus.extend_attributes(X, self.sentiments, compute_values=cv)
        return corpus


if __name__ == "__main__":
    corpus = Corpus.from_file('deerwester')
    liu = Liu_Hu_Sentiment('Slovenian')
    corpus2 = liu.transform(corpus[:5])
