import numpy as np

from nltk.corpus import opinion_lexicon
from nltk.sentiment import SentimentIntensityAnalyzer
from orangecontrib.text import Corpus
from orangecontrib.text.preprocess import WordPunctTokenizer
from orangecontrib.text.vectorization.bagofwords import BoWPreprocessTransform, BoWComputeValue


class Liu_Hu_Sentiment:
    positive = set(opinion_lexicon.positive())
    negative = set(opinion_lexicon.negative())
    sentiments = ('sentiment',)
    name = 'Liu Hu'

    def __init__(self):
        super().__init__()
        self.dic = {0: 'sentiment'}

    def transform(self, corpus):
        scores = []
        tokenizer = WordPunctTokenizer()
        tokens = tokenizer(corpus.documents)

        for doc in tokens:
            pos_words = sum(word in self.positive for word in doc)
            neg_words = sum(word in self.negative for word in doc)
            scores.append([100*(pos_words - neg_words)/max(len(doc), 1)])
        X = np.array(scores).reshape((-1, len(self.sentiments)))

        # set  compute values
        shared_cv = BoWPreprocessTransform(None, self, self.dic)
        cv = [BoWComputeValue(self.dic[i], shared_cv) for i in range(len(self.dic))]

        corpus = corpus.copy()
        corpus.extend_attributes(X, self.sentiments, compute_values=cv)
        return corpus


class Vader_Sentiment:
    sentiments = ('pos', 'neg', 'neu', 'compound')
    name = 'Vader'

    def __init__(self):
        super().__init__()
        self.vader = SentimentIntensityAnalyzer()
        self.dic = dict(enumerate(self.sentiments))

    def transform(self, corpus):
        scores = []
        for text in corpus.documents:
            pol_sc = self.vader.polarity_scores(text)
            scores.append([pol_sc[x] for x in self.sentiments])
        X = np.array(scores).reshape((-1, len(self.sentiments)))

        # set  compute values
        shared_cv = BoWPreprocessTransform(None, self, self.dic)
        cv = [BoWComputeValue(self.dic[i], shared_cv) for i in range(len(self.dic))]

        corpus = corpus.copy()
        corpus.extend_attributes(X, self.sentiments, compute_values=cv)
        return corpus


if __name__ == "__main__":
    corpus = Corpus.from_file('deerwester')
    liu = Liu_Hu_Sentiment()
    corpus2 = liu.transform(corpus[:5])
