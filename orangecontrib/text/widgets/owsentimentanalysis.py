import numpy as np
from nltk.corpus import opinion_lexicon
from AnyQt.QtWidgets import QApplication

from Orange.data import Domain, ContinuousVariable
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget
from orangecontrib.text import Corpus
from orangecontrib.text.preprocess import WordPunctTokenizer


class OWSentimentAnalysis(OWWidget):
    name = "Sentiment Analysis"
    description = "Predict sentiment from text."
    icon = "icons/SentimentAnalysis"
    priority = 30001

    inputs = [("Corpus", Corpus, "set_corpus")]
    outputs = [("Corpus", Corpus, )]

    autocommit = Setting(True)
    want_main_area = False

    def __init__(self):
        super().__init__()

        self.corpus = None
        self.tokens = None
        self.positive = set(opinion_lexicon.positive())
        self.negative = set(opinion_lexicon.negative())

        info_box = gui.widgetBox(self.controlArea)
        self.info_label = gui.label(info_box, self, 'n/a')
        gui.rubber(self.controlArea)

        gui.auto_commit(self.controlArea, self, 'autocommit', 'Commit',
                        'Autocommit is on')

    def set_corpus(self, data=None):
        self.corpus = data
        if data is not None and not isinstance(data, Corpus):
            self.corpus = Corpus.from_table(data.domain, data)
        self.set_tokens()
        self.update_info()
        self.commit()

    def set_tokens(self):
        if self.corpus is None:
            self.tokens = None
            return
        tokenizer = WordPunctTokenizer()
        self.tokens = tokenizer(self.corpus.documents)

    def update_info(self):
        self.info_label.setText('Documents: %s' % len(self.corpus) if
                                self.corpus is not None else 'n/a')

    def sentiment_lex(self, doc):

        pos_words = 0
        neg_words = 0

        for word in doc:
            if word in self.positive:
                pos_words += 1
            elif word in self.negative:
                neg_words += 1

        return (pos_words - neg_words) / len(doc) * 100

    def update_method(self):
        self.commit()

    def commit(self):
        if not self.corpus:
            self.send('Corpus', None)
            return
        domain = self.corpus.domain
        sentiment = [self.sentiment_lex(doc) for doc in self.tokens]
        sents = ContinuousVariable('Sentiment')
        new_domain = Domain(domain.attributes, sents,
                            domain.metas + domain.class_vars)
        new_corpus = self.corpus.transform(new_domain)
        new_corpus.Y = np.array(sentiment)
        self.send('Corpus', new_corpus)

if __name__ == '__main__':
    app = QApplication([])
    widget = OWSentimentAnalysis()
    corpus = Corpus.from_file('bookexcerpts')
    corpus = corpus[:3]
    widget.set_corpus(corpus)
    widget.show()
    app.exec()
