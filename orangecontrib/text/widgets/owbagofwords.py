from PyQt4.QtGui import QApplication

from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget
from orangecontrib.text.bagofowords import BagOfWords
from orangecontrib.text.corpus import Corpus


class Input:
    CORPUS = 'Corpus'


class Output:
    CORPUS = 'Corpus'


class OWBagOfWords(OWWidget):
    name = 'Bag of Words'
    description = 'Generates a bag of words from the input corpus.'
    icon = 'icons/BagOfWords.svg'
    priority = 40

    # Input/output
    inputs = [
        (Input.CORPUS, Corpus, 'set_data'),
    ]
    outputs = [
        (Output.CORPUS, Corpus)
    ]

    want_main_area = False

    # Settings
    use_tfidf = Setting(False)

    def __init__(self):
        super().__init__()

        self.corpus = None

        # TF-IDF.
        tfidf_box = gui.widgetBox(
                self.controlArea,
                'TF-IDF',
                addSpace=False
        )

        self.tfidf_chbox = gui.checkBox(
                tfidf_box,
                self,
                'use_tfidf',
                'Use TF-IDF'
        )
        self.tfidf_chbox.stateChanged.connect(self._tfidf_changed)

        gui.button(
                self.controlArea,
                self,
                '&Apply',
                callback=self.apply,
                default=True
        )

    def set_data(self, data):
        self.corpus = data
        if self.corpus is not None:
            self.apply()

    def apply(self):
        if self.corpus is not None:
            bag_of_words = BagOfWords()
            self.corpus = bag_of_words(self.corpus)
            self.send(Output.CORPUS, self.corpus)

    def _tfidf_changed(self):
        pass

if __name__ == '__main__':
    app = QApplication([])
    widget = OWBagOfWords()
    widget.show()
    corpus = Corpus.from_file('bookexcerpts')
    widget.set_data(corpus)
    app.exec()
