from PyQt4.QtGui import QApplication, QVBoxLayout

from Orange.widgets import gui
from Orange.widgets import settings

from orangecontrib.text.vectorization import CountVectorizer
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.widgets.utils import owbasevectorizer


class OWBagOfWords(owbasevectorizer.OWBaseVectorizer):
    name = 'Bag of Words'
    description = 'Generates a bag of words from the input corpus.'
    icon = 'icons/BagOfWords.svg'
    priority = 40

    Method = CountVectorizer

    binary = settings.Setting(False)

    def create_configuration_layout(self):
        layout = QVBoxLayout()
        box = gui.checkBox(None, self, 'binary', 'Binary',
                           callback=self.on_change)
        layout.addWidget(box)
        return layout

    def update_method(self):
        self.method = self.Method(binary=self.binary)


if __name__ == '__main__':
    app = QApplication([])
    widget = OWBagOfWords()
    widget.show()
    corpus = Corpus.from_file('bookexcerpts')
    widget.set_data(corpus)
    app.exec()
