from PyQt4.QtGui import QApplication, QGridLayout, QLabel

from Orange.widgets import settings
from orangecontrib.text.vectorization import TfidfVectorizer
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.widgets.utils import widgets
from orangecontrib.text.widgets.utils import owbasevectorizer


class OWTfIdf(owbasevectorizer.OWBaseVectorizer):
    name = 'TF-IDF'
    description = 'Creates a TF-IDF matrix from the input corpus.'
    icon = 'icons/Tfidf.svg'
    priority = 42

    Method = TfidfVectorizer

    # Settings
    wlocal = settings.Setting(TfidfVectorizer.COUNT)
    wglobal = settings.Setting(TfidfVectorizer.NONE)
    normalization = settings.Setting(TfidfVectorizer.NONE)

    def create_configuration_layout(self):
        layout = QGridLayout()
        layout.setSpacing(10)
        row = 0
        combo = widgets.ComboBox(self, 'wlocal',
                                 items=tuple(TfidfVectorizer.wlocals.keys()))
        combo.currentIndexChanged.connect(self.on_change)
        layout.addWidget(QLabel('Term Frequency:'))
        layout.addWidget(combo, row, 1)

        row += 1
        combo = widgets.ComboBox(self, 'wglobal',
                                 items=tuple(TfidfVectorizer.wglobals.keys()))

        combo.currentIndexChanged.connect(self.on_change)
        layout.addWidget(QLabel('Document Frequency:'))
        layout.addWidget(combo, row, 1)

        row += 1
        combo = widgets.ComboBox(self, 'normalization',
                                 items=tuple(TfidfVectorizer.norms.keys()))

        combo.currentIndexChanged.connect(self.on_change)
        layout.addWidget(QLabel('Regularization:'))
        layout.addWidget(combo, row, 1)

        return layout

    def update_method(self):
        self.method = self.Method(norm=self.normalization,
                                  wlocal=self.wlocal,
                                  wglobal=self.wglobal)


if __name__ == '__main__':
    app = QApplication([])
    widget = OWTfIdf()
    widget.show()
    corpus = Corpus.from_file('bookexcerpts')
    widget.set_data(corpus)
    app.exec()
