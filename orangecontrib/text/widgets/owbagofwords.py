from AnyQt.QtWidgets import QApplication, QGridLayout, QLabel

from Orange.widgets import settings
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.vectorization import BowVectorizer
from orangecontrib.text.widgets.utils import owbasevectorizer, widgets


class OWTBagOfWords(owbasevectorizer.OWBaseVectorizer):
    name = "Bag of Words"
    description = 'Generates a bag of words from the input corpus.'
    icon = 'icons/BagOfWords.svg'
    priority = 300
    keywords = "bag of words, bow"

    Method = BowVectorizer

    tfoptions = list(BowVectorizer.wlocals)
    dfoptions = list(BowVectorizer.wglobals)
    roptions = list(BowVectorizer.norms)

    tflabels = ['Count', 'Binary', 'Sublinear']
    dflabels = ['(None)', 'IDF', 'Smooth IDF']
    rlabels = ['(None)', 'L1 (Sum of elements)', 'L2 (Euclidean)']

    # Settings
    wlocal = settings.Setting(BowVectorizer.COUNT)
    wglobal = settings.Setting(BowVectorizer.NONE)
    normalization = settings.Setting(BowVectorizer.NONE)

    def create_configuration_layout(self):
        layout = QGridLayout()
        layout.setSpacing(10)
        row = 0
        combo = widgets.ComboBox(self, 'wlocal',
                                 items=tuple(zip(self.tflabels, self.tfoptions)))
        combo.currentIndexChanged.connect(self.on_change)
        layout.addWidget(QLabel('Term Frequency:'))
        layout.addWidget(combo, row, 1)

        row += 1
        combo = widgets.ComboBox(self, 'wglobal',
                                 items=tuple(zip(self.dflabels, self.dfoptions)))

        combo.currentIndexChanged.connect(self.on_change)
        layout.addWidget(QLabel('Document Frequency:'))
        layout.addWidget(combo, row, 1)

        row += 1
        combo = widgets.ComboBox(self, 'normalization',
                                 items=tuple(zip(self.rlabels, self.roptions)))

        combo.currentIndexChanged.connect(self.on_change)
        layout.addWidget(QLabel('Regularization:'))
        layout.addWidget(combo, row, 1)

        return layout

    def init_method(self):
        return self.Method(
            norm=self.normalization, wlocal=self.wlocal, wglobal=self.wglobal
        )

    def send_report(self):
        self.report_items((
            ('Term Frequency', self.tflabels[self.tfoptions.index(self.wlocal)]),
            ('Document Frequency', self.dflabels[self.dfoptions.index(self.wglobal)]),
            ('Regularization', self.rlabels[self.roptions.index(self.normalization)]),
        ))

if __name__ == '__main__':
    from orangewidget.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWTBagOfWords).run(Corpus.from_file("book-excerpts"))
