from PyQt4 import QtCore

from Orange.widgets import settings
from Orange.widgets import gui
from Orange.widgets.widget import OWWidget
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.widgets.utils import widgets


class OWNgramRange(OWWidget):
    name = "Ngrams"
    description = "Transforms documents of a corpus to sets of token's ngrams."
    icon = "icons/Ngrams.svg"
    priority = 35

    # Input/output
    inputs = [("Corpus", Corpus, "set_data")]
    outputs = [("Corpus", Corpus)]

    want_main_area = False
    resizing_enabled = False
    buttons_area_orientation = QtCore.Qt.Horizontal

    ngram_range = settings.Setting((1, 1))
    autocommit = settings.Setting(True)

    def __init__(self):
        super().__init__()
        self.corpus = None

        box = gui.hBox(self.controlArea, 'Options')

        range_box = widgets.RangeWidget(self, 'ngram_range', minimum=1, maximum=10, step=1,
                                        min_label='N-grams range:', dtype=int,
                                        callback=self.on_change)
        box.layout().addWidget(range_box)

        gui.auto_commit(self.buttonsArea, self, 'autocommit', 'Commit', box=False)

    def set_data(self, data=None):
        self.corpus = data.copy()
        self.commit()

    def on_change(self):
        self.commit()

    def commit(self):
        self.apply()

    def apply(self):
        if self.corpus is not None:
            self.corpus.ngram_range = self.ngram_range

        self.send("Corpus", self.corpus)

    def send_report(self):
        self.report_items('Options', [('Ngrams range', self.ngram_range)])


if __name__ == '__main__':
    from PyQt4.QtGui import QApplication
    app = QApplication([])
    widget = OWNgramRange()
    widget.show()
    corpus = Corpus.from_file('bookexcerpts')
    widget.set_data(corpus)
    app.exec()
