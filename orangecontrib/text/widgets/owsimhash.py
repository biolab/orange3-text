from AnyQt.QtWidgets import QApplication, QFormLayout

from Orange.widgets import gui
from Orange.widgets import settings
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.vectorization import SimhashVectorizer
from orangecontrib.text.widgets.utils import owbasevectorizer


class OWSimhash(owbasevectorizer.OWBaseVectorizer):
    name = 'Similarity Hashing'
    description = 'Computes documents hashes.'
    icon = 'icons/Simhash.svg'
    priority = 310
    keywords = ["SimHash"]

    Method = SimhashVectorizer

    f = settings.Setting(64)
    shingle_len = settings.Setting(10)

    def create_configuration_layout(self):
        layout = QFormLayout()

        spin = gui.spin(self, self, 'f', minv=1,
                        maxv=SimhashVectorizer.max_f)
        spin.editingFinished.connect(self.on_change)
        layout.addRow('Simhash size:', spin)

        spin = gui.spin(self, self, 'shingle_len', minv=1, maxv=100)
        spin.editingFinished.connect(self.on_change)
        layout.addRow('Shingle length:', spin)
        return layout

    def update_method(self):
        self.method = self.Method(shingle_len=self.shingle_len,
                                  f=self.f)


if __name__ == '__main__':
    app = QApplication([])
    widget = OWSimhash()
    widget.show()
    corpus = Corpus.from_file('book-excerpts')
    widget.set_data(corpus)
    app.exec()
