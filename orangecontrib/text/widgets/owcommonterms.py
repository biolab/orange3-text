from Orange.misc import DistMatrix
from Orange.widgets import gui
from Orange.widgets.utils.signals import Input, Output
from Orange.widgets.widget import OWWidget, Msg
from PyQt5.QtWidgets import QApplication
from orangecontrib.text import Corpus
from orangecontrib.text.common_terms import common_terms


class OWCommonTerms(OWWidget):
    name = "Common Terms"
    priority = 1000
    icon = "icons/File.svg"

    class Inputs:
        data = Input("Corpus", Corpus)

    class Outputs:
        distances = Output("Distances", DistMatrix)

    want_main_area = False

    class Error(OWWidget.Error):
        no_bow_features = Msg('No bag-of-words features!')

    def __init__(self):
        super().__init__()
        self.corpus = None

        info_box = gui.widgetBox(self.controlArea, 'Info')
        gui.label(info_box, self, 'None')

    @Inputs.data
    def set_data(self, data):
        self.Error.clear()
        if len(data.domain.attributes) == 0:
            self.Error.no_bow_features()
            self.clear()
            return
        if data and not isinstance(data, Corpus):
            data = Corpus.from_table(data.domain, data)
        self.corpus = data
        self.commit()

    def commit(self):
        output = common_terms(self.corpus.X, rows=self.corpus)
        self.Outputs.distances.send(output)


if __name__ == "__main__":
    from orangecontrib.text.vectorization import BowVectorizer

    app = QApplication([])
    widget = OWCommonTerms()
    widget.show()
    corpus = Corpus.from_file('deerwester')
    vect = BowVectorizer()
    result = vect.transform(corpus)
    widget.set_data(result)
    app.exec()
