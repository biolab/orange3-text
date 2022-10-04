from collections import namedtuple
import numpy as np

from Orange.widgets.widget import OWWidget
from Orange.widgets.gui import BarRatioTableModel
from Orange.data import Domain, StringVariable, ContinuousVariable, Table
from AnyQt.QtCore import Qt, pyqtSignal as Signal
from AnyQt.QtWidgets import QTableView, QItemDelegate

from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures
from nltk import BigramCollocationFinder, TrigramCollocationFinder

from orangecontrib.text import Corpus
from orangewidget import settings, gui
from orangewidget.utils.signals import Input, Output
from orangewidget.utils.widgetpreview import WidgetPreview

NGRAM_TYPES = [BigramCollocationFinder, TrigramCollocationFinder]

ScoreMeta = namedtuple("score_meta", ["name", "scorer"])

bi_measures = BigramAssocMeasures()
tri_measures = TrigramAssocMeasures()

SCORING_METHODS = [
    ScoreMeta("Pointwise Mutual Information", [bi_measures.pmi,
                                               tri_measures.pmi]),
    ScoreMeta("Chi Square", [bi_measures.chi_sq, tri_measures.chi_sq]),
    ScoreMeta("Dice", [bi_measures.dice]),
    ScoreMeta("Fisher", [bi_measures.fisher]),
    ScoreMeta("Jaccard", [bi_measures.jaccard, tri_measures.jaccard]),
    ScoreMeta("Likelihood ratio", [bi_measures.likelihood_ratio,
                                   tri_measures.likelihood_ratio]),
    ScoreMeta("Mi Like", [bi_measures.mi_like, tri_measures.mi_like]),
    ScoreMeta("Phi Square", [bi_measures.phi_sq]),
    ScoreMeta("Poisson Stirling", [bi_measures.poisson_stirling,
                                   tri_measures.poisson_stirling]),
    ScoreMeta("Raw Frequency", [bi_measures.raw_freq, tri_measures.raw_freq]),
    ScoreMeta("Student's T", [bi_measures.student_t, tri_measures.student_t])
]

VARNAME_COL, NVAL_COL = range(2)


class TableView(QTableView):
    manualSelection = Signal()

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent=parent,
                         selectionBehavior=QTableView.SelectRows,
                         selectionMode=QTableView.ExtendedSelection,
                         sortingEnabled=True,
                         showGrid=True,
                         cornerButtonEnabled=False,
                         alternatingRowColors=False,
                         **kwargs)
        # setItemDelegate(ForColumn) doesn't take ownership of delegates
        self._bar_delegate = gui.ColoredBarItemDelegate(self)
        self._del0, self._del1 = QItemDelegate(), QItemDelegate()
        self.setItemDelegate(self._bar_delegate)
        self.setItemDelegateForColumn(VARNAME_COL, self._del0)
        self.setItemDelegateForColumn(NVAL_COL, self._del1)

        header = self.horizontalHeader()
        header.setSectionResizeMode(header.Fixed)
        header.setFixedHeight(24)
        header.setDefaultSectionSize(80)
        header.setTextElideMode(Qt.ElideMiddle)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.manualSelection.emit()


class OWCollocations(OWWidget):
    name = "Collocations"
    description = "Compute significant bigrams and trigrams."
    keywords = ["PMI"]
    icon = "icons/Collocations.svg"

    class Inputs:
        corpus = Input("Corpus", Corpus)

    class Outputs:
        corpus = Output("Table", Table)

    want_main_area = True

    # settings
    type_index = settings.Setting(0)
    selected_method = settings.Setting(0)
    freq_filter = settings.Setting(1)
    auto_apply = settings.Setting(True)

    def __init__(self) -> None:
        OWWidget.__init__(self)
        self.corpus = None
        self.type = NGRAM_TYPES[self.type_index]
        self.method = None
        self.results = None

        setting_box = gui.vBox(self.controlArea, box="Settings")
        gui.radioButtons(setting_box, self, "type_index",
                         btnLabels=["Bigrams", "Trigrams"],
                         orientation=Qt.Horizontal,
                         callback=self._change_type)

        gui.spin(setting_box, self, "freq_filter", minv=1, maxv=1000, step=1,
                 label="Frequency", callback=self.commit)

        method_box = gui.vBox(self.controlArea, box="Scoring Method")
        self.method_rb = gui.radioButtons(method_box, self, "selected_method",
                                          btnLabels=[m.name for m in
                                                     SCORING_METHODS],
                                          callback=self.commit)

        gui.rubber(self.controlArea)

        gui.button(self.buttonsArea, self, "Restore Original Order",
                   callback=self.restore_order,
                   tooltip="Show rows in the original order",
                   autoDefault=False)

        # GUI
        self.collModel = model = BarRatioTableModel(parent=self)  # type:
        # TableModel
        model.setHorizontalHeaderLabels(["Method", "Score"])
        self.collView = view = TableView(self)  # type: TableView
        self.mainArea.layout().addWidget(view)
        view.setModel(model)
        view.resizeColumnsToContents()
        view.setItemDelegateForColumn(1, gui.ColoredBarItemDelegate())
        view.setSelectionMode(QTableView.NoSelection)

    @Inputs.corpus
    def set_corpus(self, corpus):
        self.collModel.clear()
        self.collModel.resetSorting(True)
        self.corpus = corpus
        self.commit()

    def _change_type(self):
        self.type = NGRAM_TYPES[self.type_index]
        if self.type_index == 1:
            self.method_rb.buttons[2].setDisabled(True)
            self.method_rb.buttons[3].setDisabled(True)
            self.method_rb.buttons[7].setDisabled(True)
            if self.selected_method in [2, 3, 7]:
                self.method_rb.buttons[0].click()
        else:
            self.method_rb.buttons[2].setDisabled(False)
            self.method_rb.buttons[3].setDisabled(False)
            self.method_rb.buttons[7].setDisabled(False)
        self.commit()

    def compute_scores(self):
        self.collModel.clear()
        self.collModel.resetSorting(True)
        finder = self.type.from_documents(self.corpus.tokens)
        finder.apply_freq_filter(self.freq_filter)

        res = finder.score_ngrams(self.method.scorer[self.type_index])
        collocations = np.array([" ".join(col) for col, score in res],
                                dtype=object)[:, None]
        scores = np.array([score for col, score in res], dtype=float)[:, None]

        self.results = (collocations, scores)
        if len(scores):
            self.collModel.setExtremesFrom(1, scores)

    def commit(self):
        if self.corpus is None:
            return

        self.type = NGRAM_TYPES[self.type_index]
        self.method = SCORING_METHODS[self.selected_method]

        self.compute_scores()

        if not self.results:
            self.collModel.clear()
            self.Outputs.corpus.send(None)
            return

        output = self.create_scores_table()
        self.collModel[:] = np.hstack(self.results)[:20]
        self.collView.resizeColumnsToContents()

        self.Outputs.corpus.send(output)

    def create_scores_table(self):
        domain = Domain([ContinuousVariable("Collocations")],
                        metas=[StringVariable("Scores")])

        collocations, scores = self.results

        new_table = Table.from_numpy(domain, scores, metas=collocations)
        new_table.name = "Collocation Scores"
        return new_table

    def restore_order(self):
        """Restore the original data order of the current view."""
        model = self.collModel
        self.collView.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)
        if model is not None:
            model.resetSorting(yes_reset=True)

    def send_report(self):
        view = self.collView
        if self.results:
            self.report_items("Collocations", (
                ("N-grams", ["Bigrams", "Trigrams"][self.type_index]),
                ("Method", self.method.name),
                ("Frequency", self.freq_filter)
            ))
        self.report_table(view)


if __name__ == "__main__":  # pragma: no cover
    previewer = WidgetPreview(OWCollocations)
    previewer.run(Corpus.from_file("deerwester.tab"), no_exit=True)
