from typing import Optional

from itertools import chain
import numpy as np

from AnyQt.QtCore import Qt, QAbstractTableModel, QSize, QItemSelectionModel, \
    QItemSelection, QModelIndex
from AnyQt.QtWidgets import QSizePolicy, QApplication, QTableView, \
    QStyledItemDelegate
from AnyQt.QtGui import QColor
from Orange.data import Domain, StringVariable, Table

from Orange.widgets import gui
from Orange.widgets.settings import Setting, ContextSetting, PerfectDomainContextHandler
from Orange.widgets.widget import OWWidget, Msg, Input, Output
from nltk import ConcordanceIndex
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.topics import Topic
from orangecontrib.text.preprocess import WordPunctTokenizer


class HorizontalGridDelegate(QStyledItemDelegate):
    """Class for setting elide."""

    def paint(self, painter, option, index):
        if index.column() == 0:
            option.textElideMode = Qt.ElideLeft
        elif index.column() == 2:
            option.textElideMode = Qt.ElideRight
        QStyledItemDelegate.paint(self, painter, option, index)


class DocumentSelectionModel(QItemSelectionModel):
    """Sets selection for QTableView. Creates a set of selected documents."""

    def select(self, selection, flags):
        # which rows have been selected
        indexes = selection.indexes() if isinstance(selection, QItemSelection) \
                  else [selection]
        # prevent crashing when deleting the connection
        if not indexes:
            super().select(selection, flags)
            return
        # indexes[0].row() == -1 indicates clicking outside of the table
        if len(indexes) == 1 and indexes[0].row() == -1:
            self.clear()
            return
        word_index = self.model().word_index
        selected_docs = {word_index[index.row()][0] for index in indexes}
        selected_rows = [
            row_index for row_index, (doc_index, _) in enumerate(word_index)
            if doc_index in selected_docs]
        selection = QItemSelection()
        # select all rows belonging to the selected document
        for row in selected_rows:
            index = self.model().index(row, 0)
            selection.select(index, index)
        super().select(selection, flags)


class ConcordanceModel(QAbstractTableModel):
    """A model for constructing concordances from text."""

    def __init__(self):
        QAbstractTableModel.__init__(self)
        self.word = None
        self.corpus = None
        self.tokens = None
        self.n_tokens = None
        self.n_types = None
        self.indices = None
        self.word_index = None
        self.width = 8
        self.colored_rows = None

    def set_word(self, word):
        self.modelAboutToBeReset.emit()
        self.word = word
        self._compute_word_index()
        self.modelReset.emit()

    def set_corpus(self, corpus):
        self.modelAboutToBeReset.emit()
        self.corpus = corpus
        self.set_tokens()
        self._compute_indices()
        self._compute_word_index()
        self.modelReset.emit()

    def set_tokens(self):
        if self.corpus is None:
            self.tokens = None
            return
        tokenizer = WordPunctTokenizer()
        self.tokens = tokenizer(self.corpus.documents)
        self.n_tokens = sum(map(len, self.tokens))
        self.n_types = len(set(chain.from_iterable(self.tokens)))

    def set_width(self, width):
        self.modelAboutToBeReset.emit()
        self.width = width
        self.modelReset.emit()

    def flags(self, _):
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def rowCount(self, parent=QModelIndex(), *args, **kwargs):
        return 0 if parent.isValid() or self.word_index is None else len(self.word_index)

    def columnCount(self, parent=None, *args, **kwargs):
        return 3

    def data(self, index, role=Qt.DisplayRole):
        row, col = index.row(), index.column()
        doc, index = self.word_index[row]

        if role == Qt.DisplayRole:
            tokens = self.tokens
            if col == 0:
                return ' '.join(tokens[doc][max(index - self.width, 0):index])
            if col == 1:
                return tokens[doc][index]
            if col == 2:
                return ' '.join(tokens[doc][index + 1:index + self.width + 1])

        elif role == Qt.TextAlignmentRole:
            return [Qt.AlignRight | Qt.AlignVCenter,
                    Qt.AlignCenter,
                    Qt.AlignLeft | Qt.AlignVCenter][col]

        elif role == Qt.BackgroundRole:
            const = self.word_index[row][0] in self.colored_rows
            return QColor(236 + 19 * const, 243 + 12 * const, 255)

    def _compute_indices(self):  # type: () -> Optional[None, list]
        if self.corpus is None:
            self.indices = None
            return
        self.indices = [ConcordanceIndex(doc, key=lambda x: x.lower())
                        for doc in self.tokens]

    def _compute_word_index(self):
        if self.indices is None or self.word is None:
            self.word_index = self.colored_rows = None
        else:
            self.word_index = [
                (doc_idx, offset) for doc_idx, doc in enumerate(self.indices)
                for offset in doc.offsets(self.word)]
            self.colored_rows = set(sorted({d[0] for d in self.word_index})[::2])

    def matching_docs(self):
        if self.indices and self.word:
            return sum(bool(doc.offsets(self.word)) for doc in self.indices)
        else:
            return 0

    def get_data(self):
        domain = Domain([], metas=[StringVariable("Conc. {}".format(
            self.word)), StringVariable("Document")])
        data = []
        docs = []
        for row in range(self.rowCount()):
            txt = []
            for column in range(self.columnCount()):
                index = self.index(row, column)
                txt.append(str(self.data(index)))
            data.append([" ".join(txt)])
            docs.append([self.corpus.titles[self.word_index[row][0]]])
        conc = np.array(np.hstack((data, docs)), dtype=object)
        return Corpus(domain, metas=conc, text_features=[domain.metas[1]])


class OWConcordance(OWWidget):
    name = "Concordance"
    description = "Display the context of the word."
    icon = "icons/Concordance.svg"
    priority = 520

    class Inputs:
        corpus = Input("Corpus", Corpus)
        query_word = Input("Query Word", Topic)

    class Outputs:
        selected_documents = Output("Selected Documents", Corpus)
        concordances = Output("Concordances", Corpus)

    settingsHandler = PerfectDomainContextHandler(
        match_values = PerfectDomainContextHandler.MATCH_VALUES_ALL
    )
    autocommit = Setting(True)
    context_width = Setting(5)
    word = ContextSetting("", exclude_metas=False)
    selected_rows = Setting([], schema_only=True)

    class Warning(OWWidget.Warning):
        multiple_words_on_input = Msg("Multiple query words on input. "
                                      "Only the first one is considered!")

    def __init__(self):
        super().__init__()

        self.corpus = None      # Corpus
        self.n_matching = ''    # Info on docs matching the word
        self.n_tokens = ''      # Info on tokens
        self.n_types = ''       # Info on types (unique tokens)
        self.is_word_on_input = False

        # Info attributes
        info_box = gui.widgetBox(self.controlArea, 'Info')
        gui.label(info_box, self, 'Tokens: %(n_tokens)s')
        gui.label(info_box, self, 'Types: %(n_types)s')
        gui.label(info_box, self, 'Matching: %(n_matching)s')

        # Width parameter
        gui.spin(self.controlArea, self, 'context_width', 3, 10, box=True,
                 label="Number of words:", callback=self.set_width)

        gui.rubber(self.controlArea)

        # Search
        c_box = gui.widgetBox(self.mainArea, orientation="vertical")
        self.input = gui.lineEdit(
            c_box, self, 'word', orientation=Qt.Horizontal,
            sizePolicy=QSizePolicy(QSizePolicy.MinimumExpanding,
                                   QSizePolicy.Fixed),
            label='Query:', callback=self.set_word, callbackOnType=True)
        self.input.setFocus()

        # Concordances view
        self.conc_view = QTableView()
        self.model = ConcordanceModel()
        self.conc_view.setModel(self.model)
        self.conc_view.setWordWrap(False)
        self.conc_view.setSelectionBehavior(QTableView.SelectRows)
        self.conc_view.setSelectionModel(DocumentSelectionModel(self.model))
        self.conc_view.setItemDelegate(HorizontalGridDelegate())
        self.conc_view.selectionModel().selectionChanged.connect(self.selection_changed)
        self.conc_view.horizontalHeader().hide()
        self.conc_view.setShowGrid(False)
        self.mainArea.layout().addWidget(self.conc_view)
        self.set_width()

        # Auto-commit box
        gui.auto_commit(self.controlArea, self, 'autocommit', 'Commit',
                        'Auto commit is on')

    def sizeHint(self): # pragma: no cover
        return QSize(600, 400)

    def set_width(self):
        sel = self.conc_view.selectionModel().selection()
        self.model.set_width(self.context_width)
        if sel:
            self.conc_view.selectionModel().select(sel,
                QItemSelectionModel.SelectCurrent | QItemSelectionModel.Rows)

    def selection_changed(self):
        selection = self.conc_view.selectionModel().selection()
        self.selected_rows = sorted(set(cell.row() for cell in selection.indexes()))
        self.commit()

    def set_selection(self, selection):
        if selection:
            sel = QItemSelection()
            for row in selection:
                index = self.conc_view.model().index(row, 0)
                sel.select(index, index)
            self.conc_view.selectionModel().select(sel,
                QItemSelectionModel.SelectCurrent | QItemSelectionModel.Rows)

    @Inputs.corpus
    def set_corpus(self, data=None):
        self.closeContext()
        self.corpus = data
        if data is None:    # data removed, clear selection
            self.selected_rows = []

        if not self.is_word_on_input:
            self.word = ""
            self.openContext(self.corpus)

        self.model.set_corpus(self.corpus)
        self.set_word()

    @Inputs.query_word
    def set_word_from_input(self, topic):
        self.Warning.multiple_words_on_input.clear()
        if self.is_word_on_input:   # word changed, clear selection
            self.selected_rows = []
        self.is_word_on_input = topic is not None and len(topic) > 0
        self.input.setEnabled(not self.is_word_on_input)
        if self.is_word_on_input:
            if len(topic) > 1:
                self.Warning.multiple_words_on_input()
            self.word = topic.metas[0, 0]
            self.set_word()

    def set_word(self):
        self.selected_rows = []
        self.model.set_word(self.word)
        self.update_widget()
        self.commit()

    def handleNewSignals(self):
        self.set_selection(self.selected_rows)

    def resize_columns(self):
        col_width = (self.conc_view.width() -
                     self.conc_view.columnWidth(1)) / 2 - 12
        self.conc_view.setColumnWidth(0, col_width)
        self.conc_view.setColumnWidth(2, col_width)

    def resizeEvent(self, event): # pragma: no cover
        super().resizeEvent(event)
        self.resize_columns()

    def update_widget(self):
        self.conc_view.resizeColumnToContents(1)
        self.resize_columns()
        self.conc_view.resizeRowsToContents()

        if self.corpus is not None:
            self.n_matching = '{}/{}'.format(
                self.model.matching_docs() if self.word else 0,
                len(self.corpus))
            self.n_tokens = self.model.n_tokens
            self.n_types = self.model.n_types
        else:
            self.n_matching = ''
            self.n_tokens = ''
            self.n_types = ''

    def commit(self):
        selected_docs = sorted(set(self.model.word_index[row][0]
                                   for row in self.selected_rows))
        concordance = self.model.get_data()
        if selected_docs:
            selected = self.corpus[selected_docs]
            self.Outputs.selected_documents.send(selected)
        else:
            self.Outputs.selected_documents.send(None)
        self.Outputs.concordances.send(concordance)

    def send_report(self):
        view = self.conc_view
        model = self.conc_view.model()
        self.report_items("Concordances", (
            ("Query", model.word),
            ("Tokens", model.n_tokens),
            ("Types", model.n_types),
            ("Matching", self.n_matching),
        ))
        self.report_table(view)


if __name__ == '__main__': # pragma: no cover
    app = QApplication([])
    widget = OWConcordance()
    corpus = Corpus.from_file('book-excerpts')
    corpus = corpus[:3]
    widget.set_corpus(corpus)
    widget.show()
    app.exec()

