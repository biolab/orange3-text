from typing import Optional

from AnyQt.QtCore import Qt, QAbstractTableModel, QSize, QItemSelectionModel, QItemSelection
from AnyQt.QtWidgets import QSizePolicy, QApplication, QTableView, QStyledItemDelegate
from AnyQt.QtGui import QBrush, QColor

from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget
from nltk import ConcordanceIndex
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.preprocess import Preprocessor, WordPunctTokenizer


LastDocumentRole = next(gui.OrangeUserRole)


class HorizontalGridDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        if index.data(LastDocumentRole):
            painter.save()
            painter.setPen(QColor(212, 212, 212))
            painter.drawLine(option.rect.bottomLeft(), option.rect.bottomRight())
            painter.restore()
        if index.column() == 0:
            option.textElideMode = Qt.ElideLeft
        elif index.column() == 2:
            option.textElideMode = Qt.ElideRight
        QStyledItemDelegate.paint(self, painter, option, index)


class DocumentSelectionModel(QItemSelectionModel):
    def __init__(self, conc_model):
        super().__init__(conc_model)
        self.selected_docs = set()

    def select(self, selection, flags):
        indexes = selection.indexes() if isinstance(selection, QItemSelection) else [selection]
        if len(indexes) == 1 and indexes[0].row() == -1:
            self.selected_docs = set()
            self.clear()
            return
        word_index = self.model().word_index
        self.selected_docs = {word_index[index.row()][0] for index in indexes}
        selected_rows = [row_index for row_index, (doc_index, _) in enumerate(word_index)
                         if doc_index in self.selected_docs]
        selection = QItemSelection()
        for row in selected_rows:
            index = self.model().index(row, 0)
            selection.select(index, index)
        super().select(selection, flags)


class ConcordanceModel(QAbstractTableModel):
    "A model for constructing concordances from text."

    def __init__(self):
        QAbstractTableModel.__init__(self)
        self.word = None
        self.data = None
        self.indices = None
        self.word_index = None
        self.width = 8

    def set_word(self, word):
        self.modelAboutToBeReset.emit()
        self.word = word
        if self.indices is None:
            self.word_index = None
        else:
            self.word_index = [(doc_idx, offset) for doc_idx, doc in enumerate(self.indices)
                               for offset in doc.offsets(self.word)]
        self.modelReset.emit()

    def set_data(self, data):
        self.modelAboutToBeReset.emit()
        self.data = data
        self._compute_indices()
        self.modelReset.emit()

    def set_width(self, width):
        self.modelAboutToBeReset.emit()
        self.width = width
        self.modelReset.emit()

    def flags(self, _):
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def rowCount(self, parent=None, *args, **kwargs):
        return len(self.word_index) if self.word_index is not None else 0

    def columnCount(self, parent=None, *args, **kwargs):
        return 3

    def data(self, index, role=Qt.DisplayRole):
        row, col = index.row(), index.column()
        doc, index = self.word_index[row]

        if role == Qt.DisplayRole:
            tokens = self.data.tokens
            if col == 0:
                # one more if necessary to assure as much text as possible is returned
                if index - self.width > 0:
                    return ' '.join(tokens[doc][index - self.width:index])
                else:
                    return ' '.join(tokens[doc][0:index])
            if col == 1:
                return tokens[doc][index]
            if col == 2:
                if index + self.width < len(tokens[doc]):
                    return ' '.join(tokens[doc][index + 1:index + self.width])
                else:
                    return ' '.join(tokens[doc][index + 1:])

        elif role == Qt.TextAlignmentRole:
            return [Qt.AlignRight | Qt.AlignVCenter, Qt.AlignCenter, Qt.AlignLeft | Qt.AlignVCenter][col]

        elif role == Qt.ForegroundRole:
            if col == 1:
                return QBrush(QColor(0, 0, 255))

        elif role == LastDocumentRole:
             return row < len(self.word_index) - 1 and doc < self.word_index[row + 1][0]

    def _compute_indices(self):  # type: () -> Optional[None, list]
        if self.data is None:
            self.indices = None
            return
        if self.data and not self.data.has_tokens():
            preprocessor = Preprocessor(tokenizer=WordPunctTokenizer())
            preprocessor(self.data)
        self.indices = [ConcordanceIndex(doc, key=lambda x: x.lower()) for doc in self.data.tokens]

    def matching_docs(self):
        if self.indices and self.word:
            return sum(bool(doc.offsets(self.word)) for doc in self.indices)
        else:
            return 0


class OWConcordance(OWWidget):
    name = "Concordance"
    description = "Display the context of the word."
    icon = "icons/Concordance.svg"
    priority = 30000

    inputs = [('Corpus', Table, 'set_data')]
    outputs = [('Selected Documents', Table, )]

    autocommit = Setting(True)
    width = Setting(5)
    word = Setting("")
    # TODO Set selection settings.

    def __init__(self):
        super().__init__()

        self.corpus = None      # Corpus
        self.n_documents = ''
        self.n_matching = ''
        self.n_tokens = ''
        self.n_types = ''

        # Info attributes
        info_box = gui.widgetBox(self.controlArea, 'Info')
        gui.label(info_box, self, 'Documents: %(n_documents)s')
        gui.label(info_box, self, 'Tokens: %(n_tokens)s')
        gui.label(info_box, self, 'Types: %(n_types)s')
        gui.label(info_box, self, 'Matching: %(n_matching)s')

        # Width parameter
        gui.spin(self.controlArea, self, 'width', 3, 10, box=True, label="Number of words:", callback=self.set_width)

        gui.rubber(self.controlArea)

        # Search
        c_box = gui.widgetBox(self.mainArea, orientation="vertical")
        self.input = gui.lineEdit(c_box, self, 'word',
                                         orientation=Qt.Horizontal,
                                         sizePolicy=QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed),
                                         label='Query:', callback=self.refresh_search, callbackOnType=True)
        self.input.setFocus()

        # Concordances view
        self.conc_view = QTableView()
        self.model = ConcordanceModel()
        self.conc_view.setModel(self.model)
        self.conc_view.setWordWrap(False)
        self.conc_view.setSelectionBehavior(QTableView.SelectRows)
        self.conc_view.setSelectionModel(DocumentSelectionModel(self.model))
        self.conc_view.selectionModel().selectionChanged.connect(lambda: self.commit())
        self.conc_view.setItemDelegate(HorizontalGridDelegate())
        self.conc_view.horizontalHeader().hide()
        self.conc_view.setShowGrid(False)
        self.mainArea.layout().addWidget(self.conc_view)
        self.set_width()

        # Auto-commit box
        gui.auto_commit(self.controlArea, self, 'autocommit', 'Commit', 'Auto commit is on')

    def sizeHint(self):
        return QSize(600, 400)

    def set_width(self):
        self.model.set_width(self.width)

    def set_data(self, data=None):
        self.corpus = data
        if data is not None and not isinstance(data, Corpus):
            self.corpus = Corpus.from_table(data.domain, data)
        self.model.set_data(self.corpus)
        self.update_info()
        self.refresh_search()
        self.commit()

    def refresh_search(self):
        self.model.set_word(self.word)
        if self.corpus is not None and self.word:
            self.update_info()
        self.conc_view.resizeColumnToContents(1)
        self.resize_columns()
        self.conc_view.resizeRowsToContents()

    def resize_columns(self):
        col_width = (self.conc_view.width() - self.conc_view.columnWidth(1)) / 2 - 12
        self.conc_view.setColumnWidth(0, col_width)
        self.conc_view.setColumnWidth(2, col_width)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resize_columns()

    def update_info(self):
        if self.corpus is not None:
            self.n_documents = len(self.corpus)
            self.n_matching = '{}/{}'.format(self.model.matching_docs(), self.n_documents)
            self.n_tokens = sum(map(len, self.corpus.tokens)) if self.corpus.has_tokens() else 'n/a'
            self.n_types = len(self.corpus.dictionary) if self.corpus.has_tokens() else 'n/a'
        else:
            self.n_documents = ''
            self.n_matching = ''
            self.n_tokens = ''
            self.n_types = ''

    def commit(self):
        selected_docs = sorted(self.conc_view.selectionModel().selected_docs)
        if selected_docs:
            selected = self.corpus[selected_docs]
            self.send("Selected Documents", selected)
        else:
            self.send("Selected Documents", None)


if __name__ == '__main__':
    app = QApplication([])
    widget = OWConcordance()
    corpus = Corpus.from_file('bookexcerpts')
    corpus = corpus[:3]
    widget.set_data(corpus)
    widget.show()
    app.exec()

