import re
import sre_constants
from itertools import chain

from PyQt4 import QtCore
from PyQt4 import QtGui
from PyQt4.QtCore import *
from PyQt4.QtGui import *

from Orange.widgets import gui
from Orange.widgets.settings import Setting, ContextSetting
from Orange.widgets.widget import OWWidget, Msg
from Orange.data import Table
from Orange.data.domain import filter_visible
from orangecontrib.text.corpus import Corpus


class Input:
    DATA = 'Data'


class Output:
    CORPUS = "Corpus"


class DocumentViewer(QTextEdit):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setReadOnly(True)
        self.setLineWrapMode(self.WidgetWidth)

    def sizeHint(self):
        return QSize(700, 300)


class OWCorpusViewer(OWWidget):
    name = "Corpus Viewer"
    description = "Display corpus contents."
    icon = "icons/CorpusViewer.svg"
    priority = 30

    inputs = [(Input.DATA, Table, 'set_data')]
    outputs = [(Output.CORPUS, Corpus)]

    # Settings.
    selected_document = ContextSetting(0)
    search_features = ContextSetting([0])   # features included in search
    display_features = ContextSetting([0])  # features for display
    autocommit = Setting(True)

    class Warning(OWWidget.Warning):
        no_feats_search = Msg('No features included in search.')
        no_feats_display = Msg('No features selected for display.')

    def __init__(self):
        super().__init__()

        self.corpus = None              # Corpus
        self.corpus_docs = None         # Documents generated from Corpus
        self.output_mask = None         # Output corpus indices
        self.document_contents = None   # QTextDocument
        self.document_holder = None     # DocumentViewer
        self.features = []              # all attributes

        # ---- CONTROL AREA ----
        # Filtering results.
        filter_result_box = gui.widgetBox(self.controlArea, 'Info')
        self.info_all = gui.label(filter_result_box, self, 'All documents:')
        self.info_fil = gui.label(filter_result_box, self, 'After filtering:')

        # Search features
        self.search_listbox = gui.listBox(
            self.controlArea, self, 'search_features', 'features',
            selectionMode=QtGui.QListView.ExtendedSelection,
            box='Search features', callback=self.regenerate_documents,)

        # Display features
        self.display_listbox = gui.listBox(
            self.controlArea, self, 'display_features', 'features',
            selectionMode=QtGui.QListView.ExtendedSelection,
            box='Display features', callback=self.show_document, )

        # Auto-commit box.
        gui.auto_commit(self.controlArea, self, 'autocommit', 'Send data', 'Auto send is on')

        # ---- MAIN AREA ----
        # Search
        self.filter_input = gui.lineEdit(self.mainArea, self, '',
                                         orientation=Qt.Horizontal,
                                         label='RegExp Filter:')
        self.filter_input.textChanged.connect(self.refresh_search)

        h_box = gui.widgetBox(self.mainArea, orientation=Qt.Horizontal, addSpace=True)
        h_box.layout().setSpacing(0)

        # Document list.
        self.document_table = QTableView()
        self.document_table.setSelectionBehavior(QTableView.SelectRows)
        self.document_table.setSelectionMode(QTableView.ExtendedSelection)
        self.document_table.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        self.document_table.horizontalHeader().setResizeMode(QtGui.QHeaderView.Stretch)
        self.document_table.horizontalHeader().setVisible(False)
        h_box.layout().addWidget(self.document_table)

        self.document_table_model = QStandardItemModel(self)

        self.document_table.setModel(self.document_table_model)
        self.document_table.setFixedWidth(200)
        self.document_table.selectionModel().selectionChanged.connect(self.show_document)

        # Document contents.
        self.document_holder = DocumentViewer()
        h_box.layout().addWidget(self.document_holder)

    # --- DATA LOADING ---
    def set_data(self, data=None):
        self.reset_widget()  # Clear any old data.
        if data is not None:
            self.corpus = data
            if not isinstance(data, Corpus):
                self.corpus = Corpus.from_table(data.domain, data)
            self.load_features()
            self.regenerate_documents()
            # Send the corpus to output.
            self.send(Output.CORPUS, self.corpus)

    def reset_widget(self):
        # Corpus.
        self.corpus = None
        self.corpus_docs = None
        self.output_mask = None
        # Widgets.
        self.search_listbox.clear()
        self.display_listbox.clear()
        self.document_holder.clear()
        self.filter_input.clear()
        self.update_info_display()
        # Models/vars.
        self.features.clear()
        self.search_features.clear()
        self.display_features.clear()
        self.document_table_model.clear()
        # Warnings.
        self.Warning.clear()

    def load_features(self):
        self.search_features = []
        self.display_features = []
        if self.corpus is not None:
            domain = self.corpus.domain
            self.features = list(filter_visible(chain(domain.variables, domain.metas)))
            # FIXME: Select features based on ContextSetting
            self.search_features = list(range(len(self.features)))
            self.display_features = list(range(len(self.features)))

    def load_documents(self):
        """ Loads documents into the left scrolling area. """
        if not self.corpus or not self.corpus_docs:
            return

        search_keyword = self.filter_input.text().strip('|')
        try:
            is_match = re.compile(search_keyword, re.IGNORECASE).search
        except sre_constants.error:
            return

        should_filter = bool(search_keyword)
        self.output_mask = []
        self.document_table_model.clear()

        for i, (document, document_contents) in enumerate(zip(self.corpus, self.corpus_docs)):
            has_hit = not should_filter or is_match(document_contents)
            if has_hit:
                item = QStandardItem()
                item.setData('Document {}'.format(i+1), Qt.DisplayRole)
                item.setData(document, Qt.UserRole)

                self.document_table_model.appendRow(item)
                self.output_mask.append(i)

        if self.document_table_model.rowCount() > 0:
            self.document_table.selectRow(0)    # Select the first document.
        else:
            self.document_contents.clear()

        self._invalidate_selection()

    def show_document(self):
        """ Show the selected document in the right area. """
        self.Warning.no_feats_display.clear()
        if len(self.display_features) == 0 and self.corpus is not None:
            self.Warning.no_feats_display()
        self.clear_text_highlight()  # Clear.

        self.document_contents = QTextDocument(undoRedoEnabled=False)
        self.document_contents.setDefaultStyleSheet('td { padding: 5px 15px 15xp 5px; }')

        documents = []
        for index in self.document_table.selectionModel().selectedRows():
            document = ['<table>']
            for feat_index in self.display_features:
                feature = self.features[feat_index]
                value = index.data(Qt.UserRole)[feature.name]

                document.append(
                    '<tr><td><strong>{0}:</strong></td><td>{1}</td></tr>'.format(
                        feature.name, value))
            document.append('</table><hr/>')
            documents.append(document)

        self.document_contents.setHtml(''.join(chain.from_iterable(documents)))
        self.document_holder.setDocument(self.document_contents)
        self.highlight_document_hits()

    # --- WIDGET SEARCH ---
    def regenerate_documents(self):
        self.corpus_docs = None
        self.Warning.no_feats_search.clear()
        if self.corpus is not None:
            feats = [self.features[i] for i in self.search_features]
            if len(feats) == 0:
                self.Warning.no_feats_search()
            self.corpus_docs = self.corpus.documents_from_features(feats)
            self.refresh_search()

    def refresh_search(self):
        self.load_documents()
        self.highlight_document_hits()
        self.update_info_display()

    def highlight_document_hits(self):
        search_keyword = self.filter_input.text().strip('|')
        self.clear_text_highlight()
        if not search_keyword:
            self.update_info_display()
            return

        # Format of the highlighting.
        text_format = QtGui.QTextCharFormat()
        text_format.setBackground(QtGui.QBrush(QtGui.QColor('#b3d8fe')))
        # Regular expression to match.
        regex = QtCore.QRegExp(search_keyword)
        regex.setCaseSensitivity(Qt.CaseInsensitive)
        cursor = self.document_contents.find(regex, 0)
        prev_position = None
        cursor.beginEditBlock()
        while cursor.position() not in (-1, prev_position):
            cursor.mergeCharFormat(text_format)
            prev_position = cursor.position()
            cursor = self.document_contents.find(regex, cursor.position())
        cursor.endEditBlock()

    def update_info_display(self):
        if self.corpus is not None:
            self.info_all.setText('All documents: {}'.format(len(self.corpus)))
            self.info_fil.setText('After filtering: {}'.format(self.document_table_model.rowCount()))
        else:
            self.info_all.setText('All documents:')
            self.info_fil.setText('After filtering:')

    def clear_text_highlight(self):
        text_format = QtGui.QTextCharFormat()
        text_format.setBackground(QtGui.QBrush(QtGui.QColor('#ffffff')))

        cursor = self.document_holder.textCursor()
        cursor.setPosition(0)
        cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor, 1)
        cursor.mergeCharFormat(text_format)

    # --- MISC ---
    def commit(self):
        if self.output_mask is not None:
            output_corpus = Corpus.from_corpus(self.corpus.domain, self.corpus,
                                               row_indices=self.output_mask)
            self.send(Output.CORPUS, output_corpus)

    def _invalidate_selection(self):
        self.commit()


if __name__ == '__main__':
    app = QApplication([])
    widget = OWCorpusViewer()
    widget.show()
    corpus = Corpus.from_file('bookexcerpts')
    widget.set_data(corpus)
    app.exec()
