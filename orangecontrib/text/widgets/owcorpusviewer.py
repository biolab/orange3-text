from PyQt4 import QtGui
from PyQt4 import QtCore
from PyQt4.QtGui import *
from PyQt4.QtCore import *

from Orange.widgets import gui
from Orange.widgets.utils import vartype
from Orange.widgets.widget import OWWidget
from Orange.widgets.settings import Setting, ContextSetting

from orangecontrib.text.corpus import Corpus


class Input:
    CORPUS = "Corpus"


class Output:
    CORPUS = "Corpus"


class OWCorpusViewer(OWWidget):
    name = "Corpus Viewer"
    description = "Display corpus contents."
    icon = "icons/CorpusViewer.svg"
    priority = 30

    inputs = [(Input.CORPUS, Corpus, 'set_data')]
    outputs = [(Output.CORPUS, Corpus)]

    # Settings.
    selected_document = ContextSetting(0)
    selected_features = ContextSetting([0])
    autocommit = Setting(True)

    def __init__(self):
        super().__init__()

        self.corpus = None              # Corpus
        self.output_mask = None         # Output corpus indices
        self.document_contents = None   # QTextDocument
        self.document_holder = None     # QTextEdit
        self.features = []              # all attributes

        # ---- CONTROL AREA ----
        # Filtering results.
        filter_result_box = gui.widgetBox(self.controlArea, 'Info')
        self.info_all = gui.label(filter_result_box, self, 'All documents:')
        self.info_fil = gui.label(filter_result_box, self, 'After filtering:')
        # Feature selection.
        self.feature_listbox = gui.listBox(
            self.controlArea, self, 'selected_features', 'features',
            selectionMode=QtGui.QListView.ExtendedSelection,
            box='Displayed features', callback=self.load_documents,)

        # Auto-commit box.
        gui.auto_commit(self.controlArea, self, 'autocommit', 'Send data', 'Auto send is on')

        # ---- MAIN AREA ----
        # Search
        self.filter_input = gui.lineEdit(self.mainArea, self, '',
                                         orientation='horizontal', label='Filter:')
        self.filter_input.textChanged.connect(self.filter_input_changed)

        h_box = gui.widgetBox(self.mainArea, orientation='horizontal', addSpace=True)
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
        self.document_holder = QTextEdit()
        self.document_holder.setReadOnly(True)
        self.document_holder.setLineWrapMode(QTextEdit.WidgetWidth)
        h_box.layout().addWidget(self.document_holder)

    # --- DATA LOADING ---
    def set_data(self, data=None):
        self.reset_widget()  # Clear any old data.
        if data is not None:
            self.corpus = data
            self.load_features()
            self.load_documents()
            self.update_info_display()
            # Send the corpus to output.
            self.send(Output.CORPUS, self.corpus)

    def reset_widget(self):
        # Corpus.
        self.corpus = None
        self.output_mask = None
        # Widgets.
        self.feature_listbox.clear()
        self.document_holder.clear()
        self.filter_input.clear()
        self.update_info_display()
        # Models/vars.
        self.features.clear()
        self.selected_features.clear()
        self.document_table_model.clear()

    def load_features(self):
        self.selected_features = []
        if self.corpus is not None:
            self.features = [(meta.name, vartype(meta)) for meta in self.corpus.domain.metas] + \
                            [(attr.name, vartype(attr)) for attr in self.corpus.domain.attributes]
            self.selected_features = [0]  # Select the first feature.

    def load_documents(self):
        """ Loads documents into the left scrolling are. """
        if not self.corpus:
            return

        self.output_mask = []
        self.document_table_model.clear()
        search_keyword = self.filter_input.text()
        should_filter = True if search_keyword else False
        regex = QtCore.QRegExp(search_keyword)

        for i, document in enumerate(self.corpus):
            # TODO: remove corpus.documents due to problems with inconsistencies.
            document_contents = self.corpus.documents[i]

            has_hit = not should_filter     # Without a filter, every document 'has a hit'.
            if should_filter:
                index = regex.indexIn(document_contents, 0)
                if index != -1:
                    has_hit = True

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
        self.clear_text_highlight()  # Clear.

        self.document_contents = QTextDocument()    # Using a QTextDocument because it's easier to style.
        self.document_contents.setDefaultStyleSheet('td { padding: 5px 15px 15xp 5px; }')

        documents_html = []
        for index in self.document_table.selectionModel().selectedRows():
            document = index.data(Qt.UserRole)
            one_document_html = ''
            for feat_index in self.selected_features:
                meta_name = self.features[feat_index][0]  # 0 - name; 1 - index
                one_document_html += '<tr title={0}><td><strong>{0}:</strong></td><td>{1}</td></tr>'.format(
                        meta_name, document[meta_name].value)
            documents_html.append('<table>{}</table>'.format(one_document_html))

        self.document_contents.setHtml('<hr />'.join(documents_html))
        self.document_holder.setDocument(self.document_contents)
        self.highlight_document_hits()

    # --- WIDGET SEARCH ---
    def filter_input_changed(self):
        self.load_documents()
        self.highlight_document_hits()
        self.update_info_display()

    def highlight_document_hits(self):
        search_keyword = self.filter_input.text()
        self.clear_text_highlight()
        if not search_keyword:
            self.update_info_display()
            return

        # Format of the highlighting.
        text_format = QtGui.QTextCharFormat()
        text_format.setBackground(QtGui.QBrush(QtGui.QColor('#b3d8fe')))
        # Regular expression to match.
        regex = QtCore.QRegExp(search_keyword)
        cursor = self.document_contents.find(regex, 0)
        while cursor.position() != -1:
            cursor.mergeCharFormat(text_format)
            cursor = self.document_contents.find(regex, cursor.position())

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
            output_corpus = Corpus.from_table(self.corpus.domain, self.corpus,
                                              row_indices=self.output_mask)
            self.send(Output.CORPUS, output_corpus)

    def _invalidate_selection(self):
        self.commit()
