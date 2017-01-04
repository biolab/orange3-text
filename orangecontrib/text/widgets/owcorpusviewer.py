import os
import re
import sre_constants
from itertools import chain

from AnyQt.QtCore import Qt, QUrl
from AnyQt.QtGui import QStandardItemModel, QStandardItem
from AnyQt.QtWidgets import (QListView, QSizePolicy, QTableView,
                             QAbstractItemView, QHeaderView, QSplitter,
                             QApplication)

from Orange.data import Table
from Orange.data.domain import filter_visible
from Orange.widgets import gui, widget
from Orange.widgets.settings import Setting, ContextSetting
from Orange.widgets.utils.webview import HAVE_WEBKIT
from Orange.widgets.widget import OWWidget, Msg
from orangecontrib.text.corpus import Corpus


class IO:
    DATA = 'Data'
    MATCHED = "Matching Docs"
    UNMATCHED = "Other Docs"


class OWCorpusViewer(OWWidget):
    name = "Corpus Viewer"
    description = "Display corpus contents."
    icon = "icons/CorpusViewer.svg"
    priority = 70

    inputs = [(IO.DATA, Table, 'set_data')]
    outputs = [(IO.MATCHED, Corpus, widget.Default), (IO.UNMATCHED, Corpus)]

    search_indices = ContextSetting([0])   # features included in search
    display_indices = ContextSetting([0])  # features for display
    show_tokens = Setting(False)
    autocommit = Setting(True)

    class Warning(OWWidget.Warning):
        no_feats_search = Msg('No features included in search.')
        no_feats_display = Msg('No features selected for display.')

    def __init__(self):
        super().__init__()

        self.corpus = None              # Corpus
        self.corpus_docs = None         # Documents generated from Corpus
        self.output_mask = []           # Output corpus indices
        self.doc_webview = None         # WebView for showing content
        self.search_features = []       # two copies are needed since Display allows drag & drop
        self.display_features = []

        # Info attributes
        self.update_info()
        info_box = gui.widgetBox(self.controlArea, 'Info')
        gui.label(info_box, self, 'Documents: %(n_documents)s')
        gui.label(info_box, self, 'Preprocessed: %(is_preprocessed)s')
        gui.label(info_box, self, '  ◦ Tokens: %(n_tokens)s')
        gui.label(info_box, self, '  ◦ Types: %(n_types)s')
        gui.label(info_box, self, 'POS tagged: %(is_pos_tagged)s')
        gui.label(info_box, self, 'N-grams range: %(ngram_range)s')
        gui.label(info_box, self, 'Matching: %(n_matching)s')

        # Search features
        self.search_listbox = gui.listBox(
            self.controlArea, self, 'search_indices', 'search_features',
            selectionMode=QListView.ExtendedSelection,
            box='Search features', callback=self.regenerate_docs,)

        # Display features
        display_box = gui.widgetBox(self.controlArea, 'Display features')
        self.display_listbox = gui.listBox(
            display_box, self, 'display_indices', 'display_features',
            selectionMode=QListView.ExtendedSelection,
            callback=self.show_docs, enableDragDrop=True)
        self.show_tokens_checkbox = gui.checkBox(display_box, self, 'show_tokens',
                                                 'Show Tokens && Tags', callback=self.show_docs)

        # Auto-commit box
        gui.auto_commit(self.controlArea, self, 'autocommit', 'Send data', 'Auto send is on')

        # Search
        self.filter_input = gui.lineEdit(self.mainArea, self, '',
                                         orientation=Qt.Horizontal,
                                         sizePolicy=QSizePolicy(QSizePolicy.MinimumExpanding,
                                                                QSizePolicy.Fixed),
                                         label='RegExp Filter:')
        self.filter_input.textChanged.connect(self.refresh_search)

        # Main area
        self.splitter = QSplitter(
            orientation=Qt.Horizontal,
            childrenCollapsible=False,
        )

        # Document list
        self.doc_list = QTableView()
        self.doc_list.setSelectionBehavior(QTableView.SelectRows)
        self.doc_list.setSelectionMode(QTableView.ExtendedSelection)
        self.doc_list.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.doc_list.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.doc_list.horizontalHeader().setVisible(False)
        self.splitter.addWidget(self.doc_list)

        self.doc_list_model = QStandardItemModel(self)
        self.doc_list.setModel(self.doc_list_model)
        self.doc_list.selectionModel().selectionChanged.connect(self.show_docs)

        # Document contents
        # For PyQt5 WebEngine's setHtml grabs the focus and makes typing hard
        # More info: http://stackoverflow.com/questions/36609489
        # To bypass the annoying behaviour disable the widget for WebEngine
        self.doc_webview = gui.WebviewWidget(self.splitter, self,
                                             debug=True, enabled=HAVE_WEBKIT)

        self.mainArea.layout().addWidget(self.splitter)

    def copy_to_clipboard(self):
        text = self.doc_webview.selectedText()
        QApplication.clipboard().setText(text)

    def set_data(self, data=None):
        self.reset_widget()
        self.corpus = data
        if data is not None:
            if not isinstance(data, Corpus):
                self.corpus = Corpus.from_table(data.domain, data)
            self.load_features()
            self.regenerate_docs()
        self.commit()

    def reset_widget(self):
        # Corpus
        self.corpus = None
        self.corpus_docs = None
        self.output_mask = []
        # Widgets
        self.search_listbox.clear()
        self.display_listbox.clear()
        self.filter_input.clear()
        self.update_info()
        # Models/vars
        self.search_features.clear()
        self.display_features.clear()
        self.search_indices.clear()
        self.display_indices.clear()
        self.doc_list_model.clear()
        # Warnings
        self.Warning.clear()
        # WebView
        self.doc_webview.setHtml('')

    def load_features(self):
        self.search_indices = []
        self.display_indices = []
        if self.corpus is not None:
            domain = self.corpus.domain
            self.search_features = list(filter_visible(chain(domain.variables, domain.metas)))
            self.display_features = list(filter_visible(chain(domain.variables, domain.metas)))
            # FIXME: Select features based on ContextSetting
            self.search_indices = list(range(len(self.search_features)))
            self.display_indices = list(range(len(self.display_features)))

            # Enable/disable tokens checkbox
            if not self.corpus.has_tokens():
                self.show_tokens_checkbox.setCheckState(False)
            self.show_tokens_checkbox.setEnabled(self.corpus.has_tokens())

    def list_docs(self):
        """ List documents into the left scrolling area """
        search_keyword = self.filter_input.text().strip('|')
        try:
            reg = re.compile(search_keyword, re.IGNORECASE)
        except sre_constants.error:
            return

        def is_match(x):
            return not bool(search_keyword) or reg.search(x)

        self.output_mask.clear()
        self.doc_list_model.clear()

        for i, (doc, title, content) in enumerate(zip(self.corpus, self.corpus.titles,
                                                      self.corpus_docs)):
            if is_match(content):
                item = QStandardItem()
                item.setData(title, Qt.DisplayRole)
                item.setData(doc, Qt.UserRole)
                self.doc_list_model.appendRow(item)
                self.output_mask.append(i)

        if self.doc_list_model.rowCount() > 0:
            self.doc_list.selectRow(0)          # Select the first document
        else:
            self.doc_webview.setHtml('')
        self.commit()

    def show_docs(self):
        """ Show the selected documents in the right area """
        HTML = '''
        <!doctype html>
        <html>
        <head>
        <meta charset='utf-8'>
        <style>

        table {{ border-collapse: collapse; }}
        mark {{ background: #FFCD28; }}

        tr > td {{
            padding-bottom: 3px;
            padding-top: 3px;
        }}

        body {{
            font-family: Helvetica;
            font-size: 10pt;
        }}

        .line {{ border-bottom: 1px solid #000; }}
        .separator {{ height: 5px; }}

        .variables {{
            vertical-align: top;
            padding-right: 10px;
        }}

        .token {{
            padding: 3px;
            border: 1px #B0B0B0 solid;
            margin-right: 5px;
            margin-bottom: 5px;
            display: inline-block;
        }}

        img {{
            max-width: 100%;
        }}

        </style>
        </head>
        <body>
        {}
        </body>
        </html>
        '''
        if self.corpus is None:
            return

        self.Warning.no_feats_display.clear()
        if len(self.display_indices) == 0:
            self.Warning.no_feats_display()

        if self.show_tokens:
            tokens = list(self.corpus.ngrams_iterator(include_postags=True))

        marked_search_features = [f for i, f in enumerate(self.search_features)
                                  if i in self.search_indices]

        html = '<table>'
        for doc_count, index in enumerate(self.doc_list.selectionModel().selectedRows()):
            if doc_count > 0:   # add split
                html += '<tr class="line separator"><td/><td/></tr>' \
                        '<tr class="separator"><td/><td/></tr>'

            row_ind = index.data(Qt.UserRole).row_index
            for ind in self.display_indices:
                feature = self.display_features[ind]
                mark = 'class="mark-area"' if feature in marked_search_features else ''
                value = str(index.data(Qt.UserRole)[feature.name])
                is_image = feature.attributes.get('type', '') == 'image'
                if is_image and value != '?':
                    value = '<img src="{}"></img>'.format(value)
                html += '<tr><td class="variables"><strong>{}:</strong></td>' \
                        '<td {}>{}</td></tr>'.format(
                    feature.name, mark, value)

            if self.show_tokens:
                html += '<tr><td class="variables"><strong>Tokens & Tags:</strong></td>' \
                        '<td>{}</td></tr>'.format(''.join('<span class="token">{}</span>'.format(
                    token) for token in tokens[row_ind]))

        html += '</table>'

        # QUrl is a workaround to allow local resources
        # https://bugreports.qt.io/browse/QTBUG-55902?focusedCommentId=335945
        self.doc_webview.setHtml(HTML.format(html), QUrl("file://"))
        self.load_js()
        self.highlight_docs()

    def load_js(self):
        resources = os.path.join(os.path.dirname(__file__), 'resources')
        for script in ('jquery-3.1.1.min.js', 'jquery.mark.min.js', 'highlighter.js', ):
            self.doc_webview.evalJS(open(os.path.join(resources, script), encoding='utf-8').read())

    def regenerate_docs(self):
        self.corpus_docs = None
        self.Warning.no_feats_search.clear()
        if self.corpus is not None:
            feats = [self.search_features[i] for i in self.search_indices]
            if len(feats) == 0:
                self.Warning.no_feats_search()
            self.corpus_docs = self.corpus.documents_from_features(feats)
            self.refresh_search()

    def refresh_search(self):
        if self.corpus is not None:
            self.list_docs()
            self.update_info()

    def highlight_docs(self):
        search_keyword = self.filter_input.text().\
            strip('|').replace('\\', '\\\\')    # escape one \ to  two for mark.js
        if search_keyword:
            self.doc_webview.evalJS('mark("{}");'.format(search_keyword))

    def update_info(self):
        if self.corpus is not None:
            self.n_documents = len(self.corpus)
            self.n_matching = '{}/{}'.format(self.doc_list_model.rowCount(), self.n_documents)
            self.n_tokens = sum(map(len, self.corpus.tokens)) if self.corpus.has_tokens() else 'n/a'
            self.n_types = len(self.corpus.dictionary) if self.corpus.has_tokens() else 'n/a'
            self.is_preprocessed = self.corpus.has_tokens()
            self.is_pos_tagged = self.corpus.pos_tags is not None
            self.ngram_range = '{}-{}'.format(*self.corpus.ngram_range)
        else:
            self.n_documents = ''
            self.n_matching = ''
            self.n_tokens = ''
            self.n_types = ''
            self.is_preprocessed = ''
            self.is_pos_tagged = ''
            self.ngram_range = ''

    def commit(self):
        if self.corpus is not None:
            matched = self.corpus[self.output_mask]
            output_mask = set(self.output_mask)
            unmatched_mask = [i for i in range(len(self.corpus)) if i not in output_mask]
            unmatched = self.corpus[unmatched_mask]
            self.send(IO.MATCHED, matched)
            self.send(IO.UNMATCHED, unmatched)
        else:
            self.send(IO.MATCHED, None)
            self.send(IO.UNMATCHED, None)

if __name__ == '__main__':
    from orangecontrib.text.tag import pos_tagger
    app = QApplication([])
    widget = OWCorpusViewer()
    widget.show()
    corpus = Corpus.from_file('bookexcerpts')
    corpus = corpus[:3]
    corpus = pos_tagger.tag_corpus(corpus)
    corpus.ngram_range = (1, 2)
    widget.set_data(corpus)
    app.exec()
