import os
import re
import sre_constants
from typing import Any, Iterable, List, Set

import numpy as np
from AnyQt.QtCore import (
    QAbstractListModel,
    QEvent,
    QItemSelection,
    QItemSelectionModel,
    QItemSelectionRange,
    QModelIndex,
    QSortFilterProxyModel,
    Qt,
    QUrl,
)
from AnyQt.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QHeaderView,
    QListView,
    QSizePolicy,
    QSplitter,
    QTableView,
    QMessageBox,
)
from Orange.data import Variable
from Orange.data.domain import Domain, filter_visible
from Orange.widgets import gui
from Orange.widgets.settings import ContextSetting, Setting, DomainContextHandler
from Orange.widgets.utils.annotated_data import create_annotated_table
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.widget import Input, Msg, Output, OWWidget
from orangecanvas.gui.utils import disconnected
from orangewidget.utils.listview import ListViewSearch

from orangecontrib.text.corpus import Corpus

HTML = """
<!doctype html>
<html>
<head>
<script type="text/javascript" src="resources/jquery-3.1.1.min.js">
</script>
<script type="text/javascript" src="resources/jquery.mark.min.js">
</script>
<script type="text/javascript" src="resources/highlighter.js">
</script>
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

.content {{
    /* Adopted from https://css-tricks.com/snippets/css/prevent-long-urls-from-breaking-out-of-container/ */

    /* These are technically the same, but use both */
    overflow-wrap: break-word;
    word-wrap: break-word;

    -ms-word-break: break-all;
    /* This is the dangerous one in WebKit, as it breaks things wherever */
    word-break: break-all;
    /* Instead use this non-standard one: */
    word-break: break-word;

    /* Adds a hyphen where the word breaks, if supported (No Blink) */
    -ms-hyphens: auto;
    -moz-hyphens: auto;
    -webkit-hyphens: auto;
    hyphens: auto;
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
"""
SEPARATOR = (
    '<tr class="line separator"><td/><td/></tr><tr class="separator"><td/><td/></tr>'
)


def _count_matches(content: List[str], regex: re.Pattern, state: TaskState) -> int:
    """
    Count number of appears of any terms in search_string in content texts.

    Parameters
    ----------
    content
        List of texts where we count appearances
    search_string
        Strings that are searched in texts. This parameter has a format
        term1|term2|term3|...

    Returns
    -------
    Number of all matches of search_string in all texts in content list
    """
    matches = 0
    if regex.pattern:
        for i, text in enumerate(content):
            matches += len(regex.findall(text))
            state.set_progress_value((i + 1) / len(content) * 100)
    return matches


class DocumentListModel(QAbstractListModel):
    """
    Custom model for listing documents. Using custom model since Onrage's
    pylistmodel is too slow for large number of documents
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__visible_data = []
        self.__filter_content = []

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if role == Qt.DisplayRole:
            return self.__visible_data[index.row()]
        elif role == Qt.UserRole:
            return self.__filter_content[index.row()]

    def rowCount(self, parent: QModelIndex = None, *args, **kwargs) -> int:
        return len(self.__visible_data)

    def setup_data(self, data: List[str], content: List[str]):
        self.beginResetModel()
        self.__visible_data = data
        self.__filter_content = content
        self.endResetModel()

    def update_filter_content(self, content: List[str]):
        assert len(content) == len(self.__visible_data)
        self.__filter_content = content

    def get_filter_content(self) -> List[str]:
        return self.__filter_content

    def clear(self):
        self.beginResetModel()
        self.__visible_data = []
        self.__filter_content = []
        self.endResetModel()


class DocumentsFilterProxyModel(QSortFilterProxyModel):
    """Filter model for documents list"""

    __regex = None

    def set_filter_string(self, filter_string: str):
        try:
            re.compile(filter_string.strip("|"), re.IGNORECASE)
        except re.error:
            return
        self.__regex = re.compile(filter_string.strip("|"), re.IGNORECASE)
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        """Filter document that mathc the filter string"""
        if self.__regex is None:
            # filter is not defined yet - show all
            return True
        else:
            index = self.sourceModel().index(source_row, 0, source_parent)
            content = self.sourceModel().data(index, Qt.UserRole)
            res = self.__regex.search(content)
            return bool(res)


class DocumentTableView(QTableView):
    """TableView that disables unselecting all items"""

    def selectionCommand(
        self, index: QModelIndex, event: QEvent = None
    ) -> QItemSelectionModel.SelectionFlags:
        flags = super().selectionCommand(index, event)
        selmodel = self.selectionModel()
        if not index.isValid():  # Click on empty viewport; don't clear
            return QItemSelectionModel.NoUpdate
        if selmodel.isSelected(index):
            currsel = selmodel.selectedIndexes()
            if len(currsel) == 1 and index == currsel[0]:
                # Is the last selected index; do not deselect it
                return QItemSelectionModel.NoUpdate
        if (
            event is not None
            and event.type() == QEvent.MouseMove
            and flags & QItemSelectionModel.ToggleCurrent
        ):
            # Disable ctrl drag 'toggle'; can be made to deselect the last
            # index, would need to keep track of the current selection
            # (selectionModel does this but does not expose it)
            flags &= ~QItemSelectionModel.Toggle
            flags |= QItemSelectionModel.Select
        return flags


class VariableListViewSearch(ListViewSearch):
    """ListViewSearch that disables unselecting all items in the list"""

    def selectionCommand(
        self, index: QModelIndex, event: QEvent = None
    ) -> QItemSelectionModel.SelectionFlags:
        flags = super().selectionCommand(index, event)
        selmodel = self.selectionModel()
        if not index.isValid():  # Click on empty viewport; don't clear
            return QItemSelectionModel.NoUpdate
        if selmodel.isSelected(index):
            currsel = selmodel.selectedIndexes()
            if len(currsel) == 1 and index == currsel[0]:
                # Is the last selected index; do not deselect it
                return QItemSelectionModel.NoUpdate
        if (
            event is not None
            and event.type() == QEvent.MouseMove
            and flags & QItemSelectionModel.ToggleCurrent
        ):
            # Disable ctrl drag 'toggle'; can be made to deselect the last
            # index, would need to keep track of the current selection
            # (selectionModel does this but does not expose it)
            flags &= ~QItemSelectionModel.Toggle
            flags |= QItemSelectionModel.Select
        return flags

    def set_selection(self, items: Iterable[Variable]):
        """Set selected items in the list view"""
        model = self.model()
        values = self.model()[:]
        items = [it for it in items if it in values]
        selection = QItemSelection()
        if items:
            for val in items:
                index = values.index(val)
                selection.merge(
                    QItemSelection(model.index(index, 0), model.index(index, 0)),
                    QItemSelectionModel.Select,
                )
        self.selectionModel().select(selection, QItemSelectionModel.ClearAndSelect)


class VisibleDomainModel(DomainModel):
    """Domain model that filter only visible features"""

    def set_domain(self, domain):
        if domain is not None:
            domain = Domain(
                filter_visible(domain.attributes),
                class_vars=filter_visible(domain.class_vars),
                metas=filter_visible(domain.metas),
            )
        super().set_domain(domain)


class OWCorpusViewer(OWWidget, ConcurrentWidgetMixin):
    name = "Corpus Viewer"
    description = "Display corpus contents."
    icon = "icons/CorpusViewer.svg"
    priority = 500
    keywords = "corpus viewer, display"

    class Inputs:
        corpus = Input("Corpus", Corpus, replaces=["Data"])

    class Outputs:
        matching_docs = Output("Matching Docs", Corpus, default=True)
        other_docs = Output("Other Docs", Corpus)
        corpus = Output("Corpus", Corpus)

    settingsHandler = DomainContextHandler()

    settings_version = 2
    search_features: List[Variable] = ContextSetting([])
    display_features: List[Variable] = ContextSetting([])
    selected_documents: Set[int] = Setting({0}, schema_only=True)
    regexp_filter = ContextSetting("")
    show_tokens = Setting(False)
    autocommit = Setting(True)

    class Warning(OWWidget.Warning):
        no_feats_search = Msg("No features included in search.")
        no_feats_display = Msg("No features selected for display.")
        
    class Error(OWWidget.Error):
        invalid_regex = Msg("Invalid regular expression.")

    def __init__(self):
        super().__init__()
        ConcurrentWidgetMixin.__init__(self)

        self.corpus = None  # Corpus
        self.__pending_selected_documents = self.selected_documents

        # Info attributes
        self.update_info()
        info_box = gui.widgetBox(self.controlArea, "Info")
        gui.label(info_box, self, "Tokens: %(n_tokens)s")
        gui.label(info_box, self, "Types: %(n_types)s")
        gui.label(info_box, self, "Matching documents: %(n_matching)s")
        gui.label(info_box, self, "Matches: %(n_matches)s")

        # Search features
        ex_sel = QListView.ExtendedSelection
        search_box = gui.widgetBox(self.controlArea, "Search features")
        self.search_listbox = sl = VariableListViewSearch(selectionMode=ex_sel)
        search_box.layout().addWidget(sl)
        sl.setModel(VisibleDomainModel(separators=False))
        sl.selectionModel().selectionChanged.connect(self.search_features_changed)

        # Display features
        display_box = gui.widgetBox(self.controlArea, "Display features")
        self.display_listbox = dl = VariableListViewSearch(selectionMode=ex_sel)
        display_box.layout().addWidget(dl)
        dl.setModel(VisibleDomainModel(separators=False))
        dl.selectionModel().selectionChanged.connect(self.display_features_changed)

        self.show_tokens_checkbox = gui.checkBox(
            display_box,
            self,
            "show_tokens",
            "Show Tokens && Tags",
            callback=self.show_docs,
        )

        # Auto-commit box
        gui.auto_commit(
            self.controlArea, self, "autocommit", "Send data", "Auto send is on"
        )

        # Search
        self.filter_input = gui.lineEdit(
            self.mainArea,
            self,
            "regexp_filter",
            orientation=Qt.Horizontal,
            sizePolicy=QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed),
            label="RegExp Filter:",
            callback=self.refresh_search,
        )

        # Main area
        self.splitter = QSplitter(orientation=Qt.Horizontal, childrenCollapsible=False)
        # Document list
        self.doc_list = DocumentTableView()
        self.doc_list.setSelectionBehavior(QTableView.SelectRows)
        self.doc_list.setSelectionMode(QTableView.ExtendedSelection)
        self.doc_list.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.doc_list.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.doc_list.horizontalHeader().setVisible(False)
        self.splitter.addWidget(self.doc_list)

        self.doc_list_model = DocumentListModel()
        proxy_model = DocumentsFilterProxyModel()
        proxy_model.setSourceModel(self.doc_list_model)
        self.doc_list.setModel(proxy_model)
        self.doc_list.selectionModel().selectionChanged.connect(self.selection_changed)
        # Document contents
        self.doc_webview = gui.WebviewWidget(self.splitter, debug=False)

        self.mainArea.layout().addWidget(self.splitter)

    def copy_to_clipboard(self):
        text = self.doc_webview.selectedText()
        QApplication.clipboard().setText(text)

    @Inputs.corpus
    def set_data(self, corpus=None):
        self.closeContext()
        self.reset_widget()
        self.corpus = corpus
        if corpus is not None:
            self.setup_controls()
            self.openContext(self.corpus)
            self.doc_list.model().set_filter_string(self.regexp_filter)
            self.select_variables()
            self.list_docs()
            self.update_info()
            self.set_selection()
            self.show_docs()
        self.commit.now()

    def reset_widget(self):
        # Corpus
        self.corpus = None
        # Widgets
        self.search_listbox.model().set_domain(None)
        self.display_listbox.model().set_domain(None)
        self.filter_input.clear()
        self.update_info()
        # Models/vars
        self.doc_list_model.clear()
        # Warnings
        self.Warning.clear()
        # WebView
        self.doc_webview.setHtml("")

    def setup_controls(self):
        """Setup controls in control area"""
        domain = self.corpus.domain
        if not self.corpus.has_tokens():
            self.show_tokens_checkbox.setCheckState(Qt.Unchecked)
        self.show_tokens_checkbox.setEnabled(self.corpus.has_tokens())
        self.search_listbox.model().set_domain(domain)
        self.display_listbox.model().set_domain(domain)
        self.search_features = self.search_listbox.model()[:]
        self.display_features = self.display_listbox.model()[:]

    def select_variables(self):
        """Set selection to display and search features view boxes"""
        smodel = self.search_listbox.model()
        dmodel = self.display_listbox.model()
        # it can happen that domain handler will set some features that are
        # not part of domain - remove them
        self.search_features = [f for f in self.search_features if f in smodel]
        self.display_features = [f for f in self.display_features if f in dmodel]
        # if no features after removing non-existent, select all - default
        if not self.search_features:
            self.search_features = smodel[:]
        if not self.display_features:
            self.display_features = dmodel[:]
        with disconnected(
            self.search_listbox.selectionModel().selectionChanged,
            self.search_features_changed,
        ):
            self.search_listbox.set_selection(self.search_features)
        with disconnected(
            self.display_listbox.selectionModel().selectionChanged,
            self.display_features_changed,
        ):
            self.display_listbox.set_selection(self.display_features)

    def list_docs(self):
        """List documents into the left scrolling area"""
        docs = self.regenerate_docs()
        self.doc_list_model.setup_data(self.corpus.titles.tolist(), docs)

    def get_selected_indexes(self) -> Set[int]:
        m = self.doc_list.model().mapToSource
        return {m(i).row() for i in self.doc_list.selectionModel().selectedRows()}

    def set_selection(self) -> None:
        """
        Select documents in selected_documents attribute in the view
        """
        self.selected_documents = self.__pending_selected_documents
        self.__pending_selected_documents = {0}
        view = self.doc_list
        model = view.model()
        source_model = model.sourceModel()

        selection = QItemSelection()
        self.selected_documents = {
            r for r in self.selected_documents if r < len(self.corpus)
        }
        for row in self.selected_documents:
            index = model.mapFromSource(source_model.index(row, 0))
            selection.append(QItemSelectionRange(index, index))
        # don't emit selection change to avoid double call of commit function
        # it is already called from set_data
        with disconnected(
            self.doc_list.selectionModel().selectionChanged, self.selection_changed
        ):
            view.selectionModel().select(selection, QItemSelectionModel.ClearAndSelect)

    def selection_changed(self) -> None:
        """Function is called every time the selection changes"""
        self.selected_documents = self.get_selected_indexes()
        self.show_docs()
        self.commit.deferred()

    def show_docs(self):
        """Show the selected documents in the right area"""
        if self.corpus is None:
            return

        self.Warning.no_feats_display.clear()
        if len(self.display_features) == 0:
            self.Warning.no_feats_display()

        if self.show_tokens:
            tokens = list(self.corpus.ngrams_iterator(include_postags=True))

        parts = []
        for doc_count, c_index in enumerate(sorted(self.selected_documents)):
            text = ""
            for feature in self.display_features:
                value = str(self.corpus[c_index, feature.name])
                if feature in self.search_features:
                    value = self.__mark_text(value)
                value = value.replace("\n", "<br/>")
                is_image = feature.attributes.get("type", "") == "image"
                if is_image and value != "?":
                    value = os.path.join(feature.attributes.get("origin", ""), value)
                    value = '<img src="{}"></img>'.format(value)
                text += (
                    f'<tr><td class="variables"><strong>{feature.name}:</strong></td>'
                    f'<td class="content">{value}</td></tr>'
                )

            if self.show_tokens:
                tokens_ = "".join(
                    f'<span class="token">{token}</span>' for token in tokens[c_index]
                )
                text += (
                    f'<tr><td class="variables"><strong>Tokens & Tags:</strong></td>'
                    f"<td>{tokens_}</td></tr>"
                )
            parts.append(text)

        joined = SEPARATOR.join(parts)
        html = f"<table>{joined}</table>"
        base = QUrl.fromLocalFile(__file__)
        self.doc_webview.setHtml(HTML.format(html), base)

    def __mark_text(self, text):
        search_keyword = self.regexp_filter.strip("|")
        if not search_keyword:
            return text

        try:
            reg = re.compile(search_keyword, re.IGNORECASE | re.MULTILINE)
        except sre_constants.error:
            return text

        matches = list(reg.finditer(text))
        if not matches:
            return text

        text = list(text)
        for m in matches[::-1]:
            text[m.start() : m.end()] = list(
                f'<mark data-markjs="true">{"".join(text[m.start(): m.end()])}</mark>'
            )
        return "".join(text)

    @staticmethod
    def __get_selected_rows(view: QListView) -> List[Variable]:
        rows = view.selectionModel().selectedRows()
        values = view.model()[:]
        return [values[row.row()] for row in sorted(rows, key=lambda r: r.row())]

    def search_features_changed(self):
        self.search_features = self.__get_selected_rows(self.search_listbox)
        if self.corpus:
            self.doc_list_model.update_filter_content(self.regenerate_docs())
        self.doc_list.model().invalidateFilter()
        self.refresh_search()

    def display_features_changed(self):
        self.display_features = self.__get_selected_rows(self.display_listbox)
        self.show_docs()

    def regenerate_docs(self) -> List[str]:
        self.Warning.no_feats_search.clear()
        if len(self.search_features) == 0:
            self.Warning.no_feats_search()
        return self.corpus.documents_from_features(self.search_features)

    def refresh_search(self):
        self.Error.invalid_regex.clear()
        if self.corpus is not None:
            self.doc_list.model().set_filter_string(self.regexp_filter)
            if not self.selected_documents:
                # when currently selected items are filtered selection is empty
                # select first element in the view in that case
                self.doc_list.setCurrentIndex(self.doc_list.model().index(0, 0))
            self.update_info()
            try:
                self.compiled_regex = re.compile(self.regexp_filter.strip("|"), re.IGNORECASE)
                self.start(
                    _count_matches,
                    self.doc_list_model.get_filter_content(),
                    self.compiled_regex,
                )
            except re.error:
                self.Error.invalid_regex()
                self.compiled_regex = None 
                self.n_matching = "n/a"
                self.n_matches = "n/a"
                self.n_tokens = "n/a"
                self.n_types = "n/a"
                
            self.show_docs()
            self.commit.deferred()

    def on_done(self, res: int):
        """When matches count is done show the result in the label"""
        self.n_matches = res if res is not None else "n/a"

    def on_exception(self, ex):
        raise ex

    def update_info(self):
        if self.corpus is not None:
            has_tokens = self.corpus.has_tokens()
            self.n_matching = f"{self.doc_list.model().rowCount()}/{len(self.corpus)}"
            self.n_tokens = self.corpus.count_tokens() if has_tokens else "n/a"
            self.n_types = self.corpus.count_unique_tokens() if has_tokens else "n/a"
        else:
            self.n_matching = "n/a"
            self.n_matches = "n/a"
            self.n_tokens = "n/a"
            self.n_types = "n/a"

    @gui.deferred
    def commit(self):
        matched = unmatched = annotated_corpus = None
        if self.corpus is not None:
            selected_docs = sorted(self.get_selected_indexes())
            matched = self.corpus[selected_docs] if selected_docs else None
            mask = np.ones(len(self.corpus), bool)
            mask[selected_docs] = 0
            unmatched = self.corpus[mask] if mask.any() else None
            annotated_corpus = create_annotated_table(self.corpus, selected_docs)
        self.Outputs.matching_docs.send(matched)
        self.Outputs.other_docs.send(unmatched)
        self.Outputs.corpus.send(annotated_corpus)

    def send_report(self):
        self.report_items(
            (
                ("Query", self.regexp_filter),
                ("Matching documents", self.n_matching),
                ("Matches", self.n_matches),
            )
        )

    def showEvent(self, event):
        super().showEvent(event)
        self.update_splitter()

    def update_splitter(self):
        """
        Update splitter that document list on the left never take more
        than 1/3 of the space. It is only set on showEvent. If user
        later changes sizes it stays as it is.
        """
        w1, w2 = self.splitter.sizes()
        ws = w1 + w2
        if w2 < 2 / 3 * ws:
            self.splitter.setSizes([int(ws * 1 / 3), int(ws * 2 / 3)])

    @classmethod
    def migrate_context(cls, context, version):
        if version < 2:
            f_order = context.values.pop("display_features", None)
            display_idx = context.values.pop("display_indices", [])
            search_ids = context.values.pop("search_indices", [])
            if f_order is not None:
                f_order = f_order[0]
                display_features = [f_order[i] for i in display_idx if i < len(f_order)]
                search_features = [f_order[i] for i in search_ids if i < len(f_order)]
                context.values["display_features"] = (display_features, -3)
                context.values["search_features"] = (search_features, -3)

            # old widget used PerfectDomainContextHandler with MATCH_VALUES_ALL
            # now it uses DomainContextHandler. The difference are:
            # - perfect handler stores values in tuple while domain in dicts
            # - domain context handler store class_vars together with attributes
            #   while perfect handler store them separately
            # - since MATCH_VALUES_ALL was used discrete var values were stored
            #   with var name (replacing them with id for discrete var - 1)
            if hasattr(context, "class_vars"):
                context.attributes = {
                    attr: 1 if isinstance(v, list) else v
                    for attr, v in context.attributes + context.class_vars
                }
                context.metas = dict(context.metas)
                delattr(context, "class_vars")


if __name__ == "__main__":
    from orangewidget.utils.widgetpreview import WidgetPreview

    from orangecontrib.text.preprocess import BASE_TOKENIZER

    corpus_ = Corpus.from_file("book-excerpts")
    corpus_ = corpus_[:3]
    corpus_ = BASE_TOKENIZER(corpus_)
    WidgetPreview(OWCorpusViewer).run(corpus_)
