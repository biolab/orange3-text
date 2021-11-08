import re
from types import SimpleNamespace
from typing import Optional, Any, List, Tuple

import numpy as np

from AnyQt.QtCore import Qt, QUrl, QItemSelection, QItemSelectionModel, \
    QModelIndex
from AnyQt.QtWidgets import QTableView, QSplitter, QApplication

from Orange.data import Table, Domain, StringVariable
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.annotated_data import create_annotated_table
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.utils.itemmodels import PyTableModel
from Orange.widgets.widget import Input, Output, OWWidget, Msg

from orangecontrib.text import Corpus
from orangecontrib.text.semantic_search import SemanticSearch

IndexRole = next(gui.OrangeUserRole)

WORDS_COLUMN_NAME = "Words"
HTML = '''
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

mark {{ 
    background: #FFCD28;
}}
body {{
    font-family: Helvetica;
    font-size: 10pt;
}}
hr {{
    border: none;
    height: 1px;
    background-color: #000;

}}

</style>
</head>
<body>
{}
</body>
</html>
'''


class Results(SimpleNamespace):
    scores: List[Optional[List]] = []


def run(
        corpus: Optional[Corpus],
        words: Optional[List],
        state: TaskState
) -> Results:
    results = Results(scores=[])
    if not corpus or not words:
        return results

    def callback(i: float, status=""):
        state.set_progress_value(i * 100)
        if status:
            state.set_status(status)
        if state.is_interruption_requested():
            raise Exception

    callback(0, "Calculating...")
    semantic_search = SemanticSearch()
    results.scores = semantic_search(corpus.documents, words, callback)
    return results


class SemanticListView(QTableView):
    def __init__(self):
        super().__init__(
            sortingEnabled=True,
            editTriggers=QTableView.NoEditTriggers,
            selectionBehavior=QTableView.SelectRows,
            selectionMode=QTableView.ExtendedSelection,
            cornerButtonEnabled=False,
            alternatingRowColors=True
        )
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setDefaultSectionSize(22)
        self.verticalHeader().hide()


class DisplayDocument:
    Document, Section, Sentence = range(3)
    ITEMS = ["Document", "Section", "Sentence"]
    START_TAG, END_TAG = "<mark data-markjs='true'>", "</mark>"
    TAG = f"{START_TAG}{{}}{END_TAG}"
    SECTION_SEP = "\n"
    REP = "..."

    def __init__(self, display_type: int):
        self.__type = display_type

    def __call__(self, text: str, matches: List[Tuple]) -> str:
        if self.__type == self.Document:
            return self._tag_text(text, matches)

        elif self.__type == self.Section:
            tagged = self._tag_text(text, matches)
            tagged = re.sub(f"{self.SECTION_SEP}+", self.SECTION_SEP, tagged)
            sections = tagged.split(self.SECTION_SEP)
            replaced_sections = self._replace_sections(sections)
            purged_sections = self._purge(replaced_sections)
            return self.SECTION_SEP.join(purged_sections)

        elif self.__type == self.Sentence:
            if not matches and text:
                return self.REP

            sentences = self._replace_sentences(text, matches)
            return " ".join(sentences)

        else:
            raise NotImplementedError

    @staticmethod
    def _purge(collection: List[str]) -> List[str]:
        purged = []
        add_rep = True
        for text in collection:
            if text != DisplayDocument.REP or add_rep:
                purged.append(text)
            add_rep = text != DisplayDocument.REP
        return purged

    @staticmethod
    def _replace_sections(sections: List[str]) -> List[str]:
        start_tag, end_tag = DisplayDocument.START_TAG, DisplayDocument.END_TAG
        opened = False
        for i, section in enumerate(sections):
            if section.count(start_tag) != section.count(end_tag):
                opened = section.count(start_tag) > section.count(end_tag)
            elif not opened and not section.count(end_tag):
                sections[i] = DisplayDocument.REP
        return sections

    @staticmethod
    def _replace_sentences(text: str, matches: List[Tuple]) -> List[str]:
        def replace_unmatched(ind_start, ind_end):
            stripped = text[ind_start: ind_end].strip(" ")
            if stripped[:1] == DisplayDocument.SECTION_SEP:
                sentences.append(DisplayDocument.SECTION_SEP)
            if ind_end - ind_start > 1:
                sentences.append(DisplayDocument.REP)
                if DisplayDocument.SECTION_SEP in stripped[1:]:
                    sentences.append(DisplayDocument.SECTION_SEP)

        sentences = []
        end = 0
        for start_, end_ in matches:
            replace_unmatched(end, start_)
            sentence = text[start_: end_]
            sentences.append(DisplayDocument.TAG.format(sentence))
            end = end_

        replace_unmatched(matches[-1][1], len(text))
        return sentences

    @staticmethod
    def _tag_text(text: str, matches: List[Tuple]) -> str:
        text = list(text)

        for start, end in matches[::-1]:
            text[start: end] = list(
                DisplayDocument.TAG.format("".join(text[start:end]))
            )
        return "".join(text)


class OWSemanticViewer(OWWidget, ConcurrentWidgetMixin):
    name = "Semantic Viewer"
    description = "Infers characteristic words from the input corpus."
    icon = "icons/SemanticViewer.svg"
    priority = 1120
    keywords = ["search"]

    class Inputs:
        corpus = Input("Corpus", Corpus, default=True)
        words = Input("Words", Table)

    class Outputs:
        matching_docs = Output("Matching Docs", Corpus, default=True)
        other_docs = Output("Other Docs", Corpus)
        corpus = Output("Corpus", Corpus)

    class Warning(OWWidget.Warning):
        no_words_column = Msg("Input is missing 'Words' column.")

    threshold = Setting(0.5)
    display_index = Setting(DisplayDocument.Document)
    selection = Setting([], schema_only=True)

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        self.corpus: Optional[Corpus] = None
        self.words: Optional[List] = None
        self._results: Optional[Results] = None
        self.__pending_selection = self.selection
        self._setup_gui()

    def _setup_gui(self):
        # Control area
        box = gui.hBox(self.controlArea, "Filtering")
        gui.doubleSpin(box, self, "threshold", 0, 1, 0.01, None,
                       label="Threshold: ", orientation=Qt.Horizontal,
                       callback=self.__on_threshold_changed)

        box = gui.hBox(self.controlArea, "Display")
        gui.radioButtons(box, self, "display_index", DisplayDocument.ITEMS,
                         callback=self.__on_display_changed)

        gui.rubber(self.controlArea)

        # Main area
        model = PyTableModel(parent=self)
        self._list_view = SemanticListView()
        self._list_view.setModel(model)
        self._list_view.selectionModel().selectionChanged.connect(
            self.__on_selection_changed
        )
        self._list_view.horizontalHeader().sectionClicked.connect(
            self.__on_horizontal_header_clicked
        )

        splitter = QSplitter()
        splitter.addWidget(self._list_view)
        self._web_view = gui.WebviewWidget(splitter, debug=False)
        splitter.setSizes([200, 300])
        self.mainArea.layout().addWidget(splitter)

    def __on_threshold_changed(self):
        self._show_documents()

    def __on_display_changed(self):
        self._show_documents()

    def __on_selection_changed(self):
        self.selection = self._get_selected_indices()
        self._show_documents()
        self.commit()

    def __on_horizontal_header_clicked(self):
        self.selection = self._get_selected_indices()
        self._show_documents()

    @Inputs.corpus
    def set_corpus(self, corpus: Optional[Corpus]):
        self.corpus = corpus

    def _clear(self):
        self.cancel()
        self._results = None
        self.selection = []
        self._list_view.model().clear()
        self._web_view.setHtml("")

    @Inputs.words
    def set_words(self, words: Optional[Table]):
        self.words = None
        self.Warning.no_words_column.clear()
        if words:
            if WORDS_COLUMN_NAME in words.domain and words.domain[
                    WORDS_COLUMN_NAME].attributes.get("type") == "words":
                self.words = list(words.get_column_view(WORDS_COLUMN_NAME)[0])
            else:
                self.Warning.no_words_column()

    def handleNewSignals(self):
        self._clear()
        self.update_scores()

    def update_scores(self):
        self.start(run, self.corpus, self.words)

    def on_exception(self, ex: Exception):
        raise ex

    def on_partial_result(self, _: Any):
        pass

    # pylint: disable=arguments-differ
    def on_done(self, results: Results):
        # self._apply_sorting()
        self._results = results.scores

        if not self._results or not self.corpus or not self.words:
            self.commit()
            return

        model = self._list_view.model()
        model.setHorizontalHeaderLabels(["Match", "Score", "Document"])

        def get_avg_score(result: List) -> float:
            return "NA" if result is None else np.mean([r[1] for r in result])

        def get_n_matches(ngram):
            return sum(ngram.count(word) for word in self.words)

        data = [[get_n_matches(ngram), get_avg_score(res), title]
                for res, title, ngram in zip(self._results,
                                             self.corpus.titles.tolist(),
                                             self.corpus.ngrams)]
        model.wrap(data)
        for i in range(len(data)):
            model.setData(model.index(i, 0), i, role=IndexRole)
        self._list_view.setColumnWidth(0, 65)
        self._list_view.setColumnWidth(1, 65)

        self.select_documents()

    def select_documents(self):
        self.selection = self.__pending_selection or [0]
        self.__pending_selection = []
        self._set_selected_rows(self.selection)

    def _get_selected_indices(self) -> List[int]:
        selection_model = self._list_view.selectionModel()
        model = self._list_view.model()
        rows = []
        for row in range(selection_model.model().rowCount()):
            if selection_model.isRowSelected(row, QModelIndex()):
                rows.append(model.data(model.index(row, 0), role=IndexRole))
        return rows

    def _set_selected_rows(self, selected_rows: List[int]):
        model = self._list_view.model()
        n_columns = model.columnCount()
        selection = QItemSelection()
        for i in selected_rows:
            _selection = QItemSelection(model.index(i, 0),
                                        model.index(i, n_columns - 1))
            selection.merge(_selection, QItemSelectionModel.Select)
        self._list_view.selectionModel().select(
            selection, QItemSelectionModel.ClearAndSelect
        )

    def _show_documents(self):
        if self.corpus is None or self._results is None:
            return

        documents = self.corpus.documents
        parser = DisplayDocument(self.display_index)
        htmls = []
        for doc_index in self.selection:
            text = documents[doc_index]
            matches = [ind for ind, score in self._results[doc_index] or []
                       if score >= self.threshold]
            text = parser(text, matches)
            text = text.replace("\n", "<br/>")
            html = f"<p>{text}</p>"
            htmls.append(html)

        html = "<hr>".join(htmls)
        base = QUrl.fromLocalFile(__file__)
        self._web_view.setHtml(HTML.format(html), base)

    def onDeleteWidget(self):
        self.shutdown()
        super().onDeleteWidget()

    def commit(self):
        matched = other = annotated = None
        if self.corpus:
            mask = np.zeros(len(self.corpus), dtype=bool)
            mask[self.selection] = True
            matched = self.corpus[mask] if sum(mask) else None
            other = self.corpus[~mask] if sum(~mask) else None
            annotated = create_annotated_table(self.corpus, self.selection)
        self.Outputs.matching_docs.send(matched)
        self.Outputs.other_docs.send(other)
        self.Outputs.corpus.send(annotated)

    def send_report(self):
        if not self.corpus:
            return
        self.report_data("Corpus", self.corpus)
        if self.words is not None:
            self.report_paragraph("Words", ", ".join(self.words))
            self.report_table(self._list_view, num_format="{:.3f}")

    def copy_to_clipboard(self):
        text = self._web_view.selectedText()
        QApplication.clipboard().setText(text)


if __name__ == "__main__":
    # pylint: disable=ungrouped-imports
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    words_var_ = StringVariable(WORDS_COLUMN_NAME)
    words_var_.attributes = {"type": "words"}
    lists = [[w] for w in ["human", "graph", "minors", "trees"]]
    words_ = Table.from_list(Domain([], metas=[words_var_]), lists)
    words_.name = "Words"
    WidgetPreview(OWSemanticViewer).run(
        set_corpus=Corpus.from_file("deerwester"),  # deerwester book-excerpts
        set_words=words_
    )
