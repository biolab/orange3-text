# pylint: disable=missing-docstring
from types import SimpleNamespace
from typing import Optional, Set, List, Tuple, Dict, Any

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt, QSortFilterProxyModel, QItemSelection, \
    QItemSelectionModel, QModelIndex, Signal
from AnyQt.QtWidgets import QCheckBox, QLineEdit, QTableView, QGridLayout, \
    QRadioButton, QButtonGroup

from Orange.data import Table, Domain, StringVariable, ContinuousVariable
from Orange.util import wrap_callback
from Orange.widgets import gui
from Orange.widgets.settings import DomainContextHandler, ContextSetting, \
    Setting
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.utils.itemmodels import PyTableModel, TableModel
from Orange.widgets.widget import Input, Output, OWWidget, Msg

from orangecontrib.text import Corpus
from orangecontrib.text.keywords import ScoringMethods, AggregationMethods, \
    YAKE_LANGUAGE_MAPPING, RAKE_LANGUAGES, EMBEDDING_LANGUAGE_MAPPING
from orangecontrib.text.preprocess import BaseNormalizer

WORDS_COLUMN_NAME = "Words"
YAKE_LANGUAGES = list(YAKE_LANGUAGE_MAPPING.keys())
EMBEDDING_LANGUAGES = list(EMBEDDING_LANGUAGE_MAPPING.keys())


class Results(SimpleNamespace):
    # currently wanted (aggregated) scores
    scores: List[Tuple[Any, ...]] = []
    # labels for currently wanted scores
    labels: List[str] = []
    # all calculated keywords {method: [[(word1, score1), ...]]}
    all_keywords: Dict[str, List[List[Tuple[str, float]]]] = {}


def run(
        corpus: Optional[Corpus],
        words: Optional[List],
        cached_keywords: Dict,
        scoring_methods: Set,
        scoring_methods_kwargs: Dict,
        agg_method: int,
        state: TaskState
) -> Results:
    results = Results(scores=[], labels=[], all_keywords={})
    if not corpus:
        return results

    # passed by reference (and not copied) - to save partial results
    results.all_keywords = cached_keywords
    if not scoring_methods:
        return results

    def callback(i: float, status=""):
        state.set_progress_value(i * 100)
        if status:
            state.set_status(status)
        if state.is_interruption_requested():
            raise Exception

    callback(0, "Calculating...")
    scores = {}
    tokens = corpus.tokens
    documents = corpus.documents
    step = 1 / len(scoring_methods)
    for method_name, func in ScoringMethods.ITEMS:
        if method_name in scoring_methods:
            if method_name not in results.all_keywords:
                i = len(results.labels)
                cb = wrap_callback(callback, start=i * step,
                                   end=(i + 1) * step)

                needs_tokens = method_name in ScoringMethods.TOKEN_METHODS
                kw = {"progress_callback": cb}
                kw.update(scoring_methods_kwargs.get(method_name, {}))

                keywords = func(tokens if needs_tokens else documents, **kw)
                results.all_keywords[method_name] = keywords

            keywords = results.all_keywords[method_name]
            scores[method_name] = \
                dict(AggregationMethods.aggregate(keywords, agg_method))

            results.labels.append(method_name)

    scores = pd.DataFrame(scores)
    if words:

        # Normalize words
        for preprocessor in corpus.used_preprocessor.preprocessors:
            if isinstance(preprocessor, BaseNormalizer):
                words = [preprocessor.normalizer(w) for w in words]

        # Filter scores using words
        existing_words = [w for w in set(words) if w in scores.index]
        scores = scores.loc[existing_words] if existing_words \
            else scores.iloc[:0]

    results.scores = scores.reset_index().sort_values(
        by=[results.labels[0], "index"],
        ascending=[False, True]
    ).values.tolist()

    return results


class SelectionMethods:
    NONE, ALL, MANUAL, N_BEST = range(4)
    ITEMS = "None", "All", "Manual", "Top words"


class KeywordsTableView(QTableView):
    pressedAny = Signal()

    def __init__(self):
        super().__init__(
            sortingEnabled=True,
            editTriggers=QTableView.NoEditTriggers,
            selectionBehavior=QTableView.SelectRows,
            selectionMode=QTableView.ExtendedSelection,
            cornerButtonEnabled=False,
        )
        self.setItemDelegate(gui.ColoredBarItemDelegate(self))
        self.verticalHeader().setDefaultSectionSize(22)
        self.verticalHeader().hide()

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.pressedAny.emit()


class KeywordsTableModel(PyTableModel):
    def data(self, index, role=Qt.DisplayRole):
        if role in (gui.BarRatioRole, Qt.DisplayRole):
            return super().data(index, Qt.EditRole)
        if role == Qt.BackgroundColorRole and index.column() == 0:
            return TableModel.ColorForRole[TableModel.Meta]
        return super().data(index, role)

    def _argsortData(self, data, order):
        """Always sort NaNs last"""
        indices = np.argsort(data, kind='mergesort')
        if order == Qt.DescendingOrder:
            return np.roll(indices[::-1], -np.isnan(data).sum())
        return indices


class SortFilterProxyModel(QSortFilterProxyModel):
    def sort(self, column: int, order: Qt.SortOrder = Qt.AscendingOrder):
        self.lessThan = lambda *args: self.__nan_less_than(*args, order=order)
        super().sort(column, order)

    def __nan_less_than(self, left_ind: QModelIndex, right_ind: QModelIndex,
                        order: Qt.SortOrder = Qt.AscendingOrder) -> bool:
        left = self.sourceModel().data(left_ind, role=Qt.EditRole)
        right = self.sourceModel().data(right_ind, role=Qt.EditRole)
        if isinstance(right, float) and isinstance(left, float):
            # NaNs always at the end
            if np.isnan(right):
                right = 1 - order
            if np.isnan(left):
                left = 1 - order
            return left < right
        return super().lessThan(left_ind, right_ind)


class OWKeywords(OWWidget, ConcurrentWidgetMixin):
    name = "Extract Keywords"
    description = "Infers characteristic words from the input corpus."
    icon = "icons/Keywords.svg"
    priority = 1100
    keywords = ["characteristic", "term"]

    buttons_area_orientation = Qt.Vertical

    DEFAULT_SORTING = (1, Qt.DescendingOrder)

    settingsHandler = DomainContextHandler()
    selected_scoring_methods: Set[str] = Setting({ScoringMethods.TF_IDF})
    yake_lang_index: int = Setting(YAKE_LANGUAGES.index("English"))
    rake_lang_index: int = Setting(RAKE_LANGUAGES.index("English"))
    embedding_lang_index: int = Setting(EMBEDDING_LANGUAGES.index("English"))
    agg_method: int = Setting(AggregationMethods.MEAN)
    sel_method: int = ContextSetting(SelectionMethods.N_BEST)
    n_selected: int = ContextSetting(3)
    sort_column_order: Tuple[int, int] = Setting(DEFAULT_SORTING)
    selected_words = ContextSetting([], schema_only=True)
    auto_apply: bool = Setting(True)

    class Inputs:
        corpus = Input("Corpus", Corpus, default=True)
        words = Input("Words", Table)

    class Outputs:
        words = Output("Words", Corpus)

    class Warning(OWWidget.Warning):
        no_words_column = Msg("Input is missing 'Words' column.")

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        self.corpus: Optional[Corpus] = None
        self.words: Optional[List] = None
        self.__cached_keywords = {}
        self.model = KeywordsTableModel(parent=self)
        self._setup_gui()

    def _setup_gui(self):
        grid = QGridLayout()
        box = gui.widgetBox(self.controlArea, "Scoring Methods", grid)

        yake_cb = gui.comboBox(
            self.controlArea, self, "yake_lang_index", items=YAKE_LANGUAGES,
            callback=self.__on_yake_lang_changed
        )
        rake_cb = gui.comboBox(
            self.controlArea, self, "rake_lang_index", items=RAKE_LANGUAGES,
            callback=self.__on_rake_lang_changed
        )
        embedding_cb = gui.comboBox(
            self.controlArea, self, "embedding_lang_index",
            items=EMBEDDING_LANGUAGES, callback=self.__on_emb_lang_changed
        )

        for i, (method_name, _) in enumerate(ScoringMethods.ITEMS):
            check_box = QCheckBox(method_name, self)
            check_box.setChecked(method_name in self.selected_scoring_methods)
            check_box.stateChanged.connect(
                lambda state, name=method_name:
                self.__on_scoring_method_state_changed(state, name)
            )
            box.layout().addWidget(check_box, i, 0)
            if method_name == ScoringMethods.YAKE:
                box.layout().addWidget(yake_cb, i, 1)
            if method_name == ScoringMethods.RAKE:
                box.layout().addWidget(rake_cb, i, 1)
            if method_name == ScoringMethods.EMBEDDING:
                box.layout().addWidget(embedding_cb, i, 1)

        box = gui.vBox(self.controlArea, "Aggregation")
        gui.comboBox(
            box, self, "agg_method", items=AggregationMethods.ITEMS,
            callback=self.update_scores
        )

        box = gui.vBox(self.buttonsArea, "Select Words")
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        box.layout().addLayout(grid)

        self.__sel_method_buttons = QButtonGroup()
        for method, label in enumerate(SelectionMethods.ITEMS):
            button = QRadioButton(label)
            button.setChecked(method == self.sel_method)
            grid.addWidget(button, method, 0)
            self.__sel_method_buttons.addButton(button, method)
        self.__sel_method_buttons.buttonClicked[int].connect(
            self._set_selection_method
        )

        spin = gui.spin(
            box, self, "n_selected", 1, 999, addToLayout=False,
            callback=lambda: self._set_selection_method(
                SelectionMethods.N_BEST)
        )
        grid.addWidget(spin, 3, 1)

        gui.rubber(self.controlArea)
        gui.auto_send(self.buttonsArea, self, "auto_apply")

        self.__filter_line_edit = QLineEdit(
            textChanged=self.__on_filter_changed,
            placeholderText="Filter..."
        )
        self.mainArea.layout().addWidget(self.__filter_line_edit)

        def select_manual():
            self._set_selection_method(SelectionMethods.MANUAL)

        self.view = KeywordsTableView()
        self.view.pressedAny.connect(select_manual)
        self.view.horizontalHeader().setSortIndicator(*self.DEFAULT_SORTING)
        self.view.horizontalHeader().sectionClicked.connect(
            self.__on_horizontal_header_clicked)
        self.mainArea.layout().addWidget(self.view)

        proxy = SortFilterProxyModel()
        proxy.setFilterKeyColumn(0)
        proxy.setFilterCaseSensitivity(False)
        self.view.setModel(proxy)
        self.view.model().setSourceModel(self.model)
        self.view.selectionModel().selectionChanged.connect(
            self.__on_selection_changed
        )

    def __on_scoring_method_state_changed(self, state: int, method_name: str):
        if state == Qt.Checked:
            self.selected_scoring_methods.add(method_name)
        elif method_name in self.selected_scoring_methods:
            self.selected_scoring_methods.remove(method_name)
        self.update_scores()

    def __on_yake_lang_changed(self):
        if ScoringMethods.YAKE in self.selected_scoring_methods:
            if ScoringMethods.YAKE in self.__cached_keywords:
                del self.__cached_keywords[ScoringMethods.YAKE]
            self.update_scores()

    def __on_rake_lang_changed(self):
        if ScoringMethods.RAKE in self.selected_scoring_methods:
            if ScoringMethods.RAKE in self.__cached_keywords:
                del self.__cached_keywords[ScoringMethods.RAKE]
            self.update_scores()

    def __on_emb_lang_changed(self):
        if ScoringMethods.EMBEDDING in self.selected_scoring_methods:
            if ScoringMethods.EMBEDDING in self.__cached_keywords:
                del self.__cached_keywords[ScoringMethods.EMBEDDING]
            self.update_scores()

    def __on_filter_changed(self):
        model = self.view.model()
        model.setFilterFixedString(self.__filter_line_edit.text().strip())
        self._select_rows()

    def __on_horizontal_header_clicked(self, index: int):
        header = self.view.horizontalHeader()
        self.sort_column_order = (index, header.sortIndicatorOrder())
        self._select_rows()
        # explicitly call commit, because __on_selection_changed will not be
        # invoked, since selection is actually the same, only order is not
        if self.sel_method == SelectionMethods.MANUAL and self.selected_words \
                or self.sel_method == SelectionMethods.ALL:
            self.commit()

    def __on_selection_changed(self):
        selected_rows = self.view.selectionModel().selectedRows(0)
        model = self.view.model()
        self.selected_words = [model.data(model.index(i.row(), 0))
                               for i in selected_rows]
        self.commit()

    @Inputs.corpus
    def set_corpus(self, corpus: Optional[Corpus]):
        self.closeContext()
        self._clear()
        self.corpus = corpus
        self.openContext(self.corpus)
        self.__sel_method_buttons.button(self.sel_method).setChecked(True)

    def _clear(self):
        self.clear_messages()
        self.cancel()
        self.selected_words = []
        self.model.clear()
        self.__cached_keywords = {}

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
        self.update_scores()

    def update_scores(self):
        kwargs = {
            ScoringMethods.YAKE: {
                "language": YAKE_LANGUAGES[self.yake_lang_index],
                "max_len": self.corpus.ngram_range[1] if self.corpus else 1
            },
            ScoringMethods.RAKE: {
                "language": RAKE_LANGUAGES[self.rake_lang_index],
                "max_len": self.corpus.ngram_range[1] if self.corpus else 1
            },
            ScoringMethods.EMBEDDING: {
                "language": EMBEDDING_LANGUAGES[self.embedding_lang_index],
            },
        }
        self.start(run, self.corpus, self.words, self.__cached_keywords,
                   self.selected_scoring_methods, kwargs, self.agg_method)

    def _set_selection_method(self, method: int):
        self.sel_method = method
        self.__sel_method_buttons.button(method).setChecked(True)
        self._select_rows()

    def _select_rows(self):
        model = self.view.model()
        n_rows, n_columns = model.rowCount(), model.columnCount()
        if self.sel_method == SelectionMethods.NONE:
            selection = QItemSelection()
        elif self.sel_method == SelectionMethods.ALL:
            selection = QItemSelection(
                model.index(0, 0),
                model.index(n_rows - 1, n_columns - 1)
            )
        elif self.sel_method == SelectionMethods.MANUAL:
            selection = QItemSelection()
            for i in range(n_rows):
                word = model.data(model.index(i, 0))
                if word in self.selected_words:
                    _selection = QItemSelection(model.index(i, 0),
                                                model.index(i, n_columns - 1))
                    selection.merge(_selection, QItemSelectionModel.Select)
        elif self.sel_method == SelectionMethods.N_BEST:
            n_sel = min(self.n_selected, n_rows)
            selection = QItemSelection(
                model.index(0, 0),
                model.index(n_sel - 1, n_columns - 1)
            )
        else:
            raise NotImplementedError

        self.view.selectionModel().select(
            selection, QItemSelectionModel.ClearAndSelect
        )

    def on_exception(self, ex: Exception):
        raise ex

    def on_partial_result(self, _: Any):
        pass

    # pylint: disable=arguments-differ
    def on_done(self, results: Results):
        self.__cached_keywords = results.all_keywords
        self.model.wrap(results.scores)
        self.model.setHorizontalHeaderLabels(["Word"] + results.labels)
        self._apply_sorting()
        if self.model.rowCount() > 0:
            self._select_rows()
        else:
            self.__on_selection_changed()

    def _apply_sorting(self):
        if self.model.columnCount() <= self.sort_column_order[0]:
            self.sort_column_order = self.DEFAULT_SORTING

        header = self.view.horizontalHeader()
        current_sorting = (header.sortIndicatorSection(),
                           header.sortIndicatorOrder())
        if current_sorting != self.sort_column_order:
            header.setSortIndicator(*self.sort_column_order)
            # needed to sort nans; 1. column has strings
            # if self.sort_column_order[0] > 0:
            #     self.model.sort(*self.sort_column_order)

    def onDeleteWidget(self):
        self.shutdown()
        super().onDeleteWidget()

    def commit(self):
        words = None
        if self.selected_words:
            words_var = StringVariable(WORDS_COLUMN_NAME)
            words_var.attributes = {"type": "words"}
            model = self.model
            attrs = [ContinuousVariable(model.headerData(i + 1, Qt.Horizontal))
                     for i in range(len(self.selected_scoring_methods))]
            domain = Domain(attrs, metas=[words_var])

            sort_column, reverse = self.sort_column_order
            data = sorted(model, key=lambda a: a[sort_column], reverse=reverse)
            data = [s[1:] + s[:1] for s in data if s[0] in self.selected_words]
            words = Table.from_list(domain, data)
            words.name = "Words"

        self.Outputs.words.send(words)

    def send_report(self):
        if not self.corpus:
            return
        self.report_data("Corpus", self.corpus)
        if self.words is not None:
            self.report_paragraph("Words", ", ".join(self.words))
        self.report_table("Keywords", self.view, num_format="{:.3f}")


if __name__ == "__main__":
    # pylint: disable=ungrouped-imports
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    words_var_ = StringVariable(WORDS_COLUMN_NAME)
    words_var_.attributes = {"type": "words"}
    lists = [[w] for w in ["human", "graph", "minors", "trees"]]
    words_ = Table.from_list(Domain([], metas=[words_var_]), lists)
    words_.name = "Words"
    WidgetPreview(OWKeywords).run(
        set_corpus=Corpus.from_file("deerwester"),  # deerwester book-excerpts
        # set_words=words_
    )
