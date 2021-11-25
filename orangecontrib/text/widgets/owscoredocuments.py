import re
from collections import Counter
from contextlib import contextmanager
from inspect import signature
from typing import Callable, List, Tuple, Union

import numpy as np
from AnyQt.QtCore import (
    QItemSelection,
    QItemSelectionModel,
    QSortFilterProxyModel,
    Qt,
    Signal,
)
from AnyQt.QtWidgets import (
    QButtonGroup,
    QGridLayout,
    QHeaderView,
    QLineEdit,
    QRadioButton,
    QTableView,
)
from Orange.data.util import get_unique_names
from pandas import isnull
from sklearn.metrics.pairwise import cosine_similarity

# todo: uncomment when minimum version of Orange is 3.29.2
# from orangecanvas.gui.utils import disconnected
from orangewidget import gui
from Orange.data import ContinuousVariable, Domain, StringVariable, Table
from Orange.util import wrap_callback
from Orange.widgets.settings import ContextSetting, PerfectDomainContextHandler, Setting
from Orange.widgets.utils.annotated_data import create_annotated_table
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.utils.itemmodels import PyTableModel, TableModel
from Orange.widgets.widget import Input, Msg, Output, OWWidget

from orangecontrib.text import Corpus
from orangecontrib.text.preprocess import BaseNormalizer, BaseTransformer
from orangecontrib.text.vectorization.document_embedder import (
    LANGS_TO_ISO,
    DocumentEmbedder,
)


# todo: remove when minimum version of Orange is 3.29.2
@contextmanager
def disconnected(signal, slot, type=Qt.UniqueConnection):
    signal.disconnect(slot)
    try:
        yield
    finally:
        signal.connect(slot, type)


def _word_frequency(corpus: Corpus, words: List[str], callback: Callable) -> np.ndarray:
    res = []
    tokens = corpus.tokens
    for i, t in enumerate(tokens):
        counts = Counter(t)
        res.append([counts.get(w, 0) for w in words])
        callback((i + 1) / len(tokens))
    return np.array(res)


def _word_appearance(
    corpus: Corpus, words: List[str], callback: Callable
) -> np.ndarray:
    res = []
    tokens = corpus.tokens
    for i, t in enumerate(tokens):
        t = set(t)
        res.append([w in t for w in words])
        callback((i + 1) / len(tokens))
    return np.array(res)


def _embedding_similarity(
    corpus: Corpus,
    words: List[str],
    callback: Callable,
    embedding_language: str,
) -> np.ndarray:
    ticks = iter(np.linspace(0, 0.8, len(corpus) + len(words)))

    # TODO: currently embedding report success unify them to report progress float
    def emb_cb(sucess: bool):
        if sucess:
            callback(next(ticks))

    language = LANGS_TO_ISO[embedding_language]
    # make sure there will be only embeddings in X after calling the embedder
    corpus = Corpus.from_table(Domain([], metas=corpus.domain.metas), corpus)
    emb = DocumentEmbedder(language)
    documet_embeddings, skipped = emb(corpus, emb_cb)
    assert skipped is None
    word_embeddings = np.array(emb([[w] for w in words], emb_cb))
    return cosine_similarity(documet_embeddings.X, word_embeddings)


SCORING_METHODS = {
    # key: (Method's name, Method's function, Tooltip)
    "word_frequency": (
        "Word count",
        _word_frequency,
        "Frequency of the word in the document.",
    ),
    "word_appearance": (
        "Word presence",
        _word_appearance,
        "Score word with one if it appears in the document, with zero otherwise.",
    ),
    "embedding_similarity": (
        "Similarity",
        _embedding_similarity,
        "Cosine similarity between the document embedding and the word embedding.",
    ),
}

ADDITIONAL_OPTIONS = {
    "embedding_similarity": ("embedding_language", list(LANGS_TO_ISO.keys()))
}

AGGREGATIONS = {
    "Mean": np.mean,
    "Median": np.median,
    "Min": np.min,
    "Max": np.max,
}


def _preprocess_words(
    corpus: Corpus, words: List[str], callback: Callable
) -> List[str]:
    """
    Corpus's tokens can be preprocessed. Since they will not match correctly
    with words preprocessors that change words (e.g. normalization) must
    be applied to words too.
    """
    # workaround to preprocess words
    # TODO: currently preprocessors work only on corpus, when there will be more
    #  cases like this think about implementation of preprocessors for a list
    #  of strings
    words_feature = StringVariable("words")
    words_c = Corpus(
        Domain([], metas=[words_feature]),
        metas=np.array([[w] for w in words]),
        text_features=[words_feature],
    )
    # only transformers and normalizers preprocess on the word level
    pps = [
        pp
        for pp in corpus.used_preprocessor.preprocessors
        if isinstance(pp, (BaseTransformer, BaseNormalizer))
    ]
    for i, pp in enumerate(pps):
        words_c = pp(words_c)
        callback((i + 1) / len(pps))
    return [w[0] for w in words_c.tokens if len(w)]


def _run(
    corpus: Corpus,
    words: List[str],
    scoring_methods: List[str],
    aggregation: str,
    additional_params: dict,
    state: TaskState,
) -> None:
    """
    Perform word scoring with selected scoring methods

    Parameters
    ----------
    corpus
        Corpus of documents
    words
        List of words used for scoring
    scoring_methods
        Methods to score documents with
    aggregation
        Aggregation applied for each document on word scores
    additional_params
        Additional prameters for scores (e.g. embedding needs text language)
    state
        TaskState for reporting the task status and giving partial results
    """

    def callback(i: float) -> None:
        state.set_progress_value(i * 100)
        if state.is_interruption_requested():
            raise Exception

    cb_part = 1 / (len(scoring_methods) + 1)  # +1 for preprocessing

    words = _preprocess_words(corpus, words, wrap_callback(callback, end=cb_part))
    if len(words) == 0:
        raise Exception(
            "Empty word list after preprocessing. Please provide a valid set of words."
        )
    for i, sm in enumerate(scoring_methods):
        scoring_method = SCORING_METHODS[sm][1]
        sig = signature(scoring_method)
        add_params = {k: v for k, v in additional_params.items() if k in sig.parameters}
        scs = scoring_method(
            corpus,
            words,
            wrap_callback(callback, start=(i + 1) * cb_part, end=(i + 2) * cb_part),
            **add_params
        )
        scs = AGGREGATIONS[aggregation](scs, axis=1)
        state.set_partial_result((sm, aggregation, scs))


class SelectionMethods:
    NONE, ALL, MANUAL, N_BEST = range(4)
    ITEMS = "None", "All", "Manual", "Top documents"


class ScoreDocumentsTableView(QTableView):
    pressedAny = Signal()

    def __init__(self):
        super().__init__(
            sortingEnabled=True,
            editTriggers=QTableView.NoEditTriggers,
            selectionMode=QTableView.ExtendedSelection,
            selectionBehavior=QTableView.SelectRows,
            cornerButtonEnabled=False,
        )
        self.setItemDelegate(gui.ColoredBarItemDelegate(self))
        self.verticalHeader().setDefaultSectionSize(22)

    def update_column_widths(self) -> None:
        """
        Set columns widths such that each score column has width based on size
        hint and all scores columns have the same width.
        """
        header = self.horizontalHeader()
        col_width = max(
            [0]
            + [
                max(self.sizeHintForColumn(i), header.sectionSizeHint(i))
                for i in range(1, self.model().columnCount())
            ]
        )

        for i in range(1, self.model().columnCount()):
            header.resizeSection(i, col_width)
            header.setSectionResizeMode(i, QHeaderView.Fixed)

        # document title column is one that stretch
        header.setSectionResizeMode(0, QHeaderView.Stretch)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.pressedAny.emit()


class ScoreDocumentsProxyModel(QSortFilterProxyModel):
    @staticmethod
    def _convert(text: str) -> Union[str, int]:
        return int(text) if text.isdigit() else text.lower()

    @staticmethod
    def _alphanum_key(key: str) -> List[Union[str, int]]:
        return [ScoreDocumentsProxyModel._convert(c) for c in re.split("([0-9]+)", key)]

    def lessThan(self, left_ind, right_ind):
        """
        Sort strings of the first column naturally: Document 2 < Document 12
        """
        if left_ind.column() == 0 and right_ind.column() == 0:
            left = self.sourceModel().data(left_ind, role=Qt.DisplayRole)
            right = self.sourceModel().data(right_ind, role=Qt.DisplayRole)
            if left is not None and right is not None:
                return self._alphanum_key(left) < self._alphanum_key(right)
        return super().lessThan(left_ind, right_ind)


class ScoreDocumentsTableModel(PyTableModel):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._extremes = {}

    @staticmethod
    def simplify(s):
        """Remove tab and newline characters from the string"""
        for ch in "\n\t\r":
            s = s.replace(ch, " ")
        return s

    def data(self, index, role=Qt.DisplayRole):
        if index.column() > 0 and role == gui.BarRatioRole and index.isValid():
            # for all except first columns return ratio for distribution bar
            value = super().data(index, Qt.EditRole)
            vmin, vmax = self._extremes.get(index.column() - 1, (-np.inf, np.inf))
            return (value - vmin) / ((vmax - vmin) or 1)
        if role in (gui.BarRatioRole, Qt.DisplayRole):
            dat = super().data(index, Qt.EditRole)
            if role == Qt.DisplayRole and index.column() == 0:
                # in document title column remove newline characters from titles
                dat = self.simplify(dat)
            return dat
        if role == Qt.BackgroundColorRole and index.column() == 0:
            return TableModel.ColorForRole[TableModel.Meta]
        return super().data(index, role)

    def fill_table(self, titles, scores):
        """ Handle filling the table with titles and scores """
        for column, values in enumerate(scores.T):
            self.set_extremes(column, values)
        self.wrap([[c] + s for c, s in zip(titles, scores.tolist())])

    def set_extremes(self, column, values):
        """Set extremes for column's ratio bars from values"""
        vmin = np.nanmin(values)
        vmax = np.nanmax(values)
        self._extremes[column] = (vmin, vmax)


class OWScoreDocuments(OWWidget, ConcurrentWidgetMixin):
    name = "Score Documents"
    description = ""
    icon = "icons/ScoreDocuments.svg"
    priority = 500

    buttons_area_orientation = Qt.Vertical

    # default order - table sorted in input order
    DEFAULT_SORTING = (-1, Qt.AscendingOrder)

    settingsHandler = PerfectDomainContextHandler()
    auto_commit: bool = Setting(True)
    aggregation: int = Setting(0)

    word_frequency: bool = Setting(True)
    word_appearance: bool = Setting(False)
    embedding_similarity: bool = Setting(False)
    embedding_language: int = Setting(0)

    sort_column_order: Tuple[int, int] = Setting(DEFAULT_SORTING)
    selected_rows: List[int] = ContextSetting([], schema_only=True)
    sel_method: int = ContextSetting(SelectionMethods.N_BEST)
    n_selected: int = ContextSetting(3)

    class Inputs:
        corpus = Input("Corpus", Corpus)
        words = Input("Words", Table)

    class Outputs:
        selected_documents = Output("Selected documents", Corpus, default=True)
        corpus = Output("Corpus", Corpus)

    class Warning(OWWidget.Warning):
        corpus_not_normalized = Msg("Use Preprocess Text to normalize corpus.")

    class Error(OWWidget.Error):
        custom_err = Msg("{}")

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        self._setup_control_area()
        self._setup_main_area()
        self.corpus = None
        self.words = None
        # saves scores avoid multiple computation of the same score
        self.scores = {}

    def _setup_control_area(self) -> None:
        box = gui.widgetBox(self.controlArea, "Word Scoring Methods")
        for value, (n, _, tt) in SCORING_METHODS.items():
            b = gui.hBox(box, margin=0)
            gui.checkBox(
                b,
                self,
                value,
                label=n,
                callback=self.__setting_changed,
                tooltip=tt,
            )
            if value in ADDITIONAL_OPTIONS:
                value, options = ADDITIONAL_OPTIONS[value]
                gui.comboBox(
                    b,
                    self,
                    value,
                    items=options,
                    callback=self.__setting_changed,
                )

        box = gui.widgetBox(self.controlArea, "Aggregation")
        gui.comboBox(
            box,
            self,
            "aggregation",
            items=[n for n in AGGREGATIONS],
            callback=self.__setting_changed,
        )

        gui.rubber(self.controlArea)

        # select words box
        box = gui.vBox(self.buttonsArea, "Select Documents")
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)

        self._sel_method_buttons = QButtonGroup()
        for method, label in enumerate(SelectionMethods.ITEMS):
            button = QRadioButton(label)
            button.setChecked(method == self.sel_method)
            grid.addWidget(button, method, 0)
            self._sel_method_buttons.addButton(button, method)
        self._sel_method_buttons.buttonClicked[int].connect(self.__set_selection_method)

        spin = gui.spin(
            box,
            self,
            "n_selected",
            1,
            999,
            addToLayout=False,
            callback=lambda: self.__set_selection_method(SelectionMethods.N_BEST),
        )
        grid.addWidget(spin, 3, 1)
        box.layout().addLayout(grid)

        # autocommit
        gui.auto_send(self.buttonsArea, self, "auto_commit")

    def _setup_main_area(self) -> None:
        self._filter_line_edit = QLineEdit(
            textChanged=self.__on_filter_changed, placeholderText="Filter..."
        )
        self.mainArea.layout().addWidget(self._filter_line_edit)

        self.model = model = ScoreDocumentsTableModel(parent=self)
        model.setHorizontalHeaderLabels(["Document"])

        def select_manual():
            self.__set_selection_method(SelectionMethods.MANUAL)

        self.view = view = ScoreDocumentsTableView()
        view.pressedAny.connect(select_manual)
        self.mainArea.layout().addWidget(view)
        # by default data are sorted in the Table order
        header = self.view.horizontalHeader()
        header.sectionClicked.connect(self.__on_horizontal_header_clicked)

        proxy_model = ScoreDocumentsProxyModel()
        proxy_model.setFilterKeyColumn(0)
        proxy_model.setFilterCaseSensitivity(False)
        view.setModel(proxy_model)
        view.model().setSourceModel(self.model)
        self.view.selectionModel().selectionChanged.connect(self.__on_selection_change)

    def __on_filter_changed(self) -> None:
        model = self.view.model()
        model.setFilterFixedString(self._filter_line_edit.text().strip())

    def __on_horizontal_header_clicked(self, index: int):
        header = self.view.horizontalHeader()
        self.sort_column_order = (index, header.sortIndicatorOrder())
        self._select_rows()
        # when sorting change output table must consider the new order
        # call explicitly since selection in table is not changed
        if (
            self.sel_method == SelectionMethods.MANUAL
            and self.selected_rows
            or self.sel_method == SelectionMethods.ALL
        ):
            # retrieve selection in new order
            self.selected_rows = self.get_selected_indices()
            self._send_output()

    def __on_selection_change(self):
        self.selected_rows = self.get_selected_indices()
        self._send_output()

    def __set_selection_method(self, method: int):
        self.sel_method = method
        self._sel_method_buttons.button(method).setChecked(True)
        self._select_rows()

    @Inputs.corpus
    def set_data(self, corpus: Corpus) -> None:
        self.closeContext()
        self.Warning.corpus_not_normalized.clear()
        if corpus is None:
            self.corpus = None
            self._clear_and_run()
            return
        if not self._is_corpus_normalized(corpus):
            self.Warning.corpus_not_normalized()
        self.corpus = corpus
        self.selected_rows = []
        self.openContext(corpus)
        self._sel_method_buttons.button(self.sel_method).setChecked(True)
        self._clear_and_run()

    @staticmethod
    def _get_word_attribute(words: Table) -> None:
        attrs = [
            a
            for a in words.domain.metas + words.domain.variables
            if isinstance(a, StringVariable)
        ]
        if not attrs:
            return None
        words_attr = next(
            (a for a in attrs if a.attributes.get("type", "") == "words"), None
        )
        if words_attr:
            return words.get_column_view(words_attr)[0].tolist()
        else:
            # find the most suitable attribute - one with lowest average text
            # length - counted as a number of words
            def avg_len(attr):
                array_ = words.get_column_view(attr)[0]
                array_ = array_[~isnull(array_)]
                return sum(len(a.split()) for a in array_) / len(array_)
            attr = sorted(attrs, key=avg_len)[0]
            return words.get_column_view(attr)[0].tolist()

    @Inputs.words
    def set_words(self, words: Table) -> None:
        if words is None or len(words.domain.variables + words.domain.metas) == 0:
            self.words = None
        else:
            self.words = self._get_word_attribute(words)
        self._clear_and_run()

    def _gather_scores(self) -> Tuple[np.ndarray, List[str]]:
        """
        Gather scores and labels for the dictionary that holds scores

        Returns
        -------
        scores
            Scores table
        labels
            The list with score names for the header and variables names
        """
        if self.corpus is None:
            return np.empty((0, 0)), []
        aggregation = self._get_active_aggregation()
        scorers = self._get_active_scorers()
        methods = [m for m in scorers if (m, aggregation) in self.scores]
        scores = [self.scores[(m, aggregation)] for m in methods]
        scores = np.column_stack(scores) if scores else np.empty((len(self.corpus), 0))
        labels = [SCORING_METHODS[m][0] for m in methods]
        return scores, labels

    def _send_output(self) -> None:
        """
        Create corpus with scores and output it
        """
        if self.corpus is None:
            self.Outputs.corpus.send(None)
            self.Outputs.selected_documents.send(None)
            return

        scores, labels = self._gather_scores()
        if labels:
            d = self.corpus.domain
            domain = Domain(
                d.attributes,
                d.class_var,
                metas=d.metas + tuple(ContinuousVariable(get_unique_names(d, l))
                                      for l in labels),
            )
            out_corpus = Corpus(
                domain,
                self.corpus.X,
                self.corpus.Y,
                np.hstack([self.corpus.metas, scores]),
            )
            Corpus.retain_preprocessing(self.corpus, out_corpus)
        else:
            out_corpus = self.corpus

        self.Outputs.corpus.send(create_annotated_table(out_corpus, self.selected_rows))
        self.Outputs.selected_documents.send(
            out_corpus[self.selected_rows] if self.selected_rows else None
        )

    def _fill_table(self) -> None:
        """
        Fill the table in the widget with scores and document names
        """
        if self.corpus is None:
            self.model.clear()
            return
        scores, labels = self._gather_scores()
        labels = ["Document"] + labels
        titles = self.corpus.titles.tolist()

        # clearing selection and sorting to prevent SEGFAULT on model.wrap
        self.view.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)
        with disconnected(
            self.view.selectionModel().selectionChanged, self.__on_selection_change
        ):
            self.view.clearSelection()

        self.model.fill_table(titles, scores)
        self.model.setHorizontalHeaderLabels(labels)
        self.view.update_column_widths()
        if self.model.columnCount() > self.sort_column_order[0]:
            # if not enough columns do not apply sorting from settings since
            # sorting can besaved for score column while scores are still computing
            # tables is filled before scores are computed with document names
            self.view.horizontalHeader().setSortIndicator(*self.sort_column_order)

        self._select_rows()

    def _fill_and_output(self) -> None:
        """Fill the table in the widget and send the output"""
        self._fill_table()
        self._send_output()

    def _clear_and_run(self) -> None:
        """Clear cached scores and commit"""
        self.scores = {}
        self.cancel()
        self._fill_and_output()
        self.commit()

    def __setting_changed(self) -> None:
        self.commit()

    def commit(self) -> None:
        self.Error.custom_err.clear()
        self.cancel()
        if self.corpus is not None and self.words is not None:
            scorers = self._get_active_scorers()
            aggregation = self._get_active_aggregation()
            new_scores = [s for s in scorers if (s, aggregation) not in self.scores]
            if new_scores:
                self.start(
                    _run,
                    self.corpus,
                    self.words,
                    new_scores,
                    aggregation,
                    {
                        v: items[getattr(self, v)]
                        for v, items in ADDITIONAL_OPTIONS.values()
                    },
                )
            else:
                self._fill_and_output()

    def on_done(self, _: None) -> None:
        self._send_output()

    def on_partial_result(self, result: Tuple[str, str, np.ndarray]) -> None:
        sc_method, aggregation, scores = result
        self.scores[(sc_method, aggregation)] = scores
        self._fill_table()

    def on_exception(self, ex: Exception) -> None:
        self.Error.custom_err(ex)
        self._fill_and_output()

    def _get_active_scorers(self) -> List[str]:
        """
        Gather currently active/selected scores

        Returns
        -------
        List with selected scores names
        """
        return [attr for attr in SCORING_METHODS if getattr(self, attr)]

    def _get_active_aggregation(self) -> str:
        """
        Gather currently active/selected aggregation

        Returns
        -------
        Selected aggregation name
        """
        return list(AGGREGATIONS.keys())[self.aggregation]

    @staticmethod
    def _is_corpus_normalized(corpus: Corpus) -> bool:
        """
        Check if corpus is normalized.
        """
        return any(
            isinstance(pp, BaseNormalizer)
            for pp in corpus.used_preprocessor.preprocessors
        )

    def get_selected_indices(self) -> List[int]:
        # get indices in table's order - that the selected output table have same order
        selected_rows = sorted(
            self.view.selectionModel().selectedRows(), key=lambda idx: idx.row()
        )
        return [self.view.model().mapToSource(r).row() for r in selected_rows]

    def _select_rows(self):
        proxy_model = self.view.model()
        n_rows, n_columns = proxy_model.rowCount(), proxy_model.columnCount()
        if self.sel_method == SelectionMethods.NONE:
            selection = QItemSelection()
        elif self.sel_method == SelectionMethods.ALL:
            selection = QItemSelection(
                proxy_model.index(0, 0), proxy_model.index(n_rows - 1, n_columns - 1)
            )
        elif self.sel_method == SelectionMethods.MANUAL:
            selection = QItemSelection()
            new_sel = []
            for row in self.selected_rows:
                if row < n_rows:
                    new_sel.append(row)
                    _selection = QItemSelection(
                        self.model.index(row, 0), self.model.index(row, n_columns - 1)
                    )
                    selection.merge(
                        proxy_model.mapSelectionFromSource(_selection),
                        QItemSelectionModel.Select,
                    )
            # selected rows must be updated when the same dataset with less rows
            # appear at the input - it is not handled by selectionChanged
            # in cases when all selected rows missing in new table
            self.selected_rows = new_sel
        elif self.sel_method == SelectionMethods.N_BEST:
            n_sel = min(self.n_selected, n_rows)
            selection = QItemSelection(
                proxy_model.index(0, 0), proxy_model.index(n_sel - 1, n_columns - 1)
            )
        else:
            raise NotImplementedError

        self.view.selectionModel().select(selection, QItemSelectionModel.ClearAndSelect)


if __name__ == "__main__":
    from orangewidget.utils.widgetpreview import WidgetPreview

    from orangecontrib.text import preprocess

    corpus = Corpus.from_file("book-excerpts")
    # corpus.set_title_variable("Text")

    pp_list = [
        preprocess.LowercaseTransformer(),
        preprocess.StripAccentsTransformer(),
        preprocess.SnowballStemmer(),
    ]
    for p in pp_list:
        corpus = p(corpus)

    w = StringVariable("Words")
    w.attributes["type"] = "words"
    words = ["house", "doctor", "boy", "way", "Rum"]
    words = Table(
        Domain([], metas=[w]),
        np.empty((len(words), 0)),
        metas=np.array(words).reshape((-1, 1)),
    )
    WidgetPreview(OWScoreDocuments).run(set_data=corpus, set_words=words)
