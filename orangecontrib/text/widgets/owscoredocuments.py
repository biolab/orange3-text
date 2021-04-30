import re
from collections import Counter
from inspect import signature
from typing import List, Callable, Tuple, Union

import numpy as np
from pandas import isnull
from Orange.data import (
    Table,
    Domain,
    StringVariable,
    ContinuousVariable,
    DiscreteVariable,
)
from Orange.util import wrap_callback
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.widget import Input, Msg, Output, OWWidget
from orangewidget import gui
from orangewidget.settings import Setting
from Orange.widgets.utils.itemmodels import PyTableModel, TableModel
from AnyQt.QtWidgets import QTableView, QLineEdit, QHeaderView
from AnyQt.QtCore import Qt, QSortFilterProxyModel, QSize
from sklearn.metrics.pairwise import cosine_similarity

from orangecontrib.text import Corpus
from orangecontrib.text.preprocess import BaseNormalizer, BaseTransformer
from orangecontrib.text.vectorization.document_embedder import (
    LANGS_TO_ISO,
    DocumentEmbedder,
)


def _word_frequency(
    corpus: Corpus, words: List[str], callback: Callable
) -> np.ndarray:
    res = []
    for i, t in enumerate(corpus.tokens):
        counts = Counter(t)
        res.append([counts.get(w, 0) for w in words])
        callback((i + 1) / len(corpus.tokens))
    return np.array(res)


def _word_appearance(
    corpus: Corpus, words: List[str], callback: Callable
) -> np.ndarray:
    res = []
    for i, t in enumerate(corpus.tokens):
        t = set(t)
        res.append([w in t for w in words])
        callback((i + 1) / len(corpus.tokens))
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
    # only transformers and normalizers preprocess on the word level
    pps = [
        pp
        for pp in corpus.used_preprocessor.preprocessors
        if isinstance(pp, (BaseTransformer, BaseNormalizer))
    ]
    for i, pp in enumerate(pps):
        # TODO: _preprocess is protected make it public
        words = [pp._preprocess(w) for w in words]
        callback((i + 1) / len(pps))
    return words


def _run(
    corpus: Corpus,
    words: Table,
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

    words = _preprocess_words(
        corpus, words, wrap_callback(callback, end=cb_part)
    )
    for i, sm in enumerate(scoring_methods):
        scoring_method = SCORING_METHODS[sm][1]
        sig = signature(scoring_method)
        add_params = {
            k: v for k, v in additional_params.items() if k in sig.parameters
        }
        scs = scoring_method(
            corpus,
            words,
            wrap_callback(
                callback, start=(i + 1) * cb_part, end=(i + 2) * cb_part
            ),
            **add_params
        )
        scs = AGGREGATIONS[aggregation](scs, axis=1)
        state.set_partial_result((sm, aggregation, scs))


class ScoreDocumentsTableView(QTableView):
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

    def update_column_widths(self) -> None:
        """
        Set columns widths such that each score column has width based on size
        hint and all scores columns have the same width.
        """
        header = self.horizontalHeader()
        col_width = max(
            [0] + [
                max(self.sizeHintForColumn(i), header.sectionSizeHint(i))
                for i in range(1, self.model().columnCount())
            ]
        )

        for i in range(1, self.model().columnCount()):
            header.resizeSection(i, col_width)
            header.setSectionResizeMode(i, QHeaderView.Fixed)

        # document title column is one that stretch
        header.setSectionResizeMode(0, QHeaderView.Stretch)


class ScoreDocumentsProxyModel(QSortFilterProxyModel):
    @staticmethod
    def _convert(text: str) -> Union[str, int]:
        return int(text) if text.isdigit() else text.lower()

    @staticmethod
    def _alphanum_key(key: str) -> List[Union[str, int]]:
        return [
            ScoreDocumentsProxyModel._convert(c)
            for c in re.split("([0-9]+)", key)
        ]

    def lessThan(self, left_ind, right_ind):
        """
        Sort strings of the first column naturally: Document 2 < Document 12
        """
        if left_ind.column() == 0 and right_ind.column() == 0:
            left = self.sourceModel().data(left_ind, role=Qt.DisplayRole)
            right = self.sourceModel().data(right_ind, role=Qt.DisplayRole)
            return self._alphanum_key(left) < self._alphanum_key(right)
        return super().lessThan(left_ind, right_ind)


class ScoreDocumentsTableModel(PyTableModel):
    def data(self, index, role=Qt.DisplayRole):
        if role in (gui.BarRatioRole, Qt.DisplayRole):
            dat = super().data(index, Qt.EditRole)
            return dat
        if role == Qt.BackgroundColorRole and index.column() == 0:
            return TableModel.ColorForRole[TableModel.Meta]
        return super().data(index, role)


class OWScoreDocuments(OWWidget, ConcurrentWidgetMixin):
    name = "Score Documents"
    description = ""
    icon = "icons/ScoreDocuments.svg"
    priority = 500

    auto_commit: bool = Setting(True)
    aggregation: int = Setting(0)

    word_frequency: bool = Setting(True)
    word_appearance: bool = Setting(False)
    embedding_similarity: bool = Setting(False)
    embedding_language = Setting(0)

    class Inputs:
        corpus = Input("Corpus", Corpus)
        words = Input("Words", Table)

    class Outputs:
        corpus = Output("Corpus", Corpus)

    class Warning(OWWidget.Warning):
        missing_words = Msg("Provide words on the input")
        missing_corpus = Msg("Provide corpus on the input")
        corpus_not_normalized = Msg("Use Preprocess Text to normalize corpus.")

    class Error(OWWidget.Error):
        unknown_err = Msg("{}")

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
        box = gui.widgetBox(self.controlArea, "Word scoring")
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

        box = gui.widgetBox(self.controlArea, "Aggregate scores")
        gui.comboBox(
            box,
            self,
            "aggregation",
            searchable=True,
            items=[n for n in AGGREGATIONS],
            callback=self.__setting_changed,
        )

        gui.rubber(self.controlArea)
        gui.auto_send(self.buttonsArea, self, "auto_commit")

    def _setup_main_area(self) -> None:
        self._filter_line_edit = QLineEdit(
            textChanged=self.__on_filter_changed, placeholderText="Filter..."
        )
        self.mainArea.layout().addWidget(self._filter_line_edit)

        self.model = model = ScoreDocumentsTableModel(parent=self)
        model.setHorizontalHeaderLabels(["Document"])

        self.view = view = ScoreDocumentsTableView()
        self.mainArea.layout().addWidget(view)

        proxy_model = ScoreDocumentsProxyModel()
        proxy_model.setFilterKeyColumn(0)
        proxy_model.setFilterCaseSensitivity(False)
        view.setModel(proxy_model)
        view.model().setSourceModel(self.model)

    def __on_filter_changed(self) -> None:
        model = self.view.model()
        model.setFilterFixedString(self._filter_line_edit.text().strip())

    @Inputs.corpus
    def set_data(self, corpus: Corpus) -> None:
        self.Warning.corpus_not_normalized.clear()
        if corpus is not None:
            self.Warning.missing_corpus.clear()
            if not self._is_corpus_normalized(corpus):
                self.Warning.corpus_not_normalized()
        self.corpus = corpus
        # todo: rename
        self._clear_and_run()

    @staticmethod
    def _get_word_attribute(words: Table) -> None:
        attrs = [
            a
            for a in words.domain.metas + words.domain.variables
            if isinstance(a, (StringVariable, DiscreteVariable))
        ]
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

            _, attr = sorted((avg_len(a), a) for a in attrs)[0]
            return words.get_column_view(attr)[0].tolist()

    @Inputs.words
    def set_words(self, words: Table) -> None:
        if (
            words is None
            or len(words.domain.variables + words.domain.metas) == 0
        ):
            self.words = None
        else:
            self.Warning.missing_words.clear()
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
        scores = (
            np.column_stack(scores)
            if scores
            else np.empty((len(self.corpus), 0))
        )
        labels = [SCORING_METHODS[m][0] for m in methods]
        return scores, labels

    def _send_output(self, scores: np.ndarray, labels: List[str]) -> None:
        """
        Create corpus with scores and output it
        """
        if labels:
            d = self.corpus.domain
            domain = Domain(
                d.attributes,
                d.class_var,
                metas=d.metas + tuple(ContinuousVariable(l) for l in labels),
            )
            corpus = Corpus(
                domain,
                self.corpus.X,
                self.corpus.Y,
                np.hstack([self.corpus.metas, scores]),
            )
            Corpus.retain_preprocessing(self.corpus, corpus)
            self.Outputs.corpus.send(corpus)
        elif self.corpus is not None:
            self.Outputs.corpus.send(self.corpus)
        else:
            self.Outputs.corpus.send(None)

    def _fill_table(self, scores: np.ndarray, labels: List[str]) -> None:
        """
        Fill the table in the widget with scores and document names
        """
        if self.corpus is None:
            self.model.clear()
            return
        labels = ["Document"] + labels
        titles = self.corpus.titles.tolist()
        self.model.wrap([[c] + s for c, s in zip(titles, scores.tolist())])
        self.model.setHorizontalHeaderLabels(labels)
        self.view.update_column_widths()

        # documents are not ordered by default by any column
        self.view.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)

    def _fill_and_output(self) -> None:
        """ Fill the table in the widget and send the output """
        scores, labels = self._gather_scores()
        self._fill_table(scores, labels)
        self._send_output(scores, labels)

    def _clear_and_run(self) -> None:
        """ Clear cached scores and commit """
        self.scores = {}
        self.cancel()
        self._fill_and_output()
        self.commit()

    def __setting_changed(self) -> None:
        self.commit()

    def commit(self) -> None:
        self.Error.unknown_err.clear()
        self.cancel()
        if self.corpus is None and self.words is None:
            return
        elif self.corpus is None:
            self.Warning.missing_corpus()
        elif self.words is None:
            self.Warning.missing_words()
        else:
            scorers = self._get_active_scorers()
            aggregation = self._get_active_aggregation()
            new_scores = [
                s for s in scorers if (s, aggregation) not in self.scores
            ]
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
        scores, labels = self._gather_scores()
        self._send_output(scores, labels)

    def on_partial_result(self, result: Tuple[str, str, np.ndarray]) -> None:
        sc_method, aggregation, scores = result
        self.scores[(sc_method, aggregation)] = scores
        scores, labels = self._gather_scores()
        self._fill_table(scores, labels)

    def on_exception(self, ex: Exception) -> None:
        self.Error.unknown_err(ex)

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
