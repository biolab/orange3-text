# coding: utf-8
from collections import Counter
from itertools import cycle
from math import pi as PI
from typing import Dict, List, Optional, Tuple

import numpy as np
from AnyQt import QtCore
from AnyQt.QtCore import (QItemSelection, QItemSelectionModel, QObject, QSize,
                          QSortFilterProxyModel, Qt, pyqtSlot)

from Orange.data import ContinuousVariable, Domain, StringVariable, Table
from Orange.data.util import scale
from Orange.widgets import gui, settings, widget
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.utils.itemmodels import PyTableModel
from Orange.widgets.widget import Input, Output, OWWidget
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.topics import Topic

COLORS = ["#da1", "#629", "#787"]
GRAY_COLORS = ["#000", "#444", "#777", "#aaa"]
TOPIC_COLORS = ["#ff6600", "#00cc00"]  # [negative topic, positive topic]
GRAY_TOPIC_COLORS = ["#000", "#aaa"]  # [negative topic, positive topic]
TILT_VALUES = ("no", "30°", "45°", "60°")

N_BEST_PLOTTED = 200


def _bow_words(corpus):
    """
    This function extract words from bag of words features and assign them
    the frequency which is average bow count.
    """
    average_bows = {
        f.name: corpus.X[:, i].mean()
        for i, f in enumerate(corpus.domain.attributes)
        if f.attributes.get("bow-feature", False)
    }
    # return only positive bow weights (those == 0 are non-existing words)
    return {f: w for f, w in average_bows.items() if w > 0}


def count_words(data: Corpus, state: TaskState) -> Tuple[Counter, bool]:
    """
    This function implements counting process of the word cloud widget and
    is called in the separate thread by concurrent.

    Parameters
    ----------
    data
        Corpus with the data
    state
        State used to report status.

    Returns
    -------
    Reports counts as a counter and boolean that tell whether the data were
    retrieved on bag of words basis.
    """
    state.set_status("Calculating...")
    state.set_progress_value(0)
    bow_counts = _bow_words(data)
    state.set_progress_value(0.5)
    if bow_counts:
        corpus_counter = Counter(bow_counts)
    else:
        corpus_counter = Counter(
            w for doc in data.ngrams for w in doc
        )
    state.set_progress_value(1)
    return corpus_counter, bool(bow_counts)


class TableModel(PyTableModel):
    def __init__(self, precision, **kwargs):
        super().__init__(**kwargs)
        self.precision = precision

    def data(self, index, role=Qt.DisplayRole):
        """
        Format numbers of the first column with the number of decimal
        spaces defined by self.predictions which can be changed based on
        weights type - row counts does not have decimal spaces
        """
        row, column = self.mapToSourceRows(index.row()), index.column()
        if role == Qt.DisplayRole and column == 0:
            value = float(self[row][column])
            return f"{value:.{self.precision}f}"
        return super().data(index, role)

    def set_precision(self, precision: int):
        """
        Setter for precision.

        Parameters
        ----------
        precision
            Number of decimal spaces to format the weights.
        """
        self.precision = precision


class OWWordCloud(OWWidget, ConcurrentWidgetMixin):
    name = "Word Cloud"
    priority = 510
    icon = "icons/WordCloud.svg"

    class Inputs:
        corpus = Input("Corpus", Corpus, default=True)
        topic = Input("Topic", Topic)

    class Outputs:
        corpus = Output("Corpus", Corpus)
        selected_words = Output("Selected Words", Topic, dynamic=False)
        word_counts = Output("Word Counts", Table)

    graph_name = "webview"

    selected_words = settings.Setting(set(), schema_only=True)

    words_color = settings.Setting(True)
    words_tilt = settings.Setting(0)

    class Warning(widget.OWWidget.Warning):
        topic_precedence = widget.Msg(
            "Input signal Topic takes priority over Corpus"
        )

    class Info(widget.OWWidget.Information):
        bow_weights = widget.Msg("Showing bag of words weights.")

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        self.n_topic_words = 0
        self.documents_info_str = ""
        self.webview = None
        self.topic = None
        self.corpus = None
        self.corpus_counter = None
        self.wordlist = None
        self.shown_words = None
        self.shown_weights = None
        self.combined_size_length = None
        self._create_layout()
        self.on_corpus_change(None)
        self.update_input_summary()
        self.update_output_summary(None, None)

    def _new_webview(self):
        HTML = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
html, body {margin:0px;padding:0px;width:100%;height:100%;
            cursor:default; -webkit-user-select: none; user-select: none;
            overflow: hidden;}
span:hover {color:OrangeRed !important}
span.selected {color:red !important}
</style>
</head>
<body id="canvas">
<script src="resources/wordcloud2.js"></script>
<script src="resources/wordcloud-script.js"></script>
</body>
</html>"""
        if self.webview:
            self.mainArea.layout().removeWidget(self.webview)
            # parent is still aware of the view so we need to remove it
            # and remove it from parents tree
            self.webview.deleteLater()

        class Bridge(QObject):
            @pyqtSlot("QVariantList")
            def update_selection(_, words):
                nonlocal webview
                self.update_selection(words, webview)

        class Webview(gui.WebviewWidget):
            def update_selection(self, words):
                self.evalJS(
                    "SELECTED_WORDS = {}; selectWords();".format(list(words))
                )

        webview = self.webview = Webview(self.mainArea, Bridge(), debug=False)
        webview.setHtml(HTML, webview.toFileURL(__file__))
        self.mainArea.layout().addWidget(webview)

    def _create_layout(self):
        box = gui.widgetBox(self.controlArea, "Cloud preferences")
        gui.checkBox(
            box,
            self,
            "words_color",
            "Color words",
            callback=self.on_cloud_pref_change,
        )
        gui.valueSlider(
            box,
            self,
            "words_tilt",
            label="Words tilt:",
            values=list(range(len(TILT_VALUES))),
            callback=self.on_cloud_pref_change,
            labelFormat=lambda x: TILT_VALUES[x],
        )

        box = gui.widgetBox(self.controlArea, "Words && weights")

        class TableView(gui.TableView):
            def __init__(self, parent):
                super().__init__(parent)
                self._parent = parent
                self.__nope = False

            def setModel(self, model):
                """Otherwise QTableView.setModel() calls
                QAbstractItemView.setSelectionModel() which resets selection,
                calling selectionChanged() and overwriting any selected_words
                setting that may have been saved."""
                self.__nope = True
                super().setModel(model)
                self.__nope = False

            def selectionChanged(self, selected, deselected):
                nonlocal model, proxymodel
                super().selectionChanged(selected, deselected)
                if not self.__nope:
                    words = {
                        model[proxymodel.mapToSource(index).row()][1]
                        for index in self.selectionModel().selectedIndexes()
                    }
                    self._parent.update_selection(words, self)

            def update_selection(self, words):
                nonlocal model, proxymodel
                selection = QItemSelection()
                for i, (_, word) in enumerate(model):
                    if word in words:
                        index = proxymodel.mapFromSource(model.index(i, 1))
                        selection.select(index, index)
                self.__nope = True
                self.clearSelection()
                self.selectionModel().select(
                    selection,
                    QItemSelectionModel.Select | QItemSelectionModel.Rows,
                )
                self.__nope = False

        view = self.tableview = TableView(self)
        model = self.tablemodel = TableModel(2, parent=self)
        proxymodel = QSortFilterProxyModel(
            self,
            dynamicSortFilter=True,
            sortCaseSensitivity=Qt.CaseInsensitive,
            sortRole=Qt.EditRole,
        )
        proxymodel.setSourceModel(model)
        model.setHorizontalHeaderLabels(["Weight", "Word"])
        view.setModel(proxymodel)
        box.layout().addWidget(view)

    def define_colors(
            self, words: List[str], weights: List[float]
    ) -> Dict[str, str]:
        if (self.topic is not None
                and self.topic.attributes["topic-method-name"] == "LsiModel"):
            # when topic and topic method is LSI then color
            # positive and negative numbers
            palette = TOPIC_COLORS if self.words_color else GRAY_TOPIC_COLORS
            colors = {
                word: palette[int(weight >= 0)]
                for word, weight in zip(words, weights)
            }
        else:
            color_generator = cycle(
                COLORS if self.words_color else GRAY_COLORS
            )
            colors = {word: next(color_generator) for word in words}
        return colors

    def on_cloud_pref_change(self):
        if self.wordlist is None:
            return
        self._new_webview()

        # Generate colors
        colors_dict = self.define_colors(self.shown_words, self.shown_weights)

        self.webview.evalJS(f"colorList = {colors_dict}")
        # this function makes sure that word color is always same, so color
        # of the word depends on its letters
        self.webview.evalJS(
            """OPTIONS["color"] = function (word) {
            return colorList[word];
            }"""
        )
        self.webview.evalJS(
            f"textAreaEstimation = {self.combined_size_length}"
        )
        tilt_ratio, tilt_amount = {
            0: (0, 0),
            1: (1, PI / 6),
            2: (1, PI / 4),
            3: (0.67, PI / 3),
        }[self.words_tilt]
        self.webview.evalJS(
            'OPTIONS["minRotation"] = {}; \
                             OPTIONS["maxRotation"] = {};'.format(
                -tilt_amount, tilt_amount
            )
        )
        self.webview.evalJS('OPTIONS["rotateRatio"] = {};'.format(tilt_ratio))
        self.webview.evalJS('OPTIONS["list"] = {};'.format(self.wordlist))
        self.webview.evalJS("redrawWordCloud();")
        self.webview.update_selection(self.selected_words)

    def _repopulate_wordcloud(
        self, words: List[str], weights: List[float]
    ) -> None:
        """
        This function prepare a word list and trigger a cloud replot.

        Parameters
        ----------
        words
            List of words to show.
        weights
            Words' weights
        """
        if not len(words):
            self.clear()
            return

        def is_whole(d):
            """Whether or not d is a whole number."""
            return (
                    isinstance(d, int)
                    or (isinstance(d, float) and d.is_integer())
            )

        words, weights = words[:N_BEST_PLOTTED], weights[:N_BEST_PLOTTED]
        self.shown_words, self.shown_weights = words, weights
        # Repopulate table
        self.tablemodel.set_precision(
            0 if all(is_whole(w) for w in weights) else 2
        )
        self.tablemodel.wrap(list(zip(weights, words)))
        self.tableview.sortByColumn(0, Qt.DescendingOrder)

        # Reset wordcloud
        if self.topic is not None:
            # when weights are from topic negative weights should be treated
            # as positive when calculating the font size
            weights = np.abs(weights)
        weights = np.clip(weights, *np.percentile(weights, [2, 98]))
        weights = scale(weights, 10, 40)
        self.wordlist = np.c_[words, weights].tolist()

        # sometimes words are longer in average and word sizes in pt are bigger
        # in average - with this parameter we combine this in size scaling
        self.combined_size_length = sum([
            len(word) * float(weight) for word, weight in
            self.wordlist
        ])
        self.on_cloud_pref_change()

    @Inputs.topic
    def on_topic_change(self, data):
        self.topic = data
        self.handle_input()

    def _apply_topic(self):
        data = self.topic
        metas = data.domain.metas if data else []
        try:
            col = next(i for i, var in enumerate(metas) if var.is_string)
        except StopIteration:
            words = np.zeros((0, 1))
        else:
            words = data.metas[:, col]
            self.n_topic_words = data.metas.shape[0]
        if data and data.W.any():
            weights = data.W[:]
        elif data and "weights" in data.domain:
            weights = data.get_column_view(data.domain["weights"])[0]
        else:
            weights = np.ones(len(words))

        self._repopulate_wordcloud(words, weights)

    def _apply_corpus(self):
        words, freq = self.word_frequencies()
        self._repopulate_wordcloud(words, freq)

    def word_frequencies(self):
        counts = self.corpus_counter.most_common()
        words, freq = zip(*counts) if counts else ([], [])
        return words, freq

    def create_weight_list(self):
        wc_table = None
        if self.corpus is not None:
            words, freq = self.word_frequencies()
            words = np.array(words)[:, None]
            w_count = np.array(freq)[:, None]
            domain = Domain(
                [ContinuousVariable("Word Count")],
                metas=[StringVariable("Word")],
            )
            wc_table = Table.from_numpy(domain, X=w_count, metas=words)
            wc_table.name = "Word Counts"
        self.Outputs.word_counts.send(wc_table)

    @Inputs.corpus
    def on_corpus_change(self, data):
        self.corpus = data
        self.Info.clear()

        self.corpus_counter = Counter()
        if data is not None:
            self.start(count_words, data)
        else:
            self.handle_input()
        self.create_weight_list()

    def on_done(self, result: Tuple[Counter, bool]) -> None:
        self.corpus_counter = result[0]
        self.create_weight_list()
        if result[1]:
            self.Info.bow_weights()
        self.handle_input()

    def handle_input(self):
        if self.topic is not None and len(self.topic):
            self._apply_topic()
        elif self.corpus is not None and len(self.corpus):
            self._apply_corpus()
        else:
            self.clear()
            self.update_input_summary()
            return

        self.Warning.topic_precedence(
            shown=self.corpus is not None and self.topic is not None
        )
        if self.topic is not None or self.corpus is not None:
            if self.selected_words:
                self.update_selection(self.selected_words)
        self.commit()
        self.update_input_summary()

    def clear(self):
        self._new_webview()
        self.tablemodel.clear()
        self.wordlist = None
        self.commit()

    def update_selection(self, words, skip=None):
        assert skip is None or skip in (self.webview, self.tableview)
        self.selected_words = words = set(words)

        if self.tableview != skip:
            self.tableview.update_selection(words)
        if self.webview != skip:
            self.webview.update_selection(words)

        self.commit()

    def commit(self):
        out = None
        if self.corpus is not None:
            rows = [
                i
                for i, doc in enumerate(self.corpus.ngrams)
                if any(word in doc for word in self.selected_words)
            ]
            out = self.corpus[rows]
        self.Outputs.corpus.send(out)

        topic = None
        words = list(self.selected_words)
        if words:
            topic = Topic.from_numpy(
                Domain([], metas=[StringVariable("Words")]),
                X=np.empty((len(words), 0)),
                metas=np.c_[words].astype(object),
            )
            topic.name = "Selected Words"
        self.Outputs.selected_words.send(topic)
        self.update_output_summary(
            len(out) if out is not None else None,
            len(topic) if topic is not None else None,
        )

    def update_input_summary(self) -> None:
        if self.corpus is None and self.topic is None:
            self.info.set_input_summary(self.info.NoInput)
        else:
            input_string = ""
            input_numbers = ""
            if self.corpus is not None:
                input_string += (
                    f"{len(self.corpus)} documents with "
                    f"{len(self.corpus_counter)} words\n"
                )
                input_numbers += str(len(self.corpus_counter))
            if self.topic is not None:
                input_string += f"{self.n_topic_words} words in a topic."
                input_numbers += (
                    f"{' | ' if input_numbers else ''}" f"{self.n_topic_words}"
                )
            self.info.set_input_summary(input_numbers, input_string)

    def update_output_summary(
        self, cor_output_len: Optional[int], n_selected: Optional[int]
    ) -> None:
        if (
            cor_output_len is None
            and n_selected is None
            and (self.corpus_counter is None or len(self.corpus_counter) == 0)
        ):
            self.info.set_output_summary(self.info.NoOutput)
        else:
            cc_len = (
                len(self.corpus_counter)
                if self.corpus_counter is not None
                else 0
            )
            input_numbers = f"{cor_output_len or 0} | {n_selected or 0} | " \
                            f"{cc_len}"
            input_string = (
                f"{cor_output_len or 0} documents\n"
                f"{n_selected or 0} selected words\n"
                f"{cc_len} words with counts"
            )
            self.info.set_output_summary(input_numbers, input_string)

    def send_report(self):
        if self.webview:
            html = self.webview.html()
            start = html.index(">", html.index("<body")) + 1
            end = html.index("</body>")
            body = html[start:end]
            # create an empty div of appropriate height to compensate for
            # absolute positioning of words in the html
            height = self.webview._evalJS(
                "document.getElementById('canvas').clientHeight"
            )
            self.report_html += "<div style='position: relative; height: " \
                                f"{height}px;'>{body}</div>"

            self.report_table(self.tableview)

    def sizeHint(self) -> QtCore.QSize:
        return super().sizeHint().expandedTo(QSize(900, 500))


if __name__ == "__main__":
    from orangewidget.utils.widgetpreview import WidgetPreview

    corpus = Corpus.from_file("book-excerpts")
    WidgetPreview(OWWordCloud).run(corpus)
