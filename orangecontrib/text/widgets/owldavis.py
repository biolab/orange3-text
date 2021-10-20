# coding: utf-8
from typing import Optional, Union, List
import numpy as np

from AnyQt.QtCore import Qt, QRectF, QPointF
import pyqtgraph as pg

from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QGraphicsSceneHelpEvent, QToolTip

from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.widget import Input, OWWidget
from Orange.widgets.visualize.utils.plotutils import HelpEventDelegate
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.topics import LdaWrapper
from orangecontrib.text.topics.topics import Topics
from orangewidget.settings import Setting, SettingProvider
from orangewidget.utils.widgetpreview import WidgetPreview
from orangewidget.widget import Msg

N_BEST_PLOTTED = 20


class BarPlotGraph(pg.PlotWidget):
    bar_width = 0.7

    def __init__(self, master, parent=None):
        self.master: OWLDAvis = master
        self.bar_item: pg.BarGraphItem = None
        super().__init__(
            parent=parent,
            viewBox=pg.ViewBox(),
            background="w", enableMenu=False,
            axisItems={"left": pg.AxisItem(orientation="left",
                                           rotate_ticks=False),
                       "top": pg.AxisItem(orientation="top")}
        )
        self.hideAxis("left")
        self.hideAxis("top")
        self.getPlotItem().buttonsHidden = True
        self.getPlotItem().setContentsMargins(10, 0, 0, 10)

        self.tooltip_delegate = HelpEventDelegate(self.help_event)
        self.scene().installEventFilter(self.tooltip_delegate)

    def reset_graph(self):
        self.clear()
        self.update_bars()
        self.update_axes()
        self.reset_view()

    def update_bars(self):
        if self.bar_item is not None:
            self.removeItem(self.bar_item)
            self.bar_item = None

        values = self.master.get_values()
        if values is None:
            return

        self.bar_item = pg.BarGraphItem(
            x=np.arange(len(values)),
            height=values,
            width=self.bar_width,
            pen=pg.mkPen(QColor(Qt.white)),
            labels=self.master.get_labels(),
            brushes=[QColor(Qt.red) for _ in range(len(
                self.master.shown_words))],
        )
        self.addItem(self.bar_item)

    def update_axes(self):
        if self.bar_item is not None:
            self.showAxis("left")
            self.showAxis("top")

            self.setLabel(axis="left", text="words")
            self.setLabel(axis="top", text="weights")

            ticks = [list(enumerate(self.master.get_labels()))]
            self.getAxis("left").setTicks(ticks)
        else:
            self.hideAxis("left")
            self.hideAxis("top")

    def reset_view(self):
        if self.bar_item is None:
            return
        values = np.append(self.bar_item.opts["height"], 0)
        min_ = np.nanmin(values)
        max_ = -min_ + np.nanmax(values)
        rect = QRectF(-0.5, min_, len(values) - 1, max_)
        self.getViewBox().setRange(rect)

    def __get_index_at(self, p: QPointF):
        x = p.x()
        index = round(x)
        heights = self.bar_item.opts["height"]
        if 0 <= index < len(heights) and abs(x - index) <= self.bar_width / 2:
            height = heights[index]  # pylint: disable=unsubscriptable-object
            if 0 <= p.y() <= height or height <= p.y() <= 0:
                return index
        return None

    def help_event(self, ev: QGraphicsSceneHelpEvent):
        if self.bar_item is None:
            return False

        index = self.__get_index_at(self.bar_item.mapFromScene(ev.scenePos()))
        text = ""
        if index is not None:
            text = self.master.get_tooltip(index)
        if text:
            QToolTip.showText(ev.screenPos(), text, widget=self)
            return True
        else:
            return False


class OWLDAvis(OWWidget):
    name = "LDAvis"
    description = "Interactive exploration of LDA topics."
    priority = 410
    icon = "icons/LDAvis.svg"

    selected_topic = Setting(0, schema_only=True)
    relevance = Setting(0.5)

    graph = SettingProvider(BarPlotGraph)
    graph_name = "graph.plotItem"

    class Inputs:
        topics = Input("Topics", Topics)

    class Error(OWWidget.Error):
        # Relevant Terms cannot work with LSI or HDP, because it expects
        # topic-term probabilities.
        wrong_model = Msg("Relevant Terms only accepts output from LDA.")

    def __init__(self):
        OWWidget.__init__(self)
        self.data = None
        self.topic_list = []
        self.topic_frequency = None
        self.term_topic_matrix = None
        self.term_frequency = None
        self.shown_words = None
        self.shown_weights = None
        self.shown_term_topic_freq = None
        self.shown_marg_prob = None
        # should be used later for bar chart
        self.shown_ratio = None
        self.graph: Optional[BarPlotGraph] = None
        self._create_layout()

    def _create_layout(self):
        self._add_graph()
        box = gui.widgetBox(self.controlArea, "Relevance")
        self.rel_slider = gui.hSlider(
            box, self, "relevance", minValue=0, maxValue=1, step=0.1,
            intOnly=False, labelFormat="%.1f",
            callback_finished=self.on_params_change,
            createLabel=True)

        self.topic_box = gui.listBox(
            self.controlArea, self, "selected_topic", "topic_list",
            box="Topics",
            callback=self.on_params_change
        )

    def _add_graph(self):
        box = gui.vBox(self.mainArea, True, margin=0)
        self.graph = BarPlotGraph(self)
        box.layout().addWidget(self.graph)

    def __parameter_changed(self):
        self.graph.reset_graph()

    def compute_relevance(self, tp, mp):
        """
        Relevance is defined as lambda*log(topic_probability) + (
        1-lambda)*log(topic_probability/marginal_probability).
        https://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf

        :param tp: probability of a word in a topic
        :param mp: probability of a word in a corpus
        :return: adjusted probability
        """
        #
        return self.relevance * np.log(abs(tp)) + (1 - self.relevance) * \
               np.log(abs(tp) / mp) if (tp and mp) else 0

    def compute_weights(self, topic):
        return np.array([self.compute_relevance(weight, probability)
                         for weight, probability in zip(topic,
                                                        self.term_frequency)])

    def compute_distributions(self, topics):
        # term-topic column is multiplied by marginal topic probability
        # how likely is the term in a topic * how likely is the topic
        return topics * self.topic_frequency[:, None]

    def on_params_change(self):
        if self.data is None:
            return
        topic = self.data.X[:, self.selected_topic]
        words = self.data.metas[:, 0]
        adj_prob = self.compute_weights(topic)
        idx = np.argsort(adj_prob, axis=None)[::-1]
        self.shown_weights = np.around(adj_prob[idx][:N_BEST_PLOTTED], 5)
        self.shown_words = words[idx][:N_BEST_PLOTTED]
        self.shown_term_topic_freq = self.term_topic_matrix[
                                       self.selected_topic].T[idx][:N_BEST_PLOTTED]
        self.shown_marg_prob = self.term_frequency[idx][:N_BEST_PLOTTED]
        self.shown_ratio = [f"{a}/{b}" for a, b in zip(
            self.shown_term_topic_freq, self.shown_marg_prob)]

    @Inputs.topics
    def set_data(self, data):
        self.clear()
        if data is None:
            return
        if data.attributes["Model"] != "Latent Dirichlet Allocation":
            self.Error.wrong_model()
            return

        # test if the same as Marginal Topic Probability
        self.topic_frequency = data.get_column_view("Marginal Topic Probability")[0]
        self.data = Table.transpose(data, "Topics", "Words")
        self.topic_list = [var.name for var in self.data.domain.attributes]
        self.term_topic_matrix = self.compute_distributions(data.X)
        self.term_frequency = np.sum(self.term_topic_matrix, axis=0)
        # is this even a correct way to select the first row in the list? It
        # should probably consider settings?
        self.selected_topic = 0
        self.on_params_change()

    def handleNewSignals(self):
        self.setup_plot()

    def get_values(self) -> Optional[np.ndarray]:
        if not self.data:
            return None
        return self.shown_weights

    def get_labels(self) -> Optional[Union[List, np.ndarray]]:
        if not self.data:
            return None
        else:
            return self.shown_words

    def get_tooltip(self, index: int) -> str:
        if not self.data:
            return ""
        else:
            return self.shown_ratio[index]

    def setup_plot(self):
        self.graph.reset_graph()

    def clear(self):
        self.Error.clear()
        self.data = None
        self.topic_list = []
        self.topic_frequency = None
        self.term_topic_matrix = None
        self.term_frequency = None
        self.shown_words, self.shown_weights = None, None
        self.shown_ratio = None
        self.shown_term_topic_freq = None
        self.shown_marg_prob = None


if __name__ == "__main__":
    corpus = Corpus.from_file('deerwester')
    lda = LdaWrapper(num_topics=5)
    lda.fit_transform(corpus)
    topics = lda.get_all_topics_table()
    WidgetPreview(OWLDAvis).run(topics)
