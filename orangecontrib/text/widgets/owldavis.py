from typing import Optional, List
import numpy as np

from AnyQt.QtCore import Qt, QPointF
import pyqtgraph as pg

from PyQt5.QtGui import QColor, QPen
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
        self.marg_prob_item: Optional[pg.BarGraphItem] = None
        self.labels: List[str] = []
        super().__init__(
            parent=parent,
            viewBox=pg.ViewBox(),
            background="w", enableMenu=False,
            axisItems={"left": pg.AxisItem(orientation="left", rotate_ticks=False, pen=QPen(Qt.NoPen), ),
                       "top": pg.AxisItem(orientation="top", maxTickLength=0)}
        )
        self.hideAxis("left")
        self.hideAxis("top")
        self.hideAxis("bottom")

        self.getPlotItem().buttonsHidden = True
        self.getPlotItem().setContentsMargins(10, 10, 10, 10)
        self.getPlotItem().setMouseEnabled(x=False, y=False)

        self.tooltip_delegate = HelpEventDelegate(self.help_event)
        self.scene().installEventFilter(self.tooltip_delegate)

    def clear_all(self):
        self.clear()
        self.hideAxis("left")
        self.hideAxis("top")
        self.marg_prob_item = None

    def update_graph(self, words, term_topic_freq, marginal_probability):
        self.clear()
        marginal_probability = marginal_probability[::-1]
        term_topic_freq = term_topic_freq[::-1]
        words = words[::-1]

        self.marg_prob_item = pg.BarGraphItem(
            x0=0,
            y=np.arange(len(marginal_probability)),
            height=self.bar_width,
            width=marginal_probability,
            brushes=[QColor(Qt.gray) for _ in marginal_probability],
            pen = QColor(Qt.gray)
        )
        term_topic_freq_item = pg.BarGraphItem(
            x0=0,
            y=np.arange(len(term_topic_freq)),
            height=self.bar_width,
            width=term_topic_freq,
            brushes=[QColor(Qt.red) for _ in term_topic_freq],
            pen=QColor(Qt.red)
        )
        self.addItem(self.marg_prob_item)
        self.addItem(term_topic_freq_item)
        self.setXRange(1, marginal_probability.max(), padding=0)
        self.setYRange(0, len(marginal_probability) - 1)
        self.labels = [
            f"{w} - Term frequency: {tf:.3f}, Marginal probability{mp:.3f}"
            for w, tf, mp in zip(words, term_topic_freq, marginal_probability)
        ]

        self.update_axes(words)

    def update_axes(self, words):
        self.showAxis("left")
        self.showAxis("top")

        self.setLabel(axis="left", text="words")
        self.setLabel(axis="top", text="weights")

        ticks = [list(enumerate(words))]
        # todo: ticks lengths - labels can be long truncate them
        #  it can be done together with implementing plot settings
        self.getAxis("left").setTicks(ticks)

    def __get_index_at(self, p: QPointF):
        index = round(p.y())
        widths = self.marg_prob_item.opts["width"]
        if 0 <= index < len(widths) and abs(p.y() - index) <= self.bar_width / 2:
            width = widths[index]
            if 0 <= p.x() <= width:
                return index
        return None

    def help_event(self, ev: QGraphicsSceneHelpEvent):
        if self.marg_prob_item is None:
            return False

        index = self.__get_index_at(self.marg_prob_item.mapFromScene(ev.scenePos()))
        if index is not None:
            QToolTip.showText(ev.screenPos(), self.labels[index], widget=self)
            return True
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
        self.term_topic_matrix = None
        self.term_frequency = None
        self.num_tokens = None
        # should be used later for bar chart
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
        self.graph = BarPlotGraph(self)
        self.mainArea.layout().addWidget(self.graph)

    def compute_relevance(self, topic):
        """
        Relevance is defined as lambda*log(topic_probability) + (
        1-lambda)*log(topic_probability/marginal_probability).
        https://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf
        """
        nonzero = (topic > 0) & (self.term_frequency > 0)
        tp, mp = topic[nonzero], self.term_frequency[nonzero]
        adj_prob = np.zeros(topic.shape)
        adj_prob[nonzero] = self.relevance * np.log(tp) + (1 - self.relevance) * np.log(tp / mp)
        return adj_prob

    @staticmethod
    def compute_distributions(data):
        """
        term-topic column is multiplied by marginal topic probability
        how likely is the term in a topic * how likely is the topic
        """
        topic_frequency = data.get_column_view("Marginal Topic Probability")[0].astype(float)
        return data.X * topic_frequency[:, None]

    def on_params_change(self):
        if self.data is None:
            return
        topic = self.data.X[:, self.selected_topic]
        adj_prob = self.compute_relevance(topic)

        idx = np.argsort(adj_prob, axis=None)[::-1][:N_BEST_PLOTTED]

        words = self.data.metas[:, 0][idx]
        term_topic_freq = self.term_topic_matrix[self.selected_topic].T[idx]
        marg_prob = self.term_frequency[idx]

        # convert to absolute frequencies
        term_topic_freq = term_topic_freq * self.num_tokens
        marg_prob = marg_prob * self.num_tokens

        self.graph.update_graph(words, term_topic_freq, marg_prob)

    @Inputs.topics
    def set_data(self, data):
        self.clear()
        if data is None:
            return
        if data.attributes.get("Model", "") != "Latent Dirichlet Allocation":
            self.Error.wrong_model()
            return

        self.data = Table.transpose(data, "Topics", "Words")
        self.topic_list = [var.name for var in self.data.domain.attributes]
        self.num_tokens = data.attributes.get("Number of tokens", "")
        self.term_topic_matrix = self.compute_distributions(data)
        self.term_frequency = np.sum(self.term_topic_matrix, axis=0)

        # workaround: selected_topic is not marked after topic list is redefined
        # todo: find and fix the bug on the listview
        self.selected_topic = self.selected_topic

        self.on_params_change()

    def clear(self):
        self.Error.clear()
        self.graph.clear_all()
        self.data = None
        self.topic_list = []
        self.term_topic_matrix = None
        self.term_frequency = None
        self.num_tokens = None

    def send_report(self):
        self.report_items((
            ("Relevance", self.relevance),
            ("Shown topic", self.topic_list[self.selected_topic])
        ))
        self.report_plot()


if __name__ == "__main__":
    corpus = Corpus.from_file('deerwester')
    lda = LdaWrapper(num_topics=5)
    lda.fit_transform(corpus, chunk_number=100)
    topics = lda.get_all_topics_table()

    WidgetPreview(OWLDAvis).run(topics)
