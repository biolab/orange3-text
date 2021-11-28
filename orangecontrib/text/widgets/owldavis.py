from typing import Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from AnyQt.QtCore import QPointF, Qt
from PyQt5.QtGui import QColor, QPen
from PyQt5.QtWidgets import QGraphicsSceneHelpEvent, QToolTip
from pyqtgraph import LabelItem, AxisItem
from pyqtgraph.graphicsItems.LegendItem import ItemSample

from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.visualize.owscatterplotgraph import LegendItem
from Orange.widgets.visualize.utils.customizableplot import (
    CommonParameterSetter,
    Updater,
)
from Orange.widgets.visualize.utils.plotutils import HelpEventDelegate
from Orange.widgets.widget import Input, OWWidget
from orangewidget.settings import Setting, SettingProvider
from orangewidget.utils.visual_settings_dlg import (
    KeyType,
    ValueType,
    VisualSettingsDialog,
)
from orangewidget.utils.widgetpreview import WidgetPreview
from orangewidget.widget import Msg

from orangecontrib.text.corpus import Corpus
from orangecontrib.text.topics import LdaWrapper
from orangecontrib.text.topics.topics import Topics


N_BEST_PLOTTED = 20


class ParameterSetter(CommonParameterSetter):
    GRID_LABEL, SHOW_GRID_LABEL = "Gridlines", "Show"
    DEFAULT_ALPHA_GRID, DEFAULT_SHOW_GRID = 80, False

    def __init__(self, master):
        self.grid_settings: Optional[Dict] = None
        self.master: BarPlotGraph = master
        super().__init__()

    def update_setters(self) -> None:
        self.grid_settings = {
            Updater.ALPHA_LABEL: self.DEFAULT_ALPHA_GRID,
            self.SHOW_GRID_LABEL: self.DEFAULT_SHOW_GRID,
        }

        self.initial_settings = {
            self.LABELS_BOX: {
                self.FONT_FAMILY_LABEL: self.FONT_FAMILY_SETTING,
                self.AXIS_TITLE_LABEL: self.FONT_SETTING,
                self.AXIS_TICKS_LABEL: self.FONT_SETTING,
                self.LEGEND_LABEL: self.FONT_SETTING,
            },
            self.PLOT_BOX: {
                self.GRID_LABEL: {
                    self.SHOW_GRID_LABEL: (None, self.DEFAULT_SHOW_GRID),
                    Updater.ALPHA_LABEL: (range(0, 255, 5), self.DEFAULT_ALPHA_GRID),
                },
            },
        }

        def update_grid(**settings):
            self.grid_settings.update(**settings)
            self.master.showGrid(
                x=self.grid_settings[self.SHOW_GRID_LABEL],
                alpha=self.grid_settings[Updater.ALPHA_LABEL] / 255,
            )

        self._setters[self.PLOT_BOX] = {self.GRID_LABEL: update_grid}

    @property
    def axis_items(self) -> List[AxisItem]:
        return [value["item"] for value in self.master.getPlotItem().axes.values()]

    @property
    def legend_items(self) -> List[Tuple[ItemSample, LabelItem]]:
        return self.master.legend.items


class BarPlotGraph(pg.PlotWidget):
    bar_width = 0.7
    colors = {
        "Overall term frequency": QColor(Qt.gray),
        "Term frequency within topic": QColor(Qt.red),
    }

    def __init__(self, master, parent=None):
        self.master: OWLDAvis = master
        self.parameter_setter = ParameterSetter(self)
        self.marg_prob_item: Optional[pg.BarGraphItem] = None
        self.term_topic_freq_item: Optional[pg.BarGraphItem] = None

        self.labels: List[str] = []
        super().__init__(
            parent=parent,
            viewBox=pg.ViewBox(),
            background="w",
            enableMenu=False,
            axisItems={
                "left": pg.AxisItem(
                    orientation="left", rotate_ticks=False, pen=QPen(Qt.NoPen)
                ),
                "top": pg.AxisItem(orientation="top"),
            },
        )
        self.hideAxis("left")
        self.hideAxis("top")
        self.hideAxis("bottom")

        self.getPlotItem().buttonsHidden = True
        self.getPlotItem().setContentsMargins(10, 10, 10, 10)
        self.getPlotItem().setMouseEnabled(x=False, y=False)
        self.showGrid(
            x=self.parameter_setter.DEFAULT_SHOW_GRID,
            alpha=self.parameter_setter.DEFAULT_ALPHA_GRID / 255,
        )

        self.tooltip_delegate = HelpEventDelegate(self.help_event)
        self.scene().installEventFilter(self.tooltip_delegate)
        self.legend = self._create_legend()

    def clear_all(self) -> None:
        self.clear()
        self.hideAxis("left")
        self.hideAxis("top")
        self.marg_prob_item = None
        self.legend.hide()

    def update_graph(
        self,
        words: List[str],
        term_topic_freq: np.ndarray,
        marginal_probability: np.ndarray,
    ) -> None:
        self.clear()
        marginal_probability = marginal_probability[::-1]
        term_topic_freq = term_topic_freq[::-1]
        words = words[::-1]

        self.marg_prob_item = pg.BarGraphItem(
            x0=0,
            y=np.arange(len(marginal_probability)),
            height=self.bar_width,
            width=marginal_probability,
            brushes=[
                self.colors["Overall term frequency"] for _ in marginal_probability
            ],
            pen=self.colors["Overall term frequency"],
        )
        self.term_topic_freq_item = pg.BarGraphItem(
            x0=0,
            y=np.arange(len(term_topic_freq)),
            height=self.bar_width,
            width=term_topic_freq,
            brushes=[
                self.colors["Term frequency within topic"] for _ in term_topic_freq
            ],
            pen=self.colors["Term frequency within topic"],
        )
        self.addItem(self.marg_prob_item)
        self.addItem(self.term_topic_freq_item)
        self.setXRange(1, marginal_probability.max(), padding=0)
        self.setYRange(0, len(marginal_probability) - 1)
        self.labels = [
            f"{w} - Frequency withing topic: {tf:.3f}, Overall frequency {mp:.3f}"
            for w, tf, mp in zip(words, term_topic_freq, marginal_probability)
        ]

        self.update_axes(words)
        self.update_legend()

    def update_axes(self, words: List[str]) -> None:
        self.showAxis("left")
        self.showAxis("top")

        self.setLabel(axis="left", text="words")
        self.setLabel(axis="top", text="weights")
        self.getAxis("left").setTextPen(QPen(Qt.black))
        self.getAxis("top").setTextPen(QPen(Qt.black))
        self.getAxis("top").setPen(QPen(Qt.black))

        ticks = [list(enumerate(words))]
        self.getAxis("left").setTicks(ticks)

    def _create_legend(self) -> LegendItem:
        legend = LegendItem()
        legend.setParentItem(self.getViewBox())
        legend.anchor((1, 1), (1, 1), offset=(-1, -1))
        legend.hide()
        return legend

    def update_legend(self) -> None:
        self.legend.clear()
        for text, c in self.colors.items():
            dot = pg.ScatterPlotItem(pen=pg.mkPen(color=c), brush=pg.mkBrush(color=c))
            self.legend.addItem(dot, text)
            self.legend.show()
        Updater.update_legend_font(
            self.legend.items, **self.parameter_setter.legend_settings
        )

    def __get_index_at(self, p: QPointF) -> Optional[int]:
        index = round(p.y())
        widths = self.marg_prob_item.opts["width"]
        if 0 <= index < len(widths) and abs(p.y() - index) <= self.bar_width / 2:
            width = widths[index]
            if 0 <= p.x() <= width:
                return index
        return None

    def help_event(self, ev: QGraphicsSceneHelpEvent) -> bool:
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
    visual_settings = Setting({}, schema_only=True)

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

        VisualSettingsDialog(self, self.graph.parameter_setter.initial_settings)

    def _create_layout(self):
        self._add_graph()
        box = gui.widgetBox(self.controlArea, "Relevance")
        self.rel_slider = gui.hSlider(
            box,
            self,
            "relevance",
            minValue=0,
            maxValue=1,
            step=0.1,
            intOnly=False,
            labelFormat="%.1f",
            callback_finished=self.on_params_change,
            createLabel=True,
        )

        self.topic_box = gui.listBox(
            self.controlArea,
            self,
            "selected_topic",
            "topic_list",
            box="Topics",
            callback=self.on_params_change,
        )

    def _add_graph(self):
        self.graph = BarPlotGraph(self)
        self.mainArea.layout().addWidget(self.graph)

    def compute_relevance(self, topic: np.ndarray) -> np.ndarray:
        """
        Relevance is defined as lambda*log(topic_probability) + (
        1-lambda)*log(topic_probability/marginal_probability).
        https://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf
        """
        nonzero = (topic > 0) & (self.term_frequency > 0)
        tp, mp = topic[nonzero], self.term_frequency[nonzero]
        adj_prob = np.zeros(topic.shape)
        rel = self.relevance
        adj_prob[nonzero] = rel * np.log(tp) + (1 - rel) * np.log(tp / mp)
        return adj_prob

    @staticmethod
    def compute_distributions(data: Topics) -> np.ndarray:
        """
        Compute how likely is the term in each topic
        Term-topic column is multiplied by marginal topic probability
        """
        topic_frequency = data.get_column_view("Marginal Topic Probability")[0]
        return data.X * topic_frequency[:, None].astype(float)

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
    def set_data(self, data: Optional[Topics]):
        prev_topic = self.selected_topic
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

        self.selected_topic = prev_topic if prev_topic < len(self.topic_list) else 0
        self.on_params_change()

    def set_visual_settings(self, key: KeyType, value: ValueType):
        self.graph.parameter_setter.set_parameter(key, value)
        self.visual_settings[key] = value

    def clear(self):
        self.Error.clear()
        self.graph.clear_all()
        self.data = None
        self.topic_list = []
        self.term_topic_matrix = None
        self.term_frequency = None
        self.num_tokens = None

    def send_report(self):
        self.report_items(
            (
                ("Relevance", self.relevance),
                ("Shown topic", self.topic_list[self.selected_topic]),
            )
        )
        self.report_plot()


if __name__ == "__main__":
    corpus = Corpus.from_file("deerwester")
    lda = LdaWrapper(num_topics=5)
    lda.fit_transform(corpus)
    topics = lda.get_all_topics_table()

    WidgetPreview(OWLDAvis).run(topics)
