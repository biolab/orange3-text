# coding: utf-8
import numpy as np

from AnyQt.QtCore import Qt

from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.utils.itemmodels import PyTableModel
from Orange.widgets.widget import Input, OWWidget
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsLinearLayout
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.topics import LdaWrapper
from orangecontrib.text.topics.topics import Topics
from orangewidget.gui import TableView
from orangewidget.settings import Setting
from orangewidget.utils.widgetpreview import WidgetPreview

from Orange.widgets.data.utils.histogram import ProportionalBarItem
from orangewidget.widget import Msg

N_BEST_PLOTTED = 20


class TableModel(PyTableModel):
    def __init__(self, precision, **kwargs):
        super().__init__(**kwargs)
        self.precision = precision

    def data(self, index, role=Qt.DisplayRole):
        """
        Format numbers of the first column with the number of decimal
        spaces defined by self.precision which can be changed based on
        weights type - row counts does not have decimal spaces
        """
        row, column = self.mapToSourceRows(index.row()), index.column()
        if role == Qt.DisplayRole and column == 0:
            value = float(self[row][column])
            return f"{value:.{self.precision}f}"
        if role == Qt.DisplayRole and column == 2:
            # value = float(self[row][column])
            # bar_layout = QGraphicsLinearLayout(Qt.Horizontal)
            # bar_layout.setSpacing(0)
            # bar_layout.addStretch()
            # bar = ProportionalBarItem(  # pylint: disable=blacklisted-name
            #     distribution=np.array([value, 1]),
            #     colors=[QColor(Qt.blue), QColor(Qt.red)],
            #     height=1
            # )
            # bar_layout.addItem(bar)
            # return bar_layout
            return f"{self[row][column]}"

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


class OWRelevantTerms(OWWidget):
    name = "Relevant Terms"
    priority = 410
    icon = "icons/RelevantTerms.svg"

    selected_topic = Setting(0, schema_only=True)
    relevance = Setting(0.5)

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
        # should be used later for bar chart
        self.shown_ratio = None
        self._create_layout()

    def _create_layout(self):
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

        box = gui.widgetBox(self.mainArea, "Words && weights")
        # insert list of words into mainArea
        view = self.tableview = TableView(self)
        model = self.tablemodel = TableModel(precision=5, parent=self)
        model.setHorizontalHeaderLabels(["Weight", "Word", "Distribution"])
        view.setModel(model)
        box.layout().addWidget(view)

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
        self.shown_weights = adj_prob[idx][:N_BEST_PLOTTED]
        self.shown_words = words[idx][:N_BEST_PLOTTED]
        term_topic_freq = self.term_topic_matrix[self.selected_topic].T[idx][
                          :N_BEST_PLOTTED]
        marg_prob = self.term_frequency[idx][:N_BEST_PLOTTED]
        self.shown_ratio = [f"{a:.2f}/{b:.2f}" for a, b in zip(
            term_topic_freq, marg_prob)]
        self.repopulate_table()

    def repopulate_table(self):
        self.tablemodel.wrap(list(zip(self.shown_weights, self.shown_words,
                                      self.shown_ratio)))
        self.tableview.sortByColumn(0, Qt.DescendingOrder)

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

    # TODO: check behaviour when None on input
    def clear(self):
        self.Error.clear()
        self.tablemodel.clear()
        self.tableview.update()
        self.data = None
        self.topic_list = []
        self.topic_frequency = None
        self.term_topic_matrix = None
        self.term_frequency = None


if __name__ == "__main__":
    corpus = Corpus.from_file('deerwester')
    lda = LdaWrapper(num_topics=5)
    lda.fit_transform(corpus)
    topics = lda.get_all_topics_table()
    WidgetPreview(OWRelevantTerms).run(topics)
