from typing import Tuple, Any
from AnyQt.QtWidgets import QLayout
from AnyQt.QtCore import QSize

from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.settings import Setting
from Orange.widgets.gui import widgetBox, comboBox, spin, button
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.data import Table

try:
    from orangecontrib.network import Network
    from orangecontrib.text.corpus_to_network import CorpusToNetwork
except Exception:
    Network = None
    CorpusToNetwork = None
from orangecontrib.text import Corpus


def run(
    corpus_to_network: CorpusToNetwork,
    document_nodes: bool,
    threshold: int,
    window_size: int,
    freq_threshold: int,
    state: TaskState,
) -> Tuple[Network, Table]:
    def advance(progress):
        if state.is_interruption_requested():
            raise InterruptedError
        state.set_progress_value(progress)

    network = corpus_to_network(
        document_nodes=document_nodes,
        window_size=window_size,
        threshold=threshold,
        freq_threshold=freq_threshold,
        progress_callback=advance,
    )
    items = corpus_to_network.get_current_items(document_nodes)

    return (network, items)


class OWCorpusToNetwork(OWWidget, ConcurrentWidgetMixin):
    name = "Corpus to Network"
    description = "Constructs network from given corpus."
    keywords = ["text, network"]
    icon = "icons/CorpusToNetwork.svg"
    priority = 250

    want_main_area = False
    _auto_apply = Setting(default=True)
    node_type = Setting(default=0)
    threshold = Setting(default=1)
    window_size = Setting(default=1)
    freq_threshold = Setting(default=1)

    class Inputs:
        corpus = Input("Corpus", Corpus)

    class Outputs:
        network = Output("Network", "orangecontrib.network.Network")
        items = Output("Node Data", Table)

    class Error(OWWidget.Error):
        unexpected_error = Msg("Unknown error: {}")
        no_network_addon = Msg(
            "Please install network add-on to use this widget."
        )

    class Information(OWWidget.Information):
        params_changed = Msg(
            "Parameters have been changed. Press Start to"
            + " run with new parameters."
        )

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)

        self._corpus_to_network = None
        self.corpus = None
        self.node_types = ["Document", "Word"]
        self._task_state = "waiting"
        self._setup_layout()
        if Network is None:
            self.Error.no_network_addon()

    @staticmethod
    def sizeHint():
        return QSize(300, 300)

    def _setup_layout(self):
        self.controlArea.setMinimumWidth(self.sizeHint().width())
        self.layout().setSizeConstraint(QLayout.SetFixedSize)

        widget_box = widgetBox(self.controlArea, "Settings")

        self.node_type_cb = comboBox(
            widget=widget_box,
            master=self,
            value="node_type",
            label="Node type",
            items=self.node_types,
            callback=self._option_changed,
        )

        self.threshold_spin = spin(
            widget=widget_box,
            master=self,
            value="threshold",
            minv=1,
            maxv=10000,
            step=1,
            label="Threshold",
            callback=self._option_changed,
        )

        self.window_size_spin = spin(
            widget=widget_box,
            master=self,
            value="window_size",
            minv=1,
            maxv=1000,
            step=1,
            label="Window size",
            callback=self._option_changed,
        )

        self.freq_thresold_spin = spin(
            widget=widget_box,
            master=self,
            value="freq_threshold",
            minv=1,
            maxv=10000,
            step=1,
            label="Frequency Threshold",
            callback=self._option_changed,
        )

        self.button = button(
            widget=self.controlArea,
            master=self,
            label="Start",
            callback=self._toggle,
        )

        self._option_changed()

    @Inputs.corpus
    def set_data(self, data):
        if Network is None:
            self.Error.no_network_addon()
            return
        self.cancel()
        self._task_state = "running"
        self.button.setText("Stop")
        if not data:
            self._corpus_to_network = None
            self.info.set_input_summary(self.info.NoInput)
            self.clear_outputs()
            return

        self.corpus = data
        summary = str(len(self.corpus))
        details = "Corpus with {} documents.".format(len(self.corpus))
        self.info.set_input_summary(summary, details)
        self._corpus_to_network = CorpusToNetwork(corpus=data)
        self.commit()

    def commit(self):
        if Network is None:
            self.Error.no_network_addon()
            return
        if self.corpus is None:
            self.clear_outputs()
            return

        self.Error.clear()
        self.Information.params_changed(shown=False)

        self.start(
            run,
            self._corpus_to_network,
            (self.node_type == 0),
            self.threshold,
            self.window_size,
            self.freq_threshold,
        )

    def _option_changed(self):
        word_active = self.node_type == 1
        self.window_size_spin.setDisabled(not word_active)
        self.freq_thresold_spin.setDisabled(not word_active)
        self.Information.params_changed(shown=(self._task_state == "running"))
        self.cancel()

    def _toggle(self):
        if Network is None:
            self.Error.no_network_addon()
            return
        if self._task_state == "waiting":
            self._task_state = "running"
            self.button.setText("Stop")
            self.commit()
        else:
            self.cancel()

    def clear_outputs(self):
        self._send_output_signals([None, None])

    def _send_output_signals(self, result):
        self.Outputs.network.send(result[0])
        self.Outputs.items.send(result[1])

    def cancel(self):
        self._task_state = "waiting"
        self.button.setText("Start")
        super().cancel()

    def on_done(self, result: Any) -> None:
        self._task_state = "waiting"
        self.button.setText("Start")
        network = result[0]
        self._send_output_signals(result)
        nodes = network.number_of_nodes()
        edges = network.number_of_edges()
        summary = "{} / {}".format(nodes, edges)
        directed = "Directed" if network.edges[0].directed else "Undirected"
        details = "{} network with {} nodes and {} edges.".format(
            directed, nodes, edges
        )
        self.info.set_output_summary(summary, details)

    def on_partial_result(self, result: Any):
        self.cancel()

    def on_exception(self, ex: Exception):
        self.Error.unexpected_error(type(ex).__name__)
        self.cancel()

    def onDeleteWidget(self):
        del self._corpus_to_network
        super().onDeleteWidget()


if __name__ == "__main__":
    from orangewidget.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWCorpusToNetwork).run(Corpus.from_file("book-excerpts"))
