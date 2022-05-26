from typing import Any

from AnyQt.QtWidgets import QGroupBox

from Orange.widgets import gui
from Orange.widgets import settings
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.widget import OWWidget, Input, Output
from orangecontrib.text.corpus import Corpus


class Vectorizer:
    new_corpus = None
    new_attributes = {}
    method = None

    def __init__(self, method, corpus):
        self.method = method
        self.corpus = corpus

    def _hide_attrs(self, hidden: bool) -> None:
        if self.new_corpus is not None:
            new_attrs = {f.name for f in self.new_corpus.domain.attributes} - {
                f.name for f in self.corpus.domain.attributes
            }
            new_domain = self.new_corpus.domain
            for f in new_domain.attributes:
                if f.name in new_attrs:
                    f.attributes["hidden"] = hidden
            self.new_corpus = self.new_corpus.transform(new_domain)

    def _transform(self, callback):
        self.new_corpus = self.method.transform(self.corpus, callback=callback)

    def run(self, hidden: bool, task_state: TaskState):
        def callback(progress):
            if task_state.is_interruption_requested():
                raise Exception
            task_state.set_progress_value(progress * 100)

        if self.new_corpus is None:
            self._transform(callback)
        self._hide_attrs(hidden)


class OWBaseVectorizer(OWWidget, ConcurrentWidgetMixin, openclass=True):
    """A base class for feature extraction methods."""

    # Input/output
    class Inputs:
        corpus = Input("Corpus", Corpus)

    class Outputs:
        corpus = Output("Corpus", Corpus, default=True, replaces=["Embeddings"])

    want_main_area = False
    resizing_enabled = False

    # Settings
    autocommit = settings.Setting(True)
    hidden_cb = settings.Setting(True)

    Method = NotImplemented

    def __init__(self):
        super().__init__()
        ConcurrentWidgetMixin.__init__(self)
        self.corpus = None
        self.vectorizer = None

        box = QGroupBox(title="Options")
        box.setLayout(self.create_configuration_layout())
        self.controlArea.layout().addWidget(box)

        output_layout = gui.hBox(self.controlArea)
        gui.checkBox(
            output_layout,
            self,
            "hidden_cb",
            "Hide bow attributes",
            callback=self.commit.deferred,
        )

        gui.auto_commit(self.buttonsArea, self, "autocommit", "Commit")
        self.update_method()

    @Inputs.corpus
    def set_data(self, data):
        self.corpus = data
        self.update_method()
        self.commit.now()

    @gui.deferred
    def commit(self):
        if self.corpus is not None:
            self.start(self.vectorizer.run, self.hidden_cb)
        else:
            self.cancel()

    def on_done(self, _) -> None:
        self.Outputs.corpus.send(self.vectorizer.new_corpus)

    def on_partial_result(self, result: Any):
        pass

    def on_exception(self, ex: Exception):
        raise ex

    def update_method(self):
        self.vectorizer = Vectorizer(self.init_method(), self.corpus)

    def init_method(self):
        raise NotImplementedError

    def on_change(self):
        self.update_method()
        self.commit.deferred()

    def send_report(self):
        self.report_items(self.method.report())

    def create_configuration_layout(self):
        raise NotImplementedError

    def cancel(self):
        self.Outputs.corpus.send(None)
        super().cancel()
