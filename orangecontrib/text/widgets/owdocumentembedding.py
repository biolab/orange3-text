from typing import Any, Tuple
import numpy as np

from AnyQt.QtWidgets import QPushButton, QStyle, QLayout
from AnyQt.QtCore import Qt, QSize

from Orange.widgets.gui import widgetBox, comboBox, auto_commit, hBox
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget, Msg, Input, Output
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState

from Orange.misc.utils.embedder_utils import EmbeddingConnectionError

from orangecontrib.text.vectorization.document_embedder import DocumentEmbedder
from orangecontrib.text.vectorization.document_embedder import LANGS_TO_ISO, AGGREGATORS
from orangecontrib.text.corpus import Corpus


def run_pretrained_embedder(corpus: Corpus,
                            language: str,
                            aggregator: str,
                            state: TaskState) -> Tuple[Corpus, Corpus]:
    """Runs DocumentEmbedder.

    Parameters
    ----------
    corpus : Corpus
        Corpus on which transform is performed.
    language : str
        ISO 639-1 (two-letter) code of desired language.
    aggregator : str
        Aggregator which creates document embedding (single
        vector) from word embeddings (multiple vectors).
        Allowed values are mean, sum, max, min.
    state : TaskState
        State object.

    Returns
    -------
    Corpus
        New corpus with additional features.
    """

    embedder = DocumentEmbedder(language=language,
                                aggregator=aggregator)

    ticks = iter(np.linspace(0., 100., len(corpus)))

    def advance(success=True):
        if state.is_interruption_requested():
            embedder.set_cancelled()
        if success:
            state.set_progress_value(next(ticks))

    new_corpus, skipped_corpus = embedder(corpus, processed_callback=advance)
    return new_corpus, skipped_corpus


class OWDocumentEmbedding(OWWidget, ConcurrentWidgetMixin):
    name = "Document Embedding"
    description = "Document embedding using pretrained models."
    keywords = ['embedding', 'document embedding', 'text']
    icon = 'icons/TextEmbedding.svg'
    priority = 300

    want_main_area = False
    _auto_apply = Setting(default=True)

    class Inputs:
        corpus = Input('Corpus', Corpus)

    class Outputs:
        new_corpus = Output('Embeddings', Corpus, default=True)
        skipped = Output('Skipped documents', Corpus)

    class Error(OWWidget.Error):
        no_connection = Msg("No internet connection. " +
                            "Please establish a connection or " +
                            "use another vectorizer.")
        unexpected_error = Msg('Embedding error: {}')

    class Warning(OWWidget.Warning):
        unsuccessful_embeddings = Msg('Some embeddings were unsuccessful.')

    language = Setting(default=0)
    aggregator = Setting(default=0)

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)

        self.languages = sorted(list(LANGS_TO_ISO.keys()))
        self.aggregators = AGGREGATORS
        self.corpus = None
        self.new_corpus = None
        self._setup_layout()

    @staticmethod
    def sizeHint():
        return QSize(300, 300)

    def _setup_layout(self):
        self.controlArea.setMinimumWidth(self.sizeHint().width())
        self.layout().setSizeConstraint(QLayout.SetFixedSize)

        widget_box = widgetBox(self.controlArea, 'Settings')

        self.language_cb = comboBox(
            widget=widget_box,
            master=self,
            value='language',
            label='Language: ',
            orientation=Qt.Horizontal,
            items=self.languages,
            callback=self._option_changed,
            searchable=True
         )
        self.language_cb.setCurrentText("English")

        self.aggregator_cb = comboBox(widget=widget_box,
                                      master=self,
                                      value='aggregator',
                                      label='Aggregator: ',
                                      orientation=Qt.Horizontal,
                                      items=self.aggregators,
                                      callback=self._option_changed)

        self.auto_commit_widget = auto_commit(widget=self.controlArea,
                                              master=self,
                                              value='_auto_apply',
                                              label='Apply',
                                              commit=self.commit,
                                              box=False)

        self.cancel_button = QPushButton(
            'Cancel',
            icon=self.style()
            .standardIcon(QStyle.SP_DialogCancelButton))

        self.cancel_button.clicked.connect(self.cancel)

        hbox = hBox(self.controlArea)
        hbox.layout().addWidget(self.cancel_button)
        self.cancel_button.setDisabled(True)

    def set_input_corpus_summary(self, corpus):
        if corpus is None:
            self.info.set_input_summary(self.info.NoInput)
        else:
            self.info.set_input_summary(str(len(corpus)), "{} documents."
                                        .format(len(corpus)))

    def set_output_corpus_summary(self, embeddings, skipped):
        if embeddings is None and skipped is None:
            self.info.set_output_summary(self.info.NoOutput)
        else:
            successful = len(embeddings) if embeddings else 0
            unsuccessful = len(skipped) if skipped else 0
            if unsuccessful > 0:
                self.Warning.unsuccessful_embeddings()
            self.info.set_output_summary(
                f"{successful}|{unsuccessful}",
                "Successful: {}, Unsuccessful: {}".format(
                    successful, unsuccessful
                )
            )

    @Inputs.corpus
    def set_data(self, data):
        self.Warning.clear()
        self.set_input_corpus_summary(data)
        self.cancel()

        if not data:
            self.corpus = None
            self.clear_outputs()
            return

        self.corpus = data
        self.unconditional_commit()

    def _option_changed(self):
        self.commit()

    def commit(self):
        if self.corpus is None:
            self.clear_outputs()
            return

        self.cancel_button.setDisabled(False)

        self.start(run_pretrained_embedder,
                   self.corpus,
                   LANGS_TO_ISO[self.languages[self.language]],
                   self.aggregators[self.aggregator])

        self.Error.clear()

    def on_done(self, embeddings: Tuple[Corpus, Corpus]) -> None:
        self.cancel_button.setDisabled(True)
        self._send_output_signals(embeddings[0], embeddings[1])

    def on_partial_result(self, result: Any):
        self.cancel()
        self.Error.no_connection()

    def on_exception(self, ex: Exception):
        self.cancel_button.setDisabled(True)
        if isinstance(ex, EmbeddingConnectionError):
            self.Error.no_connection()
        else:
            self.Error.unexpected_error(type(ex).__name__)
        self.cancel()
        self.clear_outputs()

    def cancel(self):
        self.cancel_button.setDisabled(True)
        super().cancel()

    def _send_output_signals(self, embeddings, skipped):
        self.Outputs.new_corpus.send(embeddings)
        self.Outputs.skipped.send(skipped)
        self.set_output_corpus_summary(embeddings, skipped)

    def clear_outputs(self):
        self._send_output_signals(None, None)

    def onDeleteWidget(self):
        self.cancel()
        super().onDeleteWidget()


if __name__ == '__main__':
    from orangewidget.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWDocumentEmbedding).run(Corpus.from_file('book-excerpts'))
