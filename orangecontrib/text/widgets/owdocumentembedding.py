from typing import Dict, Optional, Any

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QVBoxLayout, QPushButton, QStyle
from Orange.misc.utils.embedder_utils import EmbeddingConnectionError
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output, OWWidget, Message, Msg

from orangecontrib.text.corpus import Corpus
from orangecontrib.text.language import (
    ISO2LANG, DEFAULT_LANGUAGE, LanguageModel, LANG2ISO
)
from orangecontrib.text.vectorization.document_embedder import (
    AGGREGATORS,
    AGGREGATORS_ITEMS,
    DocumentEmbedder,
    LANGUAGES,
)
from orangecontrib.text.vectorization.sbert import SBERT
from orangecontrib.text.widgets.utils.owbasevectorizer import (
    OWBaseVectorizer,
    Vectorizer,
)


class EmbeddingVectorizer(Vectorizer):
    skipped_documents = None

    def _transform(self, callback):
        embeddings, skipped = self.method.transform(self.corpus, callback=callback)
        self.new_corpus = embeddings
        self.skipped_documents = skipped


class OWDocumentEmbedding(OWBaseVectorizer):
    name = "Document Embedding"
    description = "Document embedding using pretrained models."
    keywords = "embedding, document embedding, fasttext, bert, sbert"
    icon = "icons/TextEmbedding.svg"
    priority = 300

    UserAdviceMessages = [
        Message(
            "This widget sends documents to an external server. Avoid using it with sensitive data.",
            "privacy_warning"
        )
    ]

    buttons_area_orientation = Qt.Vertical
    settings_version = 3

    Methods = [SBERT, DocumentEmbedder]

    class Outputs(OWBaseVectorizer.Outputs):
        skipped = Output("Skipped documents", Corpus)

    class Error(OWWidget.Error):
        no_connection = Msg(
            "No internet connection. Please establish a connection or use "
            "another vectorizer."
        )
        unexpected_error = Msg("Embedding error: {}")

    class Warning(OWWidget.Warning):
        unsuccessful_embeddings = Msg("Some embeddings were unsuccessful.")

    method: int = Setting(default=0)
    language: str = Setting(default=DEFAULT_LANGUAGE, schema_only=True)
    aggregator: str = Setting(default="Mean")

    def __init__(self):
        super().__init__()
        self.cancel_button = QPushButton(
            "Cancel", icon=self.style().standardIcon(QStyle.SP_DialogCancelButton)
        )
        self.cancel_button.clicked.connect(self.cancel)
        self.buttonsArea.layout().addWidget(self.cancel_button)
        self.cancel_button.setDisabled(True)
        # it should be only set when setting loaded from schema/workflow
        self.__pending_language = self.language

    def create_configuration_layout(self):
        layout = QVBoxLayout()
        rbtns = gui.radioButtons(None, self, "method", callback=self.on_change)
        layout.addWidget(rbtns)

        gui.appendRadioButton(rbtns, "Multilingual SBERT")
        gui.appendRadioButton(rbtns, "fastText:")
        ibox = gui.indentedBox(rbtns)
        self.language_cb = gui.comboBox(
            ibox,
            self,
            "language",
            model=LanguageModel(languages=LANGUAGES),
            label="Language:",
            sendSelectedValue=True,  # value is actual string not index
            orientation=Qt.Horizontal,
            callback=self.on_change,
            searchable=True,
        )
        self.aggregator_cb = gui.comboBox(
            ibox,
            self,
            "aggregator",
            items=AGGREGATORS_ITEMS,
            label="Aggregator:",
            sendSelectedValue=True,  # value is actual string not index
            orientation=Qt.Horizontal,
            callback=self.on_change,
            searchable=True,
        )
        return layout

    @OWBaseVectorizer.Inputs.corpus
    def set_data(self, corpus):
        # set language from corpus as selected language
        if corpus and corpus.language in LANGUAGES:
            self.language = corpus.language
        else:
            # if Corpus's language not supported use default language
            self.language = DEFAULT_LANGUAGE

        # when workflow loaded use language saved in workflow
        if self.__pending_language is not None:
            self.language = self.__pending_language
            self.__pending_language = None

        super().set_data(corpus)

    def update_method(self):
        disabled = self.method == 0
        self.aggregator_cb.setDisabled(disabled)
        self.language_cb.setDisabled(disabled)
        self.vectorizer = EmbeddingVectorizer(self.init_method(), self.corpus)

    def init_method(self):
        params = dict(language=self.language, aggregator=self.aggregator)
        kwargs = ({}, params)[self.method]
        return self.Methods[self.method](**kwargs)

    @gui.deferred
    def commit(self):
        self.Error.clear()
        self.Warning.clear()
        self.cancel_button.setDisabled(False)
        super().commit()

    def on_done(self, result):
        self.cancel_button.setDisabled(True)
        skipped = self.vectorizer.skipped_documents
        self.Outputs.skipped.send(skipped)
        if skipped is not None and len(skipped) > 0:
            self.Warning.unsuccessful_embeddings()
        super().on_done(result)

    def on_exception(self, ex: Exception):
        self.cancel_button.setDisabled(True)
        if isinstance(ex, EmbeddingConnectionError):
            self.Error.no_connection()
        else:
            self.Error.unexpected_error(str(ex))
        self.cancel()

    def cancel(self):
        self.Outputs.skipped.send(None)
        self.cancel_button.setDisabled(True)
        super().cancel()

    @classmethod
    def migrate_settings(cls, settings: Dict[str, Any], version: Optional[int]):
        if version is None or version < 2:
            # before version 2 settings were indexes now they are strings
            # with language name and selected aggregator name
            if "language" in settings:
                settings["language"] = LANGUAGES[settings["language"]]
            if "aggregator" in settings:
                settings["aggregator"] = AGGREGATORS[settings["aggregator"]]
        if version is None or version < 3 and "language" in settings:
            # before version 3 language settings were language names, transform to ISO
            settings["language"] = LANG2ISO[settings["language"]]

    def send_report(self):
        if self.method == 0:
            self.report_items((
                ("Embedder", "Multilingual SBERT"),
            ))
        if self.method == 1:
            items = (
                ("Embedder", "fastText"),
                ("Language", ISO2LANG[self.language]),
                ("Aggregator", self.aggregator),
            )
            self.report_items(items)


if __name__ == "__main__":
    from orangewidget.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWDocumentEmbedding).run(Corpus.from_file("book-excerpts"))
