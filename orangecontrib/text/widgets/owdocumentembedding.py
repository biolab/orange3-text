import json
import enum
import os
from typing import Dict, Optional, Any

import openai

from AnyQt.QtCore import Qt, QSettings
from AnyQt.QtWidgets import QVBoxLayout, QPushButton, QStyle

from Orange.misc.utils.embedder_utils import EmbeddingConnectionError
from Orange.widgets import gui, settings
from Orange.widgets.settings import Setting
from Orange.widgets.utils import qname
from Orange.widgets.utils.settings import QSettings_writeArray, QSettings_readArray
from Orange.widgets.widget import Msg, Output, OWWidget
from orangecanvas.utils import findf

from orangecontrib.text.corpus import Corpus
from orangecontrib.text.language import (
    ISO2LANG, DEFAULT_LANGUAGE, LanguageModel, LANG2ISO
)
from orangecontrib.text.vectorization.document_embedder import (
    AGGREGATORS,
    AGGREGATORS_ITEMS,
    DocumentEmbedder,
    LANGUAGES, OAIDocumentEmbedder,
)
from orangecontrib.text.vectorization.sbert import SBERT
from orangecontrib.text.widgets.utils.owbasevectorizer import (
    OWBaseVectorizer,
    Vectorizer,
)
from orangecontrib.text.widgets.utils.llmmodelwidget import LLMModelWidget


class EmbeddingVectorizer(Vectorizer):
    skipped_documents = None

    def _transform(self, callback):
        embeddings, skipped = self.method.transform(self.corpus, callback=callback)
        self.new_corpus = embeddings
        self.skipped_documents = skipped


class Methods(enum.Enum):
    SBERT = 0
    FastText = 1
    OpenAIEmbedder = 2


Providers = {
    "ollama": "http://localhost:11434/v1",
    "llama.cpp": "http://localhost:9931/v1",
}


def is_localhost(url: str) -> bool:
    """
    Check if the given URL points to localhost.

    Args:
        url (str): The URL to check

    Returns:
        bool: True if the URL points to localhost, False otherwise
    """
    from urllib.parse import urlparse
    parsed_url = urlparse(url)
    localhost_hosts = {'localhost', '127.0.0.1', '::1'}
    return parsed_url.hostname in localhost_hosts

ServiceName = "Orange: API key"

class OWDocumentEmbedding(OWBaseVectorizer):
    name = "Document Embedding"
    description = "Document embedding using pretrained models."
    keywords = "embedding, document embedding, fasttext, bert, sbert"
    icon = "icons/TextEmbedding.svg"
    priority = 300

    buttons_area_orientation = Qt.Vertical
    settings_version = 3

    Methods = [SBERT, DocumentEmbedder, OAIDocumentEmbedder]

    class Outputs(OWBaseVectorizer.Outputs):
        skipped = Output("Skipped documents", Corpus)

    class Error(OWWidget.Error):
        no_connection = Msg(
            "No internet connection. Please establish a connection or use "
            "another vectorizer."
        )
        unexpected_error = Msg("Embedding error: {}")
        authentication_error = Msg("Authentication error: {}")
        api_spec_error = Msg("API call error: {}")
        connection_error = Msg("Connection error. Check that the service is running and is accessible. {}")

    class Warning(OWWidget.Warning):
        unsuccessful_embeddings = Msg("Some embeddings were unsuccessful.")

    method: int = Setting(default=0)
    language: str = Setting(default=DEFAULT_LANGUAGE, schema_only=True)
    aggregator: str = Setting(default="Mean")

    base_url = Setting("", schema_only=True)
    api_key = ""
    model = Setting("", schema_only=True)

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

    DefaultModels = [
        {"base_url": "ollama", "model": "snowflake-arctic-embed:22m"},
        {"base_url": "ollama", "model": "granite-embedding:30m"},
        {"base_url": "ollama", "model": "all-minilm"},
        {"base_url": "https://api.openai.com/v1", "model": ""}
    ]

    def create_configuration_layout(self):
        layout = QVBoxLayout()
        rbtns = gui.radioButtons(None, self, "method", callback=self.on_change)
        layout.addWidget(rbtns)

        gui.appendRadioButton(rbtns, "Multilingual SBERT")
        gui.appendRadioButton(rbtns, "fastText:")
        self.fast_text_controls = ibox = gui.indentedBox(rbtns)
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
        gui.appendRadioButton(rbtns, "Other (OpenAI API compatible):")
        self.oai_controls = ibox = gui.indentedBox(rbtns)
        self.llmapiwidget = LLMModelWidget(keyringServiceName=ServiceName)
        ibox.layout().addWidget(self.llmapiwidget)

        items = self._load_history()
        items = items + self.DefaultModels
        self.llmapiwidget.setHistory(items)
        if self.base_url and self.model:
            self.llmapiwidget.setBaseUrl(self.base_url)
            self.llmapiwidget.setModelId(self.model)
        self.api_key = self.llmapiwidget.apiKey()
        self.llmapiwidget.changed.connect(self.on_api_param_change)
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
        method = Methods(self.method)
        self.fast_text_controls.setEnabled(method == Methods.FastText)
        self.oai_controls.setEnabled(method == Methods.OpenAIEmbedder)
        self.vectorizer = EmbeddingVectorizer(self.init_method(), self.corpus)

    def on_change(self):
        if Methods(self.method) != Methods.OpenAIEmbedder:
            self.Error.api_spec_error.clear()
            self.Error.authentication_error.clear()
            self.Error.connection_error.clear()
        super().on_change()

    def on_api_param_change(self):
        self.base_url = self.llmapiwidget.baseUrl()
        self.model = self.llmapiwidget.modelId()
        self.api_key = self.llmapiwidget.apiKey()
        if self.base_url and self.model:
            self._save_history_item(
                {"base_url": self.base_url, "model": self.model,
                 "has_key": bool(self.api_key)}
            )
        self.on_change()

    @classmethod
    def _local_settings(cls) -> QSettings:
        """Return a QSettings instance with local persistent QSettings for `cls`."""
        filename = "{}.ini".format(qname(cls))
        fname = os.path.join(settings.widget_settings_dir(versioned=False), filename)
        return QSettings(fname, QSettings.IniFormat)

    def _save_history_item(self, item: LLMModelWidget.Item):
        settings = self._local_settings()
        items = self._load_history()
        # find/replace item in stored history
        existing = findf(items, lambda it: it["base_url"] == item["base_url"] and it["model"] == item["model"])
        if existing:
            items.remove(existing)
        items.insert(0, item)
        QSettings_writeArray(settings, "endpoints", items)

    def _load_history(self) -> list[LLMModelWidget.Item]:
        settings = self._local_settings()
        items = QSettings_readArray(settings, "endpoints", {
            "base_url": str, "model": str, "has_key": bool
        })
        items = [item for item in items if item["base_url"].strip() and item["model"].strip()]
        return items

    def set_base_url(self, url):
        self.llmapiwidget.setBaseUrl(url)

    def set_api_key(self, key):
        self.llmapiwidget.setApiKey(key)

    def set_model(self, model):
        self.llmapiwidget.setModelId(model)

    def init_method(self):
        method = Methods(self.method)
        match method:
            case Methods.SBERT:
                kwargs = {}
            case Methods.FastText:
                kwargs = dict(language=self.language, aggregator=self.aggregator)
            case Methods.OpenAIEmbedder:
                base_url = self.llmapiwidget.baseUrl()
                api_key = self.llmapiwidget.apiKey()
                model = self.llmapiwidget.modelId()
                base_url = Providers.get(base_url, base_url)
                if is_localhost(base_url) and not api_key.strip():
                    # local providers probably do not need a key but
                    # `openai.Client` still complains about it.
                    api_key = "sk-no-key-required"
                kwargs = dict(base_url=base_url, api_key=api_key, model=model)
            case _:
                raise NameError
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
        def oaie_message(ex: openai.APIStatusError) -> str:
            """Extract message from openai error"""
            try:
                return json.loads(ex.response.content)["error"]["message"]
            except (json.JSONDecodeError, KeyError, AttributeError):
                return ex.message
        self.cancel_button.setDisabled(True)
        if isinstance(ex, EmbeddingConnectionError):
            self.Error.no_connection()
        elif isinstance(ex, openai.AuthenticationError):
            self.Error.authentication_error(oaie_message(ex))
        elif isinstance(ex, openai.BadRequestError):
            self.Error.api_spec_error(oaie_message(ex))
        elif isinstance(ex, openai.APIStatusError):
            self.Error.api_spec_error(oaie_message(ex))
        elif isinstance(ex, openai.APIConnectionError):
            ex = ex.__cause__ if ex.__cause__ is not None else ex
            self.Error.connection_error(str(ex))
        else:
            self.Error.unexpected_error(str(ex), exc_info=ex)
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
        match Methods(self.method):
            case Methods.SBERT:
                self.report_items((
                    ("Embedder", "Multilingual SBERT"),
                ))
            case Methods.FastText:
                self.report_items((
                    ("Embedder", "fastText"),
                    ("Language", ISO2LANG[self.language]),
                    ("Aggregator", self.aggregator),
                ))
            case Methods.OpenAIEmbedder:
                self.report_items ((
                    ("Base Api", self.base_url),
                    ("Model", self.model),
                ))


if __name__ == "__main__":
    from orangewidget.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWDocumentEmbedding).run(Corpus.from_file("book-excerpts"))
