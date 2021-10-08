from typing import List, Optional

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QGridLayout, QLabel, QFormLayout

from orangewidget.utils.widgetpreview import WidgetPreview
from Orange.widgets import gui
from Orange.widgets.credentials import CredentialManager
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget, Msg, Output
from Orange.widgets.utils.concurrent import TaskState, ConcurrentWidgetMixin
from tweepy import TooManyRequests

from orangecontrib.text import twitter
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.language_codes import lang2code
from orangecontrib.text.twitter import TwitterAPI
from orangecontrib.text.widgets.utils import (
    ComboBox,
    ListEdit,
    CheckListLayout,
    gui_require,
)


def search(
    api: TwitterAPI,
    max_tweets: int,
    word_list: List[str],
    collecting: bool,
    language: Optional[str],
    allow_retweets: Optional[bool],
    mode: str,
    state: TaskState,
):
    def advance(progress):
        if state.is_interruption_requested():
            raise Exception
        state.set_progress_value(progress * 100)

    if mode == "content":
        return api.search_content(
            max_tweets=max_tweets,
            content=word_list,
            lang=language,
            allow_retweets=allow_retweets,
            collecting=collecting,
            callback=advance,
        )
    elif mode == "authors":
        return api.search_authors(
            max_tweets=max_tweets,
            authors=word_list,
            collecting=collecting,
            callback=advance,
        )


class OWTwitter(OWWidget, ConcurrentWidgetMixin):
    class APICredentialsDialog(OWWidget):
        name = "Twitter API Credentials"
        want_main_area = False
        resizing_enabled = False

        cm_key = CredentialManager("Twitter API Key")
        cm_secret = CredentialManager("Twitter API Secret")

        key_input = ""
        secret_input = ""

        class Error(OWWidget.Error):
            invalid_credentials = Msg("This credentials are invalid.")

        def __init__(self, parent):
            super().__init__()
            self.parent = parent
            self.credentials = None

            form = QFormLayout()
            form.setContentsMargins(5, 5, 5, 5)
            self.key_edit = gui.lineEdit(
                self, self, "key_input", controlWidth=400
            )
            form.addRow("Key:", self.key_edit)
            self.secret_edit = gui.lineEdit(
                self, self, "secret_input", controlWidth=400
            )
            form.addRow("Secret:", self.secret_edit)
            self.controlArea.layout().addLayout(form)

            self.submit_button = gui.button(
                self.controlArea, self, "OK", self.accept
            )
            self.load_credentials()

        def load_credentials(self):
            self.key_edit.setText(self.cm_key.key)
            self.secret_edit.setText(self.cm_secret.key)

        def save_credentials(self):
            self.cm_key.key = self.key_input
            self.cm_secret.key = self.secret_input

        def check_credentials(self):
            c = twitter.Credentials(self.key_input, self.secret_input)
            if self.credentials != c:
                if c.valid:
                    self.save_credentials()
                else:
                    c = None
                self.credentials = c

        def accept(self, silent=False):
            if not silent:
                self.Error.invalid_credentials.clear()
            self.check_credentials()
            if self.credentials and self.credentials.valid:
                self.parent.update_api(self.credentials)
                super().accept()
            elif not silent:
                self.Error.invalid_credentials()

    name = "Twitter"
    description = "Load tweets from the Twitter API."
    icon = "icons/Twitter.svg"
    priority = 150

    class Outputs:
        corpus = Output("Corpus", Corpus)

    want_main_area = False
    resizing_enabled = False

    class Warning(OWWidget.Warning):
        no_text_fields = Msg("Text features are inferred when none selected.")

    class Error(OWWidget.Error):
        api_error = Msg("Api error ({})")
        rate_limit = Msg("Rate limit exceeded. Please try again later.")
        empty_authors = Msg("Please provide some authors.")
        wrong_authors = Msg("Query does not match Twitter user handle.")
        key_missing = Msg("Please provide a valid API key to get the data.")

    CONTENT, AUTHOR = 0, 1
    MODES = ["Content", "Author"]
    word_list = Setting([])
    mode = Setting(0)
    limited_search = Setting(True)
    max_tweets = Setting(100)
    language = Setting(None)
    allow_retweets = Setting(False)
    collecting = Setting(False)

    attributes = [f.name for f in twitter.TwitterAPI.string_attributes]
    text_includes = Setting([f.name for f in twitter.TwitterAPI.text_features])

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        self.api = None
        self.corpus = None
        self.api_dlg = self.APICredentialsDialog(self)
        self.api_dlg.accept(silent=True)

        # Set API key button
        gui.button(
            self.controlArea,
            self,
            "Twitter API Key",
            callback=self.open_key_dialog,
            tooltip="Set the API key for this widget.",
            focusPolicy=Qt.NoFocus,
        )

        # Query
        query_box = gui.hBox(self.controlArea, "Query")
        layout = QGridLayout()
        layout.setVerticalSpacing(5)
        layout.setColumnStretch(2, 1)  # stretch last columns
        layout.setColumnMinimumWidth(1, 15)  # add some space for checkbox
        ROW = 0
        COLUMNS = 3

        def add_row(label, items):
            nonlocal ROW, COLUMNS
            layout.addWidget(QLabel(label), ROW, 0)
            if isinstance(items, tuple):
                for i, item in enumerate(items):
                    layout.addWidget(item, ROW, 1 + i)
            else:
                layout.addWidget(items, ROW, 1, 1, COLUMNS - 1)
            ROW += 1

        # Query input
        add_row(
            "Query word list:",
            ListEdit(
                self,
                "word_list",
                "Multiple lines are joined with OR.",
                80,
                self,
            ),
        )

        # Search mode
        add_row(
            "Search by:",
            gui.comboBox(
                self, self, "mode", items=self.MODES, callback=self.mode_toggle
            ),
        )

        # Language
        self.language_combo = ComboBox(
            self,
            "language",
            items=(("Any", None),) + tuple(sorted(lang2code.items())),
        )
        add_row("Language:", self.language_combo)

        # Max tweets
        add_row(
            "Max tweets:",
            gui.spin(
                self,
                self,
                "max_tweets",
                minv=1,
                maxv=10000,
                checked="limited_search",
            ),
        )

        # Retweets
        self.retweets_checkbox = gui.checkBox(
            self, self, "allow_retweets", "", minimumHeight=30
        )
        add_row("Allow retweets:", self.retweets_checkbox)

        # Collect Results
        add_row("Collect results:", gui.checkBox(self, self, "collecting", ""))

        query_box.layout().addLayout(layout)

        self.controlArea.layout().addWidget(
            CheckListLayout(
                "Text includes",
                self,
                "text_includes",
                self.attributes,
                cols=2,
                callback=self.set_text_features,
            )
        )

        # Buttons
        self.button_box = gui.hBox(self.controlArea)
        self.search_button = gui.button(
            self.button_box,
            self,
            "Search",
            self.start_stop,
            focusPolicy=Qt.NoFocus,
        )

        self.mode_toggle()
        self.setFocus()  # to widget itself to show placeholder for query_edit

    def open_key_dialog(self):
        self.api_dlg.exec_()

    def mode_toggle(self):
        if self.mode == self.AUTHOR:
            self.language_combo.setCurrentIndex(0)
            self.retweets_checkbox.setCheckState(False)
        self.retweets_checkbox.setEnabled(self.mode == self.CONTENT)
        self.language_combo.setEnabled(self.mode == self.CONTENT)

    def start_stop(self):
        if self.task:
            self.cancel()
            self.search_button.setText("Search")
        else:
            self.run_search()

    @gui_require("api", "key_missing")
    def run_search(self):
        self.Error.clear()
        self.search()

    def search(self):
        max_tweets = self.max_tweets if self.limited_search else 0

        if self.mode == self.CONTENT:
            self.start(
                search,
                self.api,
                max_tweets,
                self.word_list,
                self.collecting,
                self.language,
                self.allow_retweets,
                "content",
            )
        else:
            if not self.word_list:
                self.Error.empty_authors()
                return None
            if not any(a.startswith("@") for a in self.word_list):
                self.Error.wrong_authors()
                return None
            self.start(
                search,
                self.api,
                max_tweets,
                self.word_list,
                self.collecting,
                None,
                None,
                "authors",
            )
        self.search_button.setText("Stop")

    def update_api(self, key):
        if key:
            self.Error.key_missing.clear()
            self.api = twitter.TwitterAPI(key)
        else:
            self.api = None

    def on_done(self, result):
        self.search_button.setText("Search")
        self.corpus = result
        self.set_text_features()

    def on_exception(self, ex):
        self.search_button.setText("Search")
        if isinstance(ex, TooManyRequests):
            self.Error.rate_limit()
        else:
            self.Error.api_error(str(ex))

    def on_partial_result(self, _):
        pass

    def set_text_features(self):
        self.Warning.no_text_fields.clear()
        if not self.text_includes:
            self.Warning.no_text_fields()

        if self.corpus is not None:
            vars_ = [
                var
                for var in self.corpus.domain.metas
                if var.name in self.text_includes
            ]
            self.corpus.set_text_features(vars_ or None)
            self.Outputs.corpus.send(self.corpus)

    @gui_require("api", "key_missing")
    def send_report(self):
        for task in self.api.search_history:
            self.report_items(task)


if __name__ == "__main__":
    WidgetPreview(OWTwitter).run()
