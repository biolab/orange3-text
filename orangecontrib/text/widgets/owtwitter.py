from typing import List, Optional

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QGridLayout, QLabel, QPlainTextEdit

from orangewidget.utils.widgetpreview import WidgetPreview
from Orange.widgets import gui
from Orange.widgets.credentials import CredentialManager
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget, Msg, Output
from Orange.widgets.utils.concurrent import TaskState, ConcurrentWidgetMixin

from orangecontrib.text import twitter
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.language_codes import code2lang
from orangecontrib.text.twitter import TwitterAPI, SUPPORTED_LANGUAGES, NoAuthorError
from orangecontrib.text.widgets.utils import ComboBox, ListEdit, gui_require


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
    else:  # mode == "authors":
        return api.search_authors(
            max_tweets=max_tweets,
            authors=[word.strip("@") for word in word_list],
            collecting=collecting,
            callback=advance,
        )


class OWTwitter(OWWidget, ConcurrentWidgetMixin):
    name = "Twitter"
    description = "Load tweets from the Twitter API."
    icon = "icons/Twitter.svg"
    keywords = ["twitter", "tweet"]
    priority = 150

    class Outputs:
        corpus = Output("Corpus", Corpus)

    want_main_area = False
    resizing_enabled = False

    class Info(OWWidget.Information):
        nut_enough_tweets = Msg(
            "Downloaded fewer tweets than requested, since not enough tweets or rate limit reached"
        )

    class Error(OWWidget.Error):
        api_error = Msg("Api error: {}")
        empty_query = Msg("Please provide {}.")
        key_missing = Msg("Please provide a valid API token.")
        wrong_author = Msg("Author '{}' does not exist.")

    CONTENT, AUTHOR = 0, 1
    MODES = ["Content", "Author"]
    word_list: List = Setting([])
    mode: int = Setting(0)
    limited_search: bool = Setting(True)
    max_tweets: int = Setting(100)
    language: Optional[str] = Setting(None)
    allow_retweets: bool = Setting(False)
    collecting: bool = Setting(False)

    class APICredentialsDialog(OWWidget):
        name = "Twitter API Credentials"
        want_main_area = False
        resizing_enabled = False

        def __init__(self, parent):
            super().__init__()
            self.cm_key = CredentialManager("Twitter Bearer Token")
            self.parent = parent

            box = gui.vBox(self.controlArea, "Bearer Token")
            self.key_edit = QPlainTextEdit()
            box.layout().addWidget(self.key_edit)

            self.submit_button = gui.button(self.buttonsArea, self, "OK", self.accept)
            self.load_credentials()

        def load_credentials(self):
            self.key_edit.setPlainText(self.cm_key.key)

        def save_credentials(self):
            self.cm_key.key = self.key_edit.toPlainText()

        def accept(self):
            token = self.key_edit.toPlainText()
            if token:
                self.save_credentials()
                self.parent.update_api(token)
                super().accept()

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        self.api = None
        self.api_dlg = self.APICredentialsDialog(self)
        self.api_dlg.accept()

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
        langs = (("Any", None),) + tuple((code2lang[l], l) for l in SUPPORTED_LANGUAGES)
        self.language_combo = ComboBox(self, "language", items=langs)
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

        # Buttons
        self.search_button = gui.button(
            self.buttonsArea,
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
        self.Info.nut_enough_tweets.clear()
        self.search()

    def search(self):
        max_tweets = self.max_tweets if self.limited_search else None
        content = self.mode == self.CONTENT
        if not self.word_list:
            self.Error.empty_query("keywords" if content else "authors")
            self.Outputs.corpus.send(None)
            return

        self.start(
            search,
            self.api,
            max_tweets,
            self.word_list,
            self.collecting,
            self.language if content else None,
            self.allow_retweets if content else None,
            "content" if content else "authors",
        )
        self.search_button.setText("Stop")

    def update_api(self, key):
        if key:
            self.Error.key_missing.clear()
            self.api = twitter.TwitterAPI(key)
        else:
            self.api = None

    def on_done(self, result_corpus):
        self.search_button.setText("Search")
        if (
            result_corpus is None  # probably because of rate error at beginning
            # or fewer tweets than expected
            or self.mode == self.CONTENT
            and len(result_corpus) < self.max_tweets
            or self.mode == self.AUTHOR
            # for authors, we expect self.max_tweets for each author
            and len(result_corpus) < self.max_tweets * len(self.word_list)
        ):
            self.Info.nut_enough_tweets()
        self.Outputs.corpus.send(result_corpus)

    def on_exception(self, ex):
        self.search_button.setText("Search")
        if isinstance(ex, NoAuthorError):
            self.Error.wrong_author(str(ex))
        else:
            self.Error.api_error(str(ex))

    def on_partial_result(self, _):
        pass

    @gui_require("api", "key_missing")
    def send_report(self):
        for task in self.api.search_history:
            self.report_items(task)


if __name__ == "__main__":
    WidgetPreview(OWTwitter).run()
