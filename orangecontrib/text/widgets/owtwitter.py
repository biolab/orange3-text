from PyQt4 import QtGui, QtCore

from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.credentials import CredentialManager
from Orange.widgets.widget import OWWidget, Msg

from orangecontrib.text import twitter
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.language_codes import lang2code
from orangecontrib.text.widgets.utils import (ComboBox, ListEdit,
                                              CheckListLayout, gui_require,
                                              OWConcurrentWidget, asynchronous)


class IO:
    CORPUS = "Corpus"


class OWTwitter(OWConcurrentWidget):
    class APICredentialsDialog(OWWidget):
        name = "Twitter API Credentials"
        want_main_area = False
        resizing_enabled = False

        cm_key = CredentialManager('Twitter API Key')
        cm_secret = CredentialManager('Twitter API Secret')

        key_input = ''
        secret_input = ''

        class Error(OWWidget.Error):
            invalid_credentials = Msg('This credentials are invalid.')

        def __init__(self, parent):
            super().__init__()
            self.parent = parent
            self.credentials = None

            form = QtGui.QFormLayout()
            form.setMargin(5)
            self.key_edit = gui.lineEdit(self, self, 'key_input', controlWidth=400)
            form.addRow('Key:', self.key_edit)
            self.secret_edit = gui.lineEdit(self, self, 'secret_input', controlWidth=400)
            form.addRow('Secret:', self.secret_edit)
            self.controlArea.layout().addLayout(form)

            self.submit_button = gui.button(self.controlArea, self, "OK", self.accept)

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
            if not silent: self.Error.invalid_credentials.clear()
            self.check_credentials()
            if self.credentials and self.credentials.valid:
                self.parent.update_api(self.credentials)
                super().accept()
            elif not silent:
                self.Error.invalid_credentials()

    name = "Twitter"
    description = "Load tweets from the Twitter API."
    icon = "icons/Twitter.svg"
    priority = 25

    outputs = [(IO.CORPUS, Corpus)]
    want_main_area = False
    resizing_enabled = False
    widgets_width = 11
    label_width = 2
    text_area_height = 80

    class Warning(OWWidget.Warning):
        missed_key = Msg('Please provide a valid API key in order to get the data.')
        no_text_fields = Msg('Text features are inferred when none are selected.')

    class Error(OWWidget.Error):
        api = Msg('Api error ({})')
        rate_limit = Msg('Rate limit exceeded. Please try again later.')

    tweets_info = 'Tweets on output: {}'

    CONTENT, AUTHOR = 0, 1
    word_list = Setting([])
    mode = Setting(0)
    limited_search = Setting(True)
    max_tweets = Setting(100)
    language = Setting(None)
    allow_retweets = Setting(False)
    collecting = Setting(False)

    attributes = [feat.name for feat in twitter.TwitterAPI.string_attributes]
    text_includes = Setting([feat.name for feat in twitter.TwitterAPI.text_features])

    def __init__(self, *args, **kwargs):
        """
        Attributes:
            api (twitter.TwitterAPI): Twitter API object.
        """
        super().__init__(*args, **kwargs)
        self.api = None
        self.corpus = None
        self.api_dlg = self.APICredentialsDialog(self)
        self.api_dlg.accept(silent=True)

        self.controlArea.layout().setContentsMargins(0, 15, 0, 0)

        # Set API key button.
        key_dialog_button = gui.button(self.controlArea, self, 'Twitter API Key',
                                       callback=self.open_key_dialog,
                                       tooltip='Set the API key for this widget.')
        key_dialog_button.setFocusPolicy(QtCore.Qt.NoFocus)

        # Query
        self.query_box = gui.hBox(self.controlArea, 'Query')

        # Queries configuration
        layout = QtGui.QGridLayout()
        layout.setSpacing(7)

        # Query
        row = 0
        query_edit = ListEdit(self, 'word_list', "Multiple lines are joined with OR.", self)
        query_edit.setFixedHeight(self.text_area_height)
        layout.addWidget(QtGui.QLabel('Query word list:'), row, 0, 1, self.label_width)
        layout.addWidget(query_edit, row, self.label_width, 1, self.widgets_width)

        # Search mode
        row += 1
        mode = gui.comboBox(self, self, 'mode')
        mode.addItem('Content')
        mode.addItem('Author')
        layout.addWidget(QtGui.QLabel('Search by:'), row, 0, 1, self.label_width)
        layout.addWidget(mode, row, self.label_width, 1, self.widgets_width)

        # Language
        row += 1
        language_edit = ComboBox(self, 'language',
                                 (('Any', None),) + tuple(
                                     sorted(lang2code.items())))
        layout.addWidget(QtGui.QLabel('Language:'), row, 0, 1,
                         self.label_width)
        layout.addWidget(language_edit, row, self.label_width, 1,
                         self.widgets_width)

        # Max tweets
        row += 1
        check, spin = gui.spin(self, self, 'max_tweets', minv=1, maxv=10000,
                               checked='limited_search')
        layout.addWidget(QtGui.QLabel('Max tweets:'), row, 0, 1,
                         self.label_width)
        layout.addWidget(check, row, self.label_width, 1, 1)
        layout.addWidget(spin, row, self.label_width + 1, 1,
                         self.widgets_width - 1)

        # Retweets
        row += 1
        check = gui.checkBox(self, self, 'allow_retweets', '')
        layout.addWidget(QtGui.QLabel('Allow retweets:'), row, 0, 1, self.label_width)
        layout.addWidget(check, row, self.label_width, 1, 1)

        # Checkbox
        row += 1
        check = gui.checkBox(self, self, 'collecting', '')
        layout.addWidget(QtGui.QLabel('Collect results:'), row, 0, 1, self.label_width)
        layout.addWidget(check, row, self.label_width, 1, 1)
        self.query_box.layout().addLayout(layout)

        self.controlArea.layout().addWidget(
            CheckListLayout('Text includes', self, 'text_includes', self.attributes, cols=2,
                            callback=self.set_text_features))

        self.info_box = gui.hBox(self.controlArea, 'Info')
        self.tweets_count_label = gui.label(self.info_box, self, self.tweets_info.format(0))

        # Buttons
        self.button_box = gui.hBox(self.controlArea)
        self.button_box.layout().addWidget(self.report_button)

        self.search_button = gui.button(self.button_box, self, 'Search', self.start_stop)
        self.search_button.setFocusPolicy(QtCore.Qt.NoFocus)

        self._tweet_count = False
        self.send_corpus()
        self.setFocus()  # set focus to widget itself so placeholder is shown for query_edit

    def open_key_dialog(self):
        self.api_dlg.exec_()

    def start_stop(self):
        if self.running:
            self.stop()
        else:
            self.search()

    def update_api(self, key):
        if key:
            self.Warning.clear()
            self.api = twitter.TwitterAPI(key)
        else:
            self.api = None

    @gui_require('api', 'missed_key')
    @asynchronous(allow_partial_results=True)
    def search(self, on_progress, should_break):
            def progress_with_info(total, current):
                on_progress(100 * current / self.max_tweets)
                self.update_tweets_num(total)

            word_list = self.word_list if self.mode == self.CONTENT else None
            authors = self.word_list if self.mode == self.AUTHOR else None

            return self.api.search(max_tweets=self.max_tweets if self.limited_search else 0,
                                   content=word_list, authors=authors, lang=self.language,
                                   allow_retweets=self.allow_retweets,
                                   collecting=self.collecting,
                                   on_progress=progress_with_info,
                                   should_break=should_break)

    def on_start(self):
        self.Error.clear()
        self.search_button.setText('Stop')
        self.send(IO.CORPUS, None)

    def on_result(self, result):
        self.search_button.setText('Search')
        self.corpus = result
        self.set_text_features()

    def update_tweets_num(self, num=0):
        self.tweets_count_label.setText(self.tweets_info.format(num))

    def send_corpus(self):
        if self.api and self.api.tweets:
            self.corpus = self.api.create_corpus()
            self.set_text_features()
        else:
            self.send(IO.CORPUS, None)

    def set_text_features(self):
        self.Warning.no_text_fields.clear()
        if not self.text_includes:
            self.Warning.no_text_fields()

        if self.corpus is not None:
            vars_ = [var for var in self.corpus.domain.metas if var.name in self.text_includes]
            self.corpus.set_text_features(vars_ or None)
            self.send(IO.CORPUS, self.corpus)

    @gui_require('api', 'missed_key')
    def send_report(self):
        for task in self.api.search_history:
            self.report_items(task)


if __name__ == '__main__':
    app = QtGui.QApplication([])
    widget = OWTwitter()
    widget.show()
    app.exec()
    widget.saveSettings()
