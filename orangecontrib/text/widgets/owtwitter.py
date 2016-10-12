from PyQt4 import QtGui, QtCore

from datetime import datetime, timedelta

from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.credentials import CredentialManager
from Orange.widgets.widget import OWWidget, Msg

from orangecontrib.text import twitter
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.language_codes import lang2code
from orangecontrib.text.widgets.utils import (ComboBox, ListEdit, CheckListLayout,
                                              DateInterval, gui_require)


class IO:
    CORPUS = "Corpus"


class OWTwitter(OWWidget):
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

    MISSED_KEY = 'missed_key'

    class Error(OWWidget.Error):
        api = Msg('Api error ({})')
        rate_limit = Msg('Rate limit exceeded. Please try again later.')

    tweets_info = 'Tweets on output: {}'

    CONTENT, AUTHOR, CONTENT_AUTHOR = 0, 1, 2
    word_list = Setting([])
    mode = Setting(0)
    limited_search = Setting(True)
    max_tweets = Setting(100)
    language = Setting(None)
    allow_retweets = Setting(False)
    advance = Setting(False)
    include = Setting(False)
    includeON = Setting(True)

    error_signal = QtCore.pyqtSignal(str)
    finish_signal = QtCore.pyqtSignal()
    progress_signal = QtCore.pyqtSignal(int)
    start_signal = QtCore.pyqtSignal()

    attributes = [feat.name for feat in twitter.TwitterAPI.string_attributes]
    text_includes = Setting([feat.name for feat in twitter.TwitterAPI.text_features])

    date_interval = Setting((datetime.now().date() - timedelta(10),
                             datetime.now().date()))

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
                                       tooltip="Set the API key for this widget.")
        key_dialog_button.setFocusPolicy(QtCore.Qt.NoFocus)

        # Query
        self.query_box = gui.hBox(self.controlArea, 'Query')

        # Queries configuration
        layout = QtGui.QGridLayout()
        layout.setSpacing(7)

        # Query
        row = 0
        query_edit = ListEdit(self, 'word_list', "Multiple lines are automatically joined with OR.", self)
        query_edit.setFixedHeight(self.text_area_height)
        layout.addWidget(QtGui.QLabel('Query word list:'), row, 0, 1, self.label_width)
        layout.addWidget(query_edit, row, self.label_width, 1, self.widgets_width)

        # Search mode
        row += 1
        mode = gui.comboBox(self, self, 'mode')
        mode.addItem('Content')
        mode.addItem('Author')
        mode.addItem('Content & Author')
        layout.addWidget(QtGui.QLabel('Search by:'), row, 0, 1, self.label_width)
        layout.addWidget(mode, row, self.label_width, 1, self.widgets_width)

        # Retweets
        row += 1
        check = gui.checkBox(self, self, 'allow_retweets', '')
        layout.addWidget(QtGui.QLabel('Allow\nretweets:'), row, 0, 1, self.label_width)
        layout.addWidget(check, row, self.label_width, 1, 1)

        # Time interval
        row += 1
        interval = DateInterval(self, 'date_interval',
                                min_date=datetime.now().date() - timedelta(10),
                                max_date=datetime.now().date(),
                                from_label='since', to_label='until')
        layout.addWidget(QtGui.QLabel('Date:'), row, 0, 1, self.label_width)
        layout.addWidget(interval, row, self.label_width, 1, self.widgets_width)

        # Language
        row += 1
        language_edit = ComboBox(self, 'language',
                                 (('Any', None),) + tuple(sorted(lang2code.items())))
        layout.addWidget(QtGui.QLabel('Language:'), row, 0, 1, self.label_width)
        layout.addWidget(language_edit, row, self.label_width, 1, self.widgets_width)

        # Max tweets
        row += 1
        check, spin = gui.spin(self, self, 'max_tweets', minv=1, maxv=10000,
                               checked='limited_search')
        layout.addWidget(QtGui.QLabel('Max tweets:'), row, 0, 1, self.label_width)
        layout.addWidget(check, row, self.label_width, 1, 1)
        layout.addWidget(spin, row, self.label_width + 1, 1, self.widgets_width - 1)

        # Checkbox
        row += 1
        check = gui.checkBox(self, self, 'advance', '')
        layout.addWidget(QtGui.QLabel('Accumulate\nresults:'), row, 0, 1, self.label_width)
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

        self.search_button = gui.button(self.button_box, self, "Search", self.search)
        self.search_button.setFocusPolicy(QtCore.Qt.NoFocus)

        self.start_signal.connect(self.on_start)
        self.error_signal.connect(self.on_error)
        self.progress_signal.connect(self.on_progress)
        self.finish_signal.connect(self.on_finish)

        self._tweet_count = False
        self.send_corpus()
        self.setFocus()  # set focus to widget itself so placeholder is shown for query_edit

    def open_key_dialog(self):
        self.api_dlg.exec_()

    def update_api(self, key):
        if key:
            self.Warning.clear()
            self.api = twitter.TwitterAPI(key,
                                          on_start=self.start_signal.emit,
                                          on_progress=self.progress_signal.emit,
                                          on_error=self.error_signal.emit,
                                          on_rate_limit=self.on_rate_limit,
                                          on_finish=self.finish_signal.emit)
        else:
            self.api = None

    @QtCore.pyqtSlot()
    @gui_require('api', MISSED_KEY)
    def search(self):
        if not self.api.running:
            if not self.advance:
                self.api.reset()
            self.search_button.setText("Stop")
            word_list = self.word_list if self.mode in [self.CONTENT, self.CONTENT_AUTHOR] else None
            authors = self.word_list if self.mode in [self.AUTHOR, self.CONTENT_AUTHOR] else None

            self.api.search(max_tweets=self.max_tweets if self.limited_search else 0,
                            word_list=word_list, authors=authors, lang=self.language,
                            since=self.date_interval[0], until=self.date_interval[1],
                            allow_retweets=self.allow_retweets)
        else:
            self.api.disconnect()
            self.search_button.setText("Search")

    def update_tweets_info(self):
        tweet_count = len(self.api.container) if self.api else 0
        self.tweets_count_label.setText(self.tweets_info.format(tweet_count))

    @QtCore.pyqtSlot()
    def on_start(self):
        self.Error.clear()

        if self.limited_search:
            self._tweet_count = self.max_tweets
            self.progressBarInit()

    @QtCore.pyqtSlot(int)
    def on_progress(self, progress):
        self.update_tweets_info()
        if self._tweet_count:
            self.progressBarSet(progress / self._tweet_count * 100)

    @QtCore.pyqtSlot()
    def on_finish(self):
        self.send_corpus()
        self.update_tweets_info()
        self.search_button.setText('Search')
        self.search_button.setEnabled(True)
        if self._tweet_count:
            self._tweet_count = False
            self.progressBarFinished()

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

    @QtCore.pyqtSlot(str)
    def on_rate_limit(self):
        self.Error.rate_limit()

    @QtCore.pyqtSlot(str)
    def on_error(self, text):
        self.Error.api(text)

    def reset(self):
        self.api.reset()
        self.update_tweets_info()
        self.send_corpus()

    @gui_require('api', MISSED_KEY)
    def send_report(self):
        for task in self.api.history:
            self.report_items(task.report())


if __name__ == '__main__':
    app = QtGui.QApplication([])
    widget = OWTwitter()
    widget.show()
    app.exec()
    widget.saveSettings()
