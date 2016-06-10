import functools
from itertools import chain

from PyQt4 import QtGui, QtCore

from PyQt4.QtGui import QComboBox, QDialog
from datetime import date, datetime, timedelta

from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget

from orangecontrib.text import twitter
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.language_codes import lang2code
from orangecontrib.text.widgets.utils import (ComboBox, ListEdit, CheckListLayout,
                                              DateInterval)


class APICredentialsDialog(QDialog):
    size_hint = (470, 100)
    widgets_min_width = 400

    def __init__(self, parent, windowTitle="Twitter API Credentials"):
        super().__init__(parent, windowTitle=windowTitle)
        self.parent = parent
        self.key = self.recent_keys[0] if self.recent_keys else None

        self.setLayout(QtGui.QVBoxLayout())
        self.layout().setMargin(5)
        self.mainArea = gui.widgetBox(self)

        self.layout().addLayout(self.setup_main_layout())

        self.button_box = gui.hBox(self)
        self.submit_button = gui.button(self.button_box, self, "OK", self.accept)
        self.button_box.layout().setAlignment(QtCore.Qt.AlignCenter)
        self.submit_button.setFixedWidth(150)

        self.update_gui()

    def setup_main_layout(self):
        layout = QtGui.QFormLayout()

        # Key combo box.
        self.key_combo = QComboBox(self)
        self.key_combo.setEditable(True)
        self.key_combo.activated[int].connect(self.key_selected)
        self.key_combo.setMinimumWidth(self.widgets_min_width)
        layout.addRow('Key:', self.key_combo)

        self.secret_line_edit = QtGui.QLineEdit(self)
        self.secret_line_edit.setMinimumWidth(self.widgets_min_width)
        layout.addRow('Secret:', self.secret_line_edit)
        self.update_gui()
        return layout

    def update_key(self):
        key = twitter.Credentials(self.key_combo.currentText().strip(),
                                  self.secret_line_edit.text().strip())
        if self.key != key:
            self.key = key

    @property
    def recent_keys(self):
        return self.parent.recent_api_keys

    def update_gui(self):
        self.key_combo.clear()

        for key in self.recent_keys:
            self.key_combo.addItem(key.consumer_key)

        if self.key:
            self.key_combo.setEditText(self.key.consumer_key)
            self.secret_line_edit.setText(self.key.consumer_secret)

    def check_credentials(self):
        self.update_key()

        if self.key.valid:
            self.save_key()

    def save_key(self):
        if self.key in self.recent_keys:
            self.recent_keys.remove(self.key)

        self.recent_keys.insert(0, self.key)

    def key_selected(self, n):
        self.key = self.recent_keys[n]
        self.recent_keys.pop(n)
        self.recent_keys.insert(0, self.key)
        self.update_gui()

    def accept(self):
        self.check_credentials()
        self.parent.update_api(self.key)
        if self.key.valid:
            super().accept()
        else:
            # TODO: add notification
            tip = 'This credentials are invalid.'
            QtGui.QToolTip.showText(QtGui.QCursor.pos(), tip)

    def sizeHint(self):
        return QtCore.QSize(*self.size_hint)


class Output:
    CORPUS = "Corpus"


def require(attribute, warning_code, warning_message):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if not getattr(self, attribute, None):
                self.warning(warning_code, warning_message)
            else:
                self.warning(warning_code)
                return func(self, *args, **kwargs)
        return wrapper
    return decorator


class OWTwitter(OWWidget):
    """
    Attributes:
        key (twitter.Credentials): Twitter API key/secret holder
        api (twitter.TwitterAPI): Twitter API object.
    """
    name = "Twitter"
    description = "Load tweets from the Twitter API."
    icon = "icons/Twitter.svg"
    priority = 25

    outputs = [(Output.CORPUS, Corpus)]
    want_main_area = False
    resizing_enabled = False
    widgets_width = 11
    label_width = 2
    text_area_height = 80

    MISSED_KEY = 1
    MISSED_KEY_MESSAGE = 'Please provide a valid API key in order to get the data.'
    MISSED_QUERY = 2
    MISSED_QUERY_MESSAGE = 'Please input query words.'
    API_ERROR = 3

    tweets_info = 'Tweets on output: {}'

    CONTENT, AUTHOR, CONTENT_AUTHOR = 0, 1, 2
    recent_api_keys = Setting([])
    word_list = Setting('')
    mode = Setting(0)
    limited_search = Setting(True)
    max_tweets = Setting(100)
    language = Setting(None)
    advance = Setting(False)
    include = Setting(False)
    includeON = Setting(True)

    error_signal = QtCore.pyqtSignal(str)
    finish_signal = QtCore.pyqtSignal()
    progress_signal = QtCore.pyqtSignal(int)
    start_signal = QtCore.pyqtSignal()

    attributes = [
        ('Author', ('author_screen_name', )),
        ('Content', ('text',)),
        ('Date', ('created_at',)),
        ('Language', ('lang',)),
        ('In Reply Ro', 'in_reply_to_user_id'),
        ('Number of Likes', ('favorite_count',)),
        ('Number of Retweets', ('retweet_count',)),
        ('Coordinates', ('coordinates_longitude', 'coordinates_latitude', 'place')),
        ('Tweet ID', ('id',)),
        ('Author ID', ('author_id',)),
        ('Author Name', ('author_name', )),
        ('Author Description', ('author_description',)),
        ('Author Statistic', ('author_statuses_count', 'author_favourites_count',
                              'author_friends_count', 'author_followers_count',
                              'author_listed_count')),
        ('Author Verified', ('author_verified',)),
    ]
    corpus_variables = Setting([val for _, val in attributes[:3]])

    date_interval = Setting((datetime.now().date() - timedelta(365),
                             datetime.now().date()))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.key = self.recent_api_keys[0] if self.recent_api_keys else None
        self.api = None

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
        query_edit = ListEdit(self, 'word_list', self)
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

        # Time interval
        row += 1
        interval = DateInterval(self, 'date_interval',
                                min_date=date(2006, 3, 21), max_date=datetime.now().date(),
                                from_label='since', to_label='until')
        layout.addWidget(QtGui.QLabel('Date:'), row, 0, 1, self.label_width)
        layout.addWidget(interval, row, self.label_width, 1, self.widgets_width)

        # Language
        row += 1
        language_edit = ComboBox(self, 'language', (('Any', None),) + tuple(lang2code.items()))
        layout.addWidget(QtGui.QLabel('Language:'), row, 0, 1, self.label_width)
        layout.addWidget(language_edit, row, self.label_width, 1, self.widgets_width)

        # Max tweets
        row += 1
        check, spin = gui.spin(self, self, 'max_tweets', minv=1, maxv=10000,
                               checked='limited_search')
        layout.addWidget(QtGui.QLabel('Max tweets:'), row, 0, 1, self.label_width)
        layout.addWidget(check, row, self.label_width, 1, 1)
        layout.addWidget(spin, row, self.label_width+1, 1, self.widgets_width-1)

        # Checkbox
        row += 1
        check = gui.checkBox(self, self, 'advance', '')
        layout.addWidget(QtGui.QLabel('Accumulate\nresults:'), row, 0, 1, self.label_width)
        layout.addWidget(check, row, self.label_width, 1, 1)
        self.query_box.layout().addLayout(layout)

        self.controlArea.layout().addWidget(
            CheckListLayout('Text includes', self, 'corpus_variables', self.attributes, cols=2))

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

        self.update_api(self.key)
        self._tweet_count = False
        self.send_corpus()

    def open_key_dialog(self):
        api_dlg = APICredentialsDialog(self)
        api_dlg.exec_()

    def update_api(self, key):
        self.key = key
        self.warning(self.MISSED_KEY)
        if self.key:
            self.api = twitter.TwitterAPI(self.key,
                                          on_start=self.start_signal.emit,
                                          on_progress=self.progress_signal.emit,
                                          on_error=self.error_signal.emit,
                                          on_finish=self.finish_signal.emit)
        else:
            self.api = None

    @QtCore.pyqtSlot()
    @require('api', MISSED_KEY, MISSED_KEY_MESSAGE)
    def search(self):
        if not self.api.running:
            if not self.advance:
                self.api.reset()
            self.search_button.setText("Stop")
            word_list = self.word_list if self.mode in [self.CONTENT, self.CONTENT_AUTHOR] else []
            authors = self.word_list if self.mode in [self.AUTHOR, self.CONTENT_AUTHOR] else []

            self.api.search(max_tweets=self.max_tweets if self.limited_search else None,
                            word_list=word_list, authors=authors, lang=self.language,
                            since=self.date_interval[0], until=self.date_interval[1])
        else:
            self.api.disconnect()
            self.search_button.setText("Search")

    @QtCore.pyqtSlot()
    def on_start(self):
        self.error(self.API_ERROR)

        if self.limited_search:
            self._tweet_count = self.max_tweets
            self.progressBarInit()

    @QtCore.pyqtSlot(int)
    def on_progress(self, progress):
        tweet_count = len(self.api.container) if self.api else 0
        self.tweets_count_label.setText(self.tweets_info.format(tweet_count))
        if self._tweet_count:
            self.progressBarSet(progress / self._tweet_count * 100)

    @QtCore.pyqtSlot()
    def on_finish(self):
        self.send_corpus()
        self.search_button.setText('Search')
        self.search_button.setEnabled(True)
        if self._tweet_count:
            self._tweet_count = False
            self.progressBarFinished()

    def send_corpus(self):
        if self.api and self.api.tweets:
            corpus = self.api.create_corpus(
                included_attributes=list(chain(*self.corpus_variables)))
            self.send(Output.CORPUS, corpus)
        else:
            self.send(Output.CORPUS, None)

    @QtCore.pyqtSlot(str)
    def on_error(self, text):
        self.error(self.API_ERROR, text)

    def reset(self):
        self.api.reset()
        self.send_corpus()

    def send_report(self):
        for task in self.api.history:
            self.report_items(task.report())


if __name__ == '__main__':
    app = QtGui.QApplication([])
    widget = OWTwitter()
    widget.show()
    app.exec()
    widget.saveSettings()
