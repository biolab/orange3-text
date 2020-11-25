from datetime import datetime, timedelta, date

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QApplication, QFormLayout

from Orange.data import StringVariable
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget, Msg
from Orange.widgets.credentials import CredentialManager
from Orange.widgets import gui
from Orange.widgets.widget import Output

from orangecontrib.text.corpus import Corpus
from orangecontrib.text.guardian import TheGuardianCredentials, TheGuardianAPI, APILimitError
from orangecontrib.text.widgets.utils import CheckListLayout, QueryBox, DatePickerInterval, gui_require, asynchronous


class OWGuardian(OWWidget):
    class CredentialsDialog(OWWidget):
        name = 'The Guardian Credentials'
        want_main_area = False
        resizing_enabled = False
        cm_key = CredentialManager('The Guardian API Key')
        key_input = 'test'

        class Error(OWWidget.Error):
            invalid_credentials = Msg('These credentials are invalid.')

        def __init__(self, parent):
            super().__init__()
            self.parent = parent
            self.api = None

            form = QFormLayout()
            form.setContentsMargins(5, 5, 5, 5)
            self.key_edit = gui.lineEdit(self, self, 'key_input', controlWidth=350)
            form.addRow('Key:', self.key_edit)
            self.controlArea.layout().addLayout(form)
            self.submit_button = gui.button(self.controlArea, self, 'OK', self.accept)

            self.load_credentials()

        def load_credentials(self):
            if self.cm_key.key:
                self.key_edit.setText(self.cm_key.key)

        def save_credentials(self):
            self.cm_key.key = self.key_input

        def check_credentials(self):
            api = TheGuardianCredentials(self.key_input)
            if api.valid:
                self.save_credentials()
            else:
                api = None
            self.api = api

        def accept(self, silent=False):
            if not silent: self.Error.invalid_credentials.clear()
            self.check_credentials()
            if self.api:
                self.parent.update_api(self.api)
                super().accept()
            elif not silent:
                self.Error.invalid_credentials()

    name = 'The Guardian'
    description = 'Fetch articles from The Guardian API.'
    icon = 'icons/TheGuardian.svg'
    priority = 120

    class Outputs:
        corpus = Output("Corpus", Corpus)

    want_main_area = False
    resizing_enabled = False

    recent_queries = Setting([])
    date_from = Setting((datetime.now().date() - timedelta(365)))
    date_to = Setting(datetime.now().date())
    attributes = [feat.name for feat, _ in TheGuardianAPI.metas if
                  isinstance(feat, StringVariable)]
    text_includes = Setting([feat.name for feat in TheGuardianAPI.text_features])

    class Warning(OWWidget.Warning):
        no_text_fields = Msg('Text features are inferred when none are selected.')

    class Error(OWWidget.Error):
        no_api = Msg('Please provide a valid API key.')
        no_query = Msg('Please provide a query.')
        limit_exceeded = Msg('Requests limit reached.')

    def __init__(self):
        super().__init__()
        self.corpus = None
        self.api = None
        self.output_info = ''

        # API Key
        self.api_dlg = self.CredentialsDialog(self)
        self.api_dlg.accept(silent=True)
        gui.button(self.controlArea, self, 'The Guardian API Key',
                   callback=self.api_dlg.exec_,
                   focusPolicy=Qt.NoFocus)

        # Query
        query_box = gui.widgetBox(self.controlArea, 'Query', addSpace=True)
        self.query_box = QueryBox(query_box, self, self.recent_queries,
                                  callback=self.new_query_input)

        # Year box
        date_box = gui.hBox(query_box)
        DatePickerInterval(date_box, self, 'date_from', 'date_to',
                           min_date=None, max_date=date.today(),
                           margin=(0, 3, 0, 0))

        # Text includes features
        self.controlArea.layout().addWidget(
            CheckListLayout('Text includes', self, 'text_includes',
                            self.attributes,
                            cols=2, callback=self.set_text_features))

        # Output
        info_box = gui.hBox(self.controlArea, 'Output')
        gui.label(info_box, self, 'Articles: %(output_info)s')

        # Buttons
        self.button_box = gui.hBox(self.controlArea)

        self.search_button = gui.button(self.button_box, self, 'Search',
                                        self.start_stop,
                                        focusPolicy=Qt.NoFocus)

    def update_api(self, api):
        self.Error.no_api.clear()
        self.api = TheGuardianAPI(api, on_progress=self.progress_with_info,
                                  should_break=self.search.should_break)

    def new_query_input(self):
        self.search.stop()
        self.search()

    def start_stop(self):
        if self.search.running:
            self.search.stop()
        else:
            self.query_box.synchronize(silent=True)
            self.run_search()

    @gui_require('api', 'no_api')
    @gui_require('recent_queries', 'no_query')
    def run_search(self):
        self.Error.limit_exceeded.clear()
        try:
            self.search()
        except APILimitError:
            self.Error.limit_exceeded()

    @asynchronous
    def search(self):
        return self.api.search(self.recent_queries[0],
                               self.date_from, self.date_to)

    @search.callback(should_raise=False)
    def progress_with_info(self, n_retrieved, n_all):
        self.progressBarSet(100 * (n_retrieved / n_all if n_all else 1))  # prevent division by 0
        self.output_info = '{}/{}'.format(n_retrieved, n_all)

    @search.on_start
    def on_start(self):
        self.Error.no_query.clear()
        self.progressBarInit()
        self.search_button.setText('Stop')
        self.Outputs.corpus.send(None)

    @search.on_result
    def on_result(self, result):
        self.search_button.setText('Search')
        self.progressBarFinished()
        self.corpus = result
        self.set_text_features()

    def set_text_features(self):
        self.Warning.no_text_fields.clear()
        if not self.text_includes:
            self.Warning.no_text_fields()

        if self.corpus is not None:
            vars_ = [var for var in self.corpus.domain.metas if var.name in self.text_includes]
            self.corpus.set_text_features(vars_ or None)
            self.Outputs.corpus.send(self.corpus)

    def send_report(self):
        self.report_items([
            ('Query', self.recent_queries[0]),
            ('Date from', self.date_from),
            ('Date to', self.date_to),
            ('Text includes', ', '.join(self.text_includes)),
            ('Output', self.output_info or 'Nothing'),
        ])

if __name__ == '__main__':
    app = QApplication([])
    widget = OWGuardian()
    widget.show()
    app.exec()
    widget.saveSettings()
