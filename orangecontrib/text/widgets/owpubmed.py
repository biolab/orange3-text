import os
from datetime import date

from AnyQt.QtCore import QDate, Qt
from AnyQt.QtWidgets import (QApplication, QComboBox, QDateEdit, QTextEdit,
                             QFrame, QDialog, QCalendarWidget, QVBoxLayout)
from validate_email import validate_email

from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget, Msg
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.pubmed import (
    Pubmed, PUBMED_TEXT_FIELDS
)


def _i(name, icon_path='icons'):
    widget_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(widget_path, icon_path, name)


class Output:
    CORPUS = 'Corpus'


class OWPubmed(OWWidget):
    name = 'Pubmed'
    description = 'Fetch data from Pubmed.'
    icon = 'icons/Pubmed.svg'
    priority = 140

    outputs = [(Output.CORPUS, Corpus)]
    want_main_area = False
    resizing_enabled = False

    QT_DATE_FORMAT = 'yyyy-MM-dd'
    PY_DATE_FORMAT = '%Y-%m-%d'
    MIN_DATE = date(1800, 1, 1)

    # Settings.
    recent_emails = Setting([])
    author = Setting('')
    pub_date_from = Setting('')
    pub_date_to = Setting('')
    recent_keywords = Setting([])
    last_advanced_query = Setting('')
    num_records = Setting(1000)

    # Text includes checkboxes.
    includes_authors = Setting(True)
    includes_title = Setting(True)
    includes_mesh = Setting(True)
    includes_abstract = Setting(True)
    includes_url = Setting(True)

    class Warning(OWWidget.Warning):
        no_query = Msg('Please specify the keywords for this query.')

    class Error(OWWidget.Error):
        api_error = Msg('API error: {}.')

    def __init__(self):
        super().__init__()

        self.output_corpus = None
        self.pubmed_api = None
        self.progress = None
        self.email_is_valid = False
        self.record_count = 0
        self.download_running = False

        # To hold all the controls. Makes access easier.
        self.pubmed_controls = []

        h_box = gui.hBox(self.controlArea)
        label = gui.label(h_box, self, 'Email:')
        label.setMaximumSize(label.sizeHint())
        # Drop-down for recent emails.
        self.email_combo = QComboBox(h_box)
        self.email_combo.setMinimumWidth(150)
        self.email_combo.setEditable(True)
        self.email_combo.lineEdit().textChanged.connect(self.sync_email)
        h_box.layout().addWidget(self.email_combo)
        self.email_combo.activated[int].connect(self.select_email)

        # RECORD SEARCH
        self.search_tabs = gui.tabWidget(self.controlArea)
        # --- Regular search ---
        regular_search_box = gui.widgetBox(self.controlArea, addSpace=True)

        # Author
        self.author_input = gui.lineEdit(regular_search_box, self, 'author',
                                         'Author:', orientation=Qt.Horizontal)
        self.pubmed_controls.append(self.author_input)

        h_box = gui.hBox(regular_search_box)
        year_box = gui.widgetBox(h_box, orientation=Qt.Horizontal)
        min_date = QDate.fromString(
                self.MIN_DATE.strftime(self.PY_DATE_FORMAT),
                self.QT_DATE_FORMAT
        )

        if not self.pub_date_from:
            self.pub_date_from = self.MIN_DATE.strftime(self.PY_DATE_FORMAT)
        if not self.pub_date_to:
            self.pub_date_to = date.today().strftime(self.PY_DATE_FORMAT)

        self.date_from = QDateEdit(
                QDate.fromString(self.pub_date_from, self.QT_DATE_FORMAT),
                displayFormat=self.QT_DATE_FORMAT,
                minimumDate=min_date,
                calendarPopup=True
        )
        self.date_to = QDateEdit(
                QDate.fromString(self.pub_date_to, self.QT_DATE_FORMAT),
                displayFormat=self.QT_DATE_FORMAT,
                minimumDate=min_date,
                calendarPopup=True
        )

        self.date_from.dateChanged.connect(
            lambda date: setattr(self, 'pub_date_from',
                                 date.toString(self.QT_DATE_FORMAT)))
        self.date_to.dateChanged.connect(
            lambda date: setattr(self, 'pub_date_to',
                                 date.toString(self.QT_DATE_FORMAT)))
        self.pubmed_controls.append(self.date_from)
        self.pubmed_controls.append(self.date_to)

        gui.label(year_box, self, 'From:')
        year_box.layout().addWidget(self.date_from)
        gui.label(year_box, self, 'to:')
        year_box.layout().addWidget(self.date_to)

        # Keywords.
        h_box = gui.hBox(regular_search_box)
        label = gui.label(h_box, self, 'Query:')
        label.setMaximumSize(label.sizeHint())
        self.keyword_combo = QComboBox(h_box)
        self.keyword_combo.setMinimumWidth(150)
        self.keyword_combo.setEditable(True)
        h_box.layout().addWidget(self.keyword_combo)
        self.keyword_combo.activated[int].connect(self.select_keywords)
        self.pubmed_controls.append(self.keyword_combo)

        tab_height = regular_search_box.sizeHint()
        regular_search_box.setMaximumSize(tab_height)

        # --- Advanced search ---
        advanced_search_box = gui.widgetBox(self.controlArea, addSpace=True)
        # Advanced search query.
        h_box = gui.hBox(advanced_search_box)
        self.advanced_query_input = QTextEdit(h_box)
        h_box.layout().addWidget(self.advanced_query_input)
        self.advanced_query_input.setMaximumSize(tab_height)
        self.pubmed_controls.append(self.advanced_query_input)

        gui.createTabPage(self.search_tabs, 'Regular search',
                          regular_search_box)
        gui.createTabPage(self.search_tabs, 'Advanced search',
                          advanced_search_box)

        # Search info label.
        self.search_info_label = gui.label(
                self.controlArea, self,
                'Number of records found: /')

        # Search for records button.
        self.run_search_button = gui.button(
                self.controlArea,
                self,
                'Find records',
                callback=self.run_search,
                tooltip='Performs a search for articles that fit the '
                        'specified parameters.')
        self.pubmed_controls.append(self.run_search_button)

        h_line = QFrame()
        h_line.setFrameShape(QFrame.HLine)
        h_line.setFrameShadow(QFrame.Sunken)
        self.controlArea.layout().addWidget(h_line)

        # RECORD RETRIEVAL
        # Text includes box.
        text_includes_box = gui.widgetBox(self.controlArea,
                                          'Text includes', addSpace=True)
        self.authors_checkbox = gui.checkBox(text_includes_box, self,
                                          'includes_authors', 'Authors')
        self.title_checkbox = gui.checkBox(text_includes_box, self,
                                        'includes_title', 'Article title')
        self.mesh_checkbox = gui.checkBox(text_includes_box, self,
                                       'includes_mesh', 'Mesh headings')
        self.abstract_checkbox = gui.checkBox(text_includes_box, self,
                                           'includes_abstract', 'Abstract')
        self.url_checkbox = gui.checkBox(text_includes_box, self,
                                         'includes_url', 'URL')
        self.pubmed_controls.append(self.authors_checkbox)
        self.pubmed_controls.append(self.title_checkbox)
        self.pubmed_controls.append(self.mesh_checkbox)
        self.pubmed_controls.append(self.abstract_checkbox)
        self.pubmed_controls.append(self.url_checkbox)

        # Num. records.
        h_box = gui.hBox(self.controlArea)
        label = gui.label(h_box, self, 'Retrieve')
        label.setMaximumSize(label.sizeHint())
        self.num_records_input = gui.spin(h_box, self, 'num_records',
                                          minv=1, maxv=100000)
        self.max_records_label = gui.label(h_box, self, 'records from /.')
        self.max_records_label.setMaximumSize(self.max_records_label
                                              .sizeHint())
        self.pubmed_controls.append(self.num_records_input)

        # Download articles.
        # Search for records button.
        self.retrieve_records_button = gui.button(
                self.controlArea,
                self,
                'Retrieve records',
                callback=self.retrieve_records,
                tooltip='Retrieves the specified documents.')
        self.pubmed_controls.append(self.retrieve_records_button)

        # Num. retrieved records info label.
        self.retrieval_info_label = gui.label(
                self.controlArea,
                self,
                'Number of records retrieved: /')

        # Load the most recent emails.
        self.set_email_list()

        # Load the most recent queries.
        self.set_keyword_list()

        # Check the email and enable controls accordingly.
        if self.recent_emails:
            email = self.recent_emails[0]
            self.email_is_valid = validate_email(email)

        self.enable_controls()

    def sync_email(self):
        email = self.email_combo.currentText()
        self.email_is_valid = validate_email(email)
        self.enable_controls()

    def enable_controls(self):
        # Enable/disable controls accordingly.
        for control in self.pubmed_controls:
            control.setEnabled(self.email_is_valid)
        if self.pubmed_api is None or self.pubmed_api.search_record_count == 0:
            self.retrieve_records_button.setEnabled(False)
        if not self.email_is_valid:
            self.email_combo.setFocus()

    def run_search(self):
        self.Error.clear()
        self.Warning.clear()
        self.run_search_button.setEnabled(False)
        self.retrieve_records_button.setEnabled(False)

        # Add the email to history.
        email = self.email_combo.currentText()
        if email not in self.recent_emails:
            self.recent_emails.insert(0, email)

        # Check if the PubMed object is present.
        if self.pubmed_api is None:
            self.pubmed_api = Pubmed(
                    email=email,
                    progress_callback=self.api_progress_callback,
                    error_callback=self.api_error_callback,
            )

        if self.search_tabs.currentIndex() == 0:
            # Get query parameters.
            terms = self.keyword_combo.currentText().split()
            authors = self.author_input.text().split()

            error = self.pubmed_api._search_for_records(
                    terms, authors, self.pub_date_from, self.pub_date_to
            )
            if error is not None:
                self.Error.api_error(str(error))
                return

            if self.keyword_combo.currentText() not in self.recent_keywords:
                self.recent_keywords.insert(
                        0,
                        self.keyword_combo.currentText()
                )
        else:
            query = self.advanced_query_input.toPlainText()
            if not query:
                self.Warning.no_query()
                self.run_search_button.setEnabled(True)
                self.retrieve_records_button.setEnabled(True)
                return
            error = self.pubmed_api._search_for_records(advanced_query=query)

            if error is not None:
                self.Error.api_error(str(error))
                return

            self.last_advanced_query = query

        self.enable_controls()
        self.update_search_info()

    def retrieve_records(self):
        self.Warning.clear()
        self.Error.clear()

        if self.pubmed_api is None:
            return

        if self.download_running:
            self.download_running = False
            self.run_search_button.setEnabled(True)
            self.retrieve_records_button.setText('Retrieve records')
            self.pubmed_api.stop_retrieving()
            return

        self.download_running = True
        self.run_search_button.setEnabled(False)
        self.output_corpus = None  # Clear the old records.

        # Change the button label.
        self.retrieve_records_button.setText('Stop retrieving')

        # Text fields.
        text_includes_params = [
            self.includes_authors,
            self.includes_title,
            self.includes_mesh,
            self.includes_abstract,
            self.includes_url,
            True,  # Publication date field; included always.
        ]
        required_text_fields = [
            field
            for field_name, field
            in zip(text_includes_params, PUBMED_TEXT_FIELDS)
            if field_name
        ]

        batch_size = min(Pubmed.MAX_BATCH_SIZE, self.num_records) + 1
        with self.progressBar(self.num_records/batch_size) as progress:
            self.progress = progress
            self.output_corpus = self.pubmed_api._retrieve_records(
                    self.num_records,
                    required_text_fields
            )
        self.retrieve_records_button.setText('Retrieve records')
        self.download_running = False

        self.send(Output.CORPUS, self.output_corpus)
        self.update_retrieval_info()
        self.run_search_button.setEnabled(True)

    def api_progress_callback(self, start_at=None):
        if start_at is not None:
            self.progress.count = start_at
        else:
            self.progress.advance()

    def api_error_callback(self, error):
        self.Error.api_error(str(error))
        if self.progress is not None:
            self.progress.finish()

    def update_search_info(self):
        max_records_count = min(
                self.pubmed_api.MAX_RECORDS,
                self.pubmed_api.search_record_count
        )
        self.search_info_label.setText(
                'Number of retrievable records for '
                'this search query: {} '.format(max_records_count)
        )
        self.max_records_label.setText(
                'records from {}.'.format(max_records_count)
        )
        self.max_records_label.setMaximumSize(self.max_records_label
                                              .sizeHint())

        self.num_records_input.setMaximum(max_records_count)
        self.retrieve_records_button.setFocus()

    def update_retrieval_info(self):
        document_count = 0
        if self.output_corpus is not None:
            document_count = len(self.output_corpus)

        self.retrieval_info_label.setText(
                'Number of records retrieved: {} '.format(document_count)
        )
        self.retrieval_info_label.setMaximumSize(
                self.retrieval_info_label.sizeHint()
        )

    def select_email(self, n):
        if n < len(self.recent_emails):
            email = self.recent_emails[n]
            del self.recent_emails[n]
            self.recent_emails.insert(0, email)

        if len(self.recent_emails) > 0:
            self.set_email_list()

    def set_email_list(self):
        self.email_combo.clear()
        for email in self.recent_emails:
            self.email_combo.addItem(email)

    def select_keywords(self, n):
        if n < len(self.recent_keywords):
            keywords = self.recent_keywords[n]
            del self.recent_keywords[n]
            self.recent_keywords.insert(0, keywords)

        if len(self.recent_keywords) > 0:
            self.set_keyword_list()

    def set_keyword_list(self):
        self.keyword_combo.clear()
        if not self.recent_keywords:
            # Sample queries.
            self.recent_keywords.append('orchid')
            self.recent_keywords.append('hypertension')
            self.recent_keywords.append('blood pressure')
            self.recent_keywords.append('radiology')
        for keywords in self.recent_keywords:
            self.keyword_combo.addItem(keywords)

    def open_calendar(self, widget):
        cal_dlg = CalendarDialog(self, 'Date picker')
        if cal_dlg.exec_():
            widget.setText(cal_dlg.picked_date)

    def send_report(self):
        if not self.pubmed_api:
            return
        max_records_count = min(
            self.pubmed_api.MAX_RECORDS,
            self.pubmed_api.search_record_count
        )
        if self.search_tabs.currentIndex() == 0:
            terms = self.keyword_combo.currentText()
            authors = self.author_input.text()
            self.report_items((
                ('Query', terms if terms else None),
                ('Authors', authors if authors else None),
                ('Date', 'from {} to {}'.format(self.pub_date_from,
                                          self.pub_date_to)),
                ('Number of records retrieved', '{}/{}'.format(len(
                    self.output_corpus) if self.output_corpus else 0,
                                                               max_records_count))
            ))
        else:
            query = self.advanced_query_input.toPlainText()
            self.report_items((
                ('Query', query if query else None),
                ('Number of records retrieved', '{}/{}'.format(len(
                    self.output_corpus) if self.output_corpus else 0,
                                                               max_records_count))
            ))


class CalendarDialog(QDialog):

    picked_date = None
    source = None
    parent = None

    def __init__(self, parent, windowTitle='Date picker'):
        super().__init__(parent, windowTitle=windowTitle)

        self.parent = parent

        self.setLayout(QVBoxLayout())
        self.mainArea = gui.widgetBox(self)
        self.layout().addWidget(self.mainArea)

        self.cal = QCalendarWidget(self)
        self.cal.setGridVisible(True)
        self.cal.move(20, 20)
        self.cal.clicked[QDate].connect(self.set_date)
        self.mainArea.layout().addWidget(self.cal)

        # Set the default date.
        self.picked_date = self.cal.selectedDate().toString('yyyy/MM/dd')

        gui.button(self.mainArea, self, 'OK', lambda: QDialog.accept(self))

    def set_date(self, date):
        self.picked_date = date.toString('yyyy/MM/dd')

if __name__ == '__main__':
    app = QApplication([])
    widget = OWPubmed()
    widget.show()
    app.exec()
