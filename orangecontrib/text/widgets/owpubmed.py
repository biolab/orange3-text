import os
import re

from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import *
from validate_email import validate_email

from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.pubmed import (
    Pubmed, PUBMED_TEXT_FIELDS
)


def _i(name, icon_path='icons'):
    widget_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(widget_path, icon_path, name)


def eval_date(date):
    if not date:
        return True
    pattern = re.compile('\d{4}/\d{2}/\d{2}')
    return pattern.match(date)


class Output:
    CORPUS = 'Corpus'


class OWPubmed(OWWidget):
    name = 'Pubmed'
    description = 'Load data from the Pubmed api.'
    icon = 'icons/Pubmed.svg'
    priority = 20

    outputs = [(Output.CORPUS, Corpus)]
    want_main_area = False

    # Settings.
    recent_emails = Setting([])
    author = Setting('')
    pub_date_from = Setting('')
    pub_date_to = Setting('')
    recent_keywords = Setting([])
    last_advanced_query = Setting('')
    num_records = Setting(1000)

    # Text includes checkboxes.
    includes_authors = Setting(False)
    includes_title = Setting(False)
    includes_mesh = Setting(False)
    includes_abstract = Setting(True)

    def __init__(self):
        super().__init__()

        self.output_corpus = None
        self.pubmed_api = None
        self.progress = None
        self.email_is_valid = False
        self.record_count = 0

        # To hold all the controls. Makes access easier.
        self.pubmed_controls = []

        h_box = gui.widgetBox(self.controlArea, orientation='horizontal')
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
                                         'Author:', orientation='horizontal')
        self.pubmed_controls.append(self.author_input)
        # Pub. date from.
        h_box = gui.widgetBox(regular_search_box, orientation='horizontal')
        self.pub_date_from_input = gui.lineEdit(h_box, self, 'pub_date_from',
                                                'Published from',
                                                orientation='horizontal')
        self.pub_date_from_input.setPlaceholderText('YYYY/MM/DD')
        self.pub_date_from_input.setMaximumSize(self.pub_date_from_input
                                                .sizeHint())
        # Calendar button from.
        open_calendar_button_from = gui.button(
                h_box, self, '',
                callback=lambda: self.open_calendar(self.pub_date_from_input),
                tooltip='Pick a date using the calendar widget.'
        )
        open_calendar_button_from.setMaximumSize(open_calendar_button_from
                                                 .sizeHint())
        open_calendar_button_from.setIcon(QIcon(_i('calendar.svg')))
        open_calendar_button_from.setIconSize(QtCore.QSize(16, 16))
        open_calendar_button_from.setFocusPolicy(QtCore.Qt.NoFocus)
        open_calendar_button_from.setSizePolicy(QtGui.QSizePolicy.Fixed,
                                                QtGui.QSizePolicy.Fixed)
        # Pub. date to.
        self.pub_date_to_input = gui.lineEdit(h_box, self, 'pub_date_to',
                                              'and', orientation='horizontal')
        self.pub_date_to_input.setPlaceholderText('YYYY/MM/DD')
        self.pub_date_to_input.setMaximumSize(self.pub_date_to_input
                                              .sizeHint())
        # Calendar button to.
        open_calendar_button_to = gui.button(
                h_box, self, '',
                callback=lambda: self.open_calendar(self.pub_date_to_input),
                tooltip='Pick a date using the calendar widget.')
        open_calendar_button_to.setMaximumSize(open_calendar_button_from
                                               .sizeHint())
        open_calendar_button_to.setIcon(QIcon(_i('calendar.svg')))
        open_calendar_button_to.setIconSize(QtCore.QSize(16, 16))
        open_calendar_button_to.setFocusPolicy(QtCore.Qt.NoFocus)
        open_calendar_button_to.setSizePolicy(QtGui.QSizePolicy.Fixed,
                                              QtGui.QSizePolicy.Fixed)

        self.pubmed_controls.append(self.pub_date_from_input)
        self.pubmed_controls.append(self.pub_date_to_input)

        # Keywords.
        h_box = gui.widgetBox(regular_search_box, orientation='horizontal')
        label = gui.label(h_box, self, 'Query:')
        label.setMaximumSize(label.sizeHint())
        self.keyword_combo = QComboBox(h_box)
        self.keyword_combo.setMinimumWidth(150)
        self.keyword_combo.setEditable(True)
        h_box.layout().addWidget(self.keyword_combo)
        self.keyword_combo.activated[int].connect(self.select_keywords)
        self.pubmed_controls.append(self.keyword_combo)

        # --- Advanced search ---
        advanced_search_box = gui.widgetBox(self.controlArea, addSpace=True)
        # Advanced search query.
        h_box = gui.widgetBox(advanced_search_box, orientation='horizontal')
        self.advanced_query_input = QTextEdit(h_box)
        h_box.layout().addWidget(self.advanced_query_input)
        self.pubmed_controls.append(self.advanced_query_input)

        gui.createTabPage(self.search_tabs, 'Regular search',
                          regular_search_box)
        gui.createTabPage(self.search_tabs, 'Advanced search',
                          advanced_search_box)

        # Search info label.
        self.search_info_label = gui.label(
                self.controlArea, self,
                'Number of retrievable records for this search query: /')

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
        self.authors_chbox = gui.checkBox(text_includes_box, self,
                                          'includes_authors', 'Authors')
        self.title_chbox = gui.checkBox(text_includes_box, self,
                                        'includes_title', 'Article title')
        self.mesh_chbox = gui.checkBox(text_includes_box, self,
                                       'includes_mesh', 'Mesh headings')
        self.abstract_chbox = gui.checkBox(text_includes_box, self,
                                           'includes_abstract', 'Abstract')
        self.pubmed_controls.append(self.authors_chbox)
        self.pubmed_controls.append(self.title_chbox)
        self.pubmed_controls.append(self.mesh_chbox)
        self.pubmed_controls.append(self.abstract_chbox)

        # Num. records.
        h_box = gui.widgetBox(self.controlArea, orientation=0)
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
                tooltip='Performs a search for articles that fit the '
                        'specified parameters.')
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
        self.warning(0)
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

            # If no keywords, alert that the query is too vague.
            if not terms:
                self.warning(0, 'Please specify the keywords for this query.')
                self.run_search_button.setEnabled(True)
                self.retrieve_records_button.setEnabled(True)
                return

            # Check date formatting.
            pdate_from_flag = self.pub_date_from and not eval_date(
                    self.pub_date_from
            )
            pdate_to_flag = self.pub_date_to and not eval_date(
                    self.pub_date_to
            )
            if pdate_from_flag or pdate_to_flag:
                self.warning(0, 'Please specify dates with digits in '
                                'YYYY/MM/DD format.')
                self.run_search_button.setEnabled(True)
                self.retrieve_records_button.setEnabled(True)
                return

            error = self.pubmed_api._search_for_records(terms,
                                                        authors,
                                                        self.pub_date_from,
                                                        self.pub_date_to)
            if error is not None:
                self.warning(0, str(error))
                return

            if self.keyword_combo.currentText() not in self.recent_keywords:
                self.recent_keywords.insert(
                        0,
                        self.keyword_combo.currentText()
                )
        else:
            query = self.advanced_query_input.toPlainText()
            if not query:
                self.warning(0, 'Please specify the keywords for this query.')
                self.run_search_button.setEnabled(True)
                self.retrieve_records_button.setEnabled(True)
                return
            error = self.pubmed_api._search_for_records_advanced(query)

            if error is not None:
                self.warning(0, str(error))
                return

            self.last_advanced_query = query

        self.enable_controls()
        self.update_search_info()

    def retrieve_records(self):
        """
        Retrieves the records that were queried with '_search_for_records()'.
        If retrieval was successful, generates a corpus with the text fields as
        meta attributes.
        :param num_records: The number of records to retrieve.
        :type num_records: int
        :return: orangecontrib.text.corpus.Corpus
        """
        self.warning(0)
        self.error(0)
        if self.pubmed_api is None:
            return

        self.output_corpus = None  # Clear the old records.

        # Text fields.
        text_includes_params = [
            self.includes_authors,
            self.includes_title,
            self.includes_mesh,
            self.includes_abstract,
            True,  # Publication date field; included always.
        ]
        required_text_fields = [
            field
            for field_name, field
            in zip(text_includes_params, PUBMED_TEXT_FIELDS)
            if field_name
        ]

        batch_size = min(Pubmed.DEFAULT_BATCH_SIZE, self.num_records)
        with self.progressBar(self.num_records/batch_size) as progress:
            self.progress = progress
            self.output_corpus = self.pubmed_api._retrieve_records(
                    self.num_records,
                    required_text_fields
            )

        self.send(Output.CORPUS, self.output_corpus)
        self.update_retrieval_info()

    def api_progress_callback(self):
        self.progress.advance()

    def api_error_callback(self, error):
        self.error(0, str(error))
        if self.progress is not None:
            self.progress.finish()

    def update_search_info(self):
        self.search_info_label.setText(
                'Number of retrievable records for this search query: {} '
                    .format(min(100000, self.pubmed_api.search_record_count))
        )
        self.max_records_label.setText(
                'records from {}.'
                    .format(min(100000, self.pubmed_api.search_record_count))
        )
        self.max_records_label.setMaximumSize(self.max_records_label
                                              .sizeHint())
        self.num_records = min(self.num_records,
                               self.pubmed_api.search_record_count)

    def update_retrieval_info(self):
        self.retrieval_info_label.setText(
                'Number of records retrieved: {} '.format(
                        len(self.output_corpus)
                )
        )
        self.retrieval_info_label.setMaximumSize(self.retrieval_info_label
                                                 .sizeHint())

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

        self.cal = QtGui.QCalendarWidget(self)
        self.cal.setGridVisible(True)
        self.cal.move(20, 20)
        self.cal.clicked[QtCore.QDate].connect(self.set_date)
        self.mainArea.layout().addWidget(self.cal)

        # Set the default date.
        self.picked_date = self.cal.selectedDate().toString('yyyy/MM/dd')

        gui.button(self.mainArea, self, 'OK', lambda: QDialog.accept(self))

    def set_date(self, date):
        self.picked_date = date.toString('yyyy/MM/dd')
