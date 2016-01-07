import os
import re
import math
import numpy as np
from time import sleep
from PyQt4.QtGui import *
from PyQt4.QtCore import QDate
from PyQt4 import QtCore, QtGui
from datetime import date, datetime, timedelta
from urllib.error import HTTPError, URLError

from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget
from Orange.widgets import gui
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.nyt import NYT, _parse_record_json, NYT_TEXT_FIELDS, _generate_corpus


def _i(name, icon_path="icons"):
    widget_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(widget_path, icon_path, name)

class Output:
    CORPUS = "Corpus"

class OWNYT(OWWidget):
    name = "NY Times"
    description = "Load data from the New York Times article search API."
    icon = "icons/NYTimes.svg"
    priority = 20

    outputs = [(Output.CORPUS, Corpus)]
    want_main_area = False
    resizing_enabled = False

    QT_DATE_FORMAT = 'yyyy-MM-dd'
    PY_DATE_FORMAT = '%Y-%m-%d'
    MIN_DATE = date(1851, 1, 1)

    # Settings.
    recent_queries = Setting([])
    recent_api_keys = Setting([])
    date_from = Setting((datetime.now().date() - timedelta(365)).strftime(PY_DATE_FORMAT))
    date_to = Setting(datetime.now().date().strftime(PY_DATE_FORMAT))
    # Text includes checkboxes.
    includes_headline = Setting(True)
    includes_lead_paragraph = Setting(True)
    includes_snippet = Setting(False)
    includes_abstract = Setting(False)
    includes_keywords = Setting(False)
    includes_type_of_material = Setting(False)
    includes_web_url = Setting(False)
    includes_word_count = Setting(False)


    def __init__(self):
        super().__init__()

        self.output_corpus = None
        self.all_hits = 0
        self.num_retrieved = 0
        self.nyt_api = None
        self.api_key = ""
        self.api_key_is_valid = False
        self.query_running = False

        # To hold all the controls. Makes access easier.
        self.nyt_controls = []

        # Root box.
        parameter_box = gui.widgetBox(self.controlArea, addSpace=True)

        # API key box.
        api_key_box = gui.widgetBox(parameter_box, orientation=0)
        # Valid API key feedback.
        self.api_key_valid_label = gui.label(api_key_box, self, "")
        self.update_validity_icon()
        self.api_key_valid_label.setMaximumSize(self.api_key_valid_label.sizeHint())
        # Set API key button.
        self.open_set_api_key_dialog_button = gui.button(api_key_box, self, 'Article API key',
                                                         callback=self.open_set_api_key_dialog,
                                                         tooltip="Set the API key for this widget.")
        self.open_set_api_key_dialog_button.setFocusPolicy(QtCore.Qt.NoFocus)

        # Query box.
        query_box = gui.widgetBox(parameter_box, orientation=0)
        q_label = gui.label(query_box, self, "Query:")
        q_label.setMaximumSize(q_label.sizeHint())
        # Drop-down for recent queries.
        self.query_combo = QComboBox(query_box)
        self.query_combo.setMinimumWidth(150)
        self.query_combo.setEditable(True)
        query_box.layout().addWidget(self.query_combo)
        self.query_combo.activated[int].connect(self.select_query)
        self.query_combo.lineEdit().returnPressed.connect(self.run_initial_query)
        self.nyt_controls.append(self.query_combo)

        # Year box.
        year_box = gui.widgetBox(parameter_box, orientation=0)

        minDate = QDate.fromString(self.MIN_DATE.strftime(self.PY_DATE_FORMAT),
                                   self.QT_DATE_FORMAT)
        date_from = QDateEdit(QDate.fromString(self.date_from, self.QT_DATE_FORMAT),
                              displayFormat=self.QT_DATE_FORMAT,
                              minimumDate=minDate,
                              calendarPopup=True)
        date_to = QDateEdit(QDate.fromString(self.date_to, self.QT_DATE_FORMAT),
                            displayFormat=self.QT_DATE_FORMAT,
                            minimumDate=minDate,
                            calendarPopup=True)
        date_from.dateChanged.connect(
            lambda date: setattr(self, 'date_from', date.toString(self.QT_DATE_FORMAT)))
        date_to.dateChanged.connect(
            lambda date: setattr(self, 'date_to', date.toString(self.QT_DATE_FORMAT)))

        gui.label(year_box, self, "From:")
        year_box.layout().addWidget(date_from)
        gui.label(year_box, self, "to:")
        year_box.layout().addWidget(date_to)

        self.nyt_controls.append(date_from)
        self.nyt_controls.append(date_to)

        # Text includes box.
        self.text_includes_box = gui.widgetBox(self.controlArea,
                                               "Text includes", addSpace=True)
        self.headline_chbox = gui.checkBox(self.text_includes_box, self,
                                           "includes_headline", "Headline")
        self.lead_paragraph_chbox = gui.checkBox(self.text_includes_box, self,
                                                 "includes_lead_paragraph",
                                                 "Lead paragraph")
        self.snippet_chbox = gui.checkBox(self.text_includes_box, self,
                                          "includes_snippet", "Snippet")
        self.abstract_chbox = gui.checkBox(self.text_includes_box, self,
                                           "includes_abstract", "Abstract")
        self.keywords_chbox = gui.checkBox(self.text_includes_box, self,
                                           "includes_keywords", "Keywords")
        self.type_of_material_chbox = gui.checkBox(self.text_includes_box, self,
                                                   "includes_type_of_material",
                                                   "Article type")
        self.web_url_chbox = gui.checkBox(self.text_includes_box, self,
                                          "includes_web_url", "URL")
        self.word_count_chbox = gui.checkBox(self.text_includes_box, self,
                                             "includes_word_count",
                                             "Word count")
        self.nyt_controls.append(self.headline_chbox)
        self.nyt_controls.append(self.lead_paragraph_chbox)
        self.nyt_controls.append(self.snippet_chbox)
        self.nyt_controls.append(self.abstract_chbox)
        self.nyt_controls.append(self.keywords_chbox)
        self.nyt_controls.append(self.type_of_material_chbox)
        self.nyt_controls.append(self.web_url_chbox)
        self.nyt_controls.append(self.word_count_chbox)

        # Run query button.
        self.run_query_button = gui.button(self.controlArea, self, 'Run query',
                                           callback=self.run_initial_query,
                                           tooltip="Run the chosen NYT article query.")
        self.run_query_button.setFocusPolicy(QtCore.Qt.NoFocus)
        self.nyt_controls.append(self.run_query_button)
        h_line = QFrame()
        h_line.setFrameShape(QFrame.HLine)
        h_line.setFrameShadow(QFrame.Sunken)
        self.controlArea.layout().addWidget(h_line)

        # Query info.
        query_info_box = gui.widgetBox(self.controlArea, addSpace=True)
        self.query_info_label = gui.label(query_info_box, self, "Records: /\nRetrieved: /")

        # Retrieve other records.
        self.retrieve_other_button = gui.button(self.controlArea, self, 'Retrieve remaining records',
                                                callback=self.retrieve_remaining_records,
                                                tooltip="Retrieve the remaining records obtained in the query.")
        self.retrieve_other_button.setFocusPolicy(QtCore.Qt.NoFocus)
        self.retrieve_other_button.setEnabled(False)

        # Load the most recent queries.
        self.set_query_list()

        # Check the API key and enable controls accordingly.
        if self.recent_api_keys:
            self.api_key = self.recent_api_keys[0]
            self.check_api_key(self.api_key)

        self.enable_controls()

    def set_query_list(self):
        self.query_combo.clear()
        if not self.recent_queries:
            # Sample queries.
            self.recent_queries.append("slovenia")
            self.recent_queries.append("text mining")
            self.recent_queries.append("orange data mining")
            self.recent_queries.append("bioinformatics")
        for query in self.recent_queries:
            self.query_combo.addItem(query)

    def select_query(self, n):
        if n < len(self.recent_queries):
            name = self.recent_queries[n]
            del self.recent_queries[n]
            self.recent_queries.insert(0, name)
        else:
            self.recent_queries.insert(0, self.query_combo.currentText())

        if len(self.recent_queries) > 0:
            self.set_query_list()

    def update_validity_icon(self):
        if self.api_key_is_valid:
            self.api_key_valid_label.setPixmap(QPixmap(_i("valid.svg"))
                                               .scaled(15, 15, QtCore.Qt.KeepAspectRatio))
        else:
            self.api_key_valid_label.setPixmap(QPixmap(_i("invalid.svg"))
                                               .scaled(15, 15, QtCore.Qt.KeepAspectRatio))

    def run_initial_query(self):
        self.warning(1)
        self.error(1)
        # Only execute if the NYT object is present(safety lock).
        # Otherwise this method cannot be called anyway.
        if not self.nyt_api:
            return

        # Interrupts on faulty inputs. #
        # Query keywords.
        qkw = self.query_combo.currentText()
        if not qkw:
            self.warning(1, "Please enter a query before attempting to fetch results.")
            return

        # Text fields.
        text_includes_params = [self.includes_headline, self.includes_lead_paragraph, self.includes_snippet,
                                self.includes_abstract, self.includes_keywords,
                                self.includes_type_of_material,
                                self.includes_web_url, self.includes_word_count]
        if True not in text_includes_params:
            self.warning(1, "You must select at least one text field.")
            return

        # Year span.
        date_from = self.validate_date(self.date_from)
        date_to = self.validate_date(self.date_to)

        # Warnings on bad inputs. #
        if date_from is not None:
            if date_from < self.MIN_DATE:
                date_from = self.MIN_DATE
                self.warning(
                    1, self.MIN_DATE.strftime(
                        "There are no records before the year %Y. "
                        "Assumed " + self.PY_DATE_FORMAT + " as start date "
                                                           "for this query."))
            if date_to is not None:
                if date_from > date_to:
                    self.warning(1, "The start date is greater than the end date.")

        # Set the query url.
        required_text_fields = [tp for tf, tp in zip(text_includes_params, NYT_TEXT_FIELDS) if tf]
        self.nyt_api._set_endpoint_url(qkw, date_from, date_to, required_text_fields)

        # Execute the query.
        res, cached, error = self.nyt_api._execute_query(0)

        # Display error if failure.
        if self.display_error_response(res, error):
            return

        # Construct a corpus for the output.
        corpus = _generate_corpus(res["response"]["docs"], required_text_fields)
        self.output_corpus = corpus
        self.send(Output.CORPUS, self.output_corpus)

        # Update the response info.
        self.all_hits = res["response"]["meta"]["hits"]
        self.num_retrieved = len(res["response"]["docs"])
        self.update_info_label()

        # Enable 'retrieve remaining' button.
        if self.num_retrieved < min(self.all_hits, 1000):
            self.retrieve_other_button.setText('Retrieve remaining {} records'
                                               .format(min(self.all_hits, 1000)-self.num_retrieved))
            self.retrieve_other_button.setEnabled(True)
            #self.retrieve_other_button.setFocus()
        else:
            self.retrieve_other_button.setText('All records retrieved')
            self.retrieve_other_button.setEnabled(False)

        # Add the query to history.
        if qkw not in self.recent_queries:
            self.recent_queries.insert(0, qkw)
            self.select_query(0)

    def retrieve_remaining_records(self):
        self.error(1)
        # If a query is running, stop it.
        if self.query_running:
            self.query_running = False
            return

        if not self.nyt_api:
            return

        num_steps = min(math.ceil(self.all_hits/10), 100)

        # Update buttons.
        self.retrieve_other_button.setText('Stop retrieving')
        self.open_set_api_key_dialog_button.setEnabled(False)
        self.run_query_button.setEnabled(False)

        self.query_running = True
        self.progressBarInit()
        for i in range(int(self.num_retrieved/10), num_steps):
            # Stop querying if the flag is not set.
            if not self.query_running:
                break

            res, cached, error = self.nyt_api._execute_query(i)

            # Display error if failure.
            if self.display_error_response(res, error):
                return

            metas, class_values = _parse_record_json(res["response"]["docs"], self.nyt_api.includes_fields)
            docs = []
            for doc in metas:
                docs.append(" ".join([d for d in doc if d is not None]).strip())

            # Update the corpus.
            self.output_corpus.extend_corpus(docs, metas, class_values)

            # Update the info label.
            self.num_retrieved += len(res["response"]["docs"])
            self.update_info_label()

            if not cached:  # Only wait if an actual request was made.
                sleep(1)

            # Update the progress bar.
            self.progressBarSet(100.0 * (i/num_steps))

        self.progressBarFinished()
        self.query_running = False

        self.send(Output.CORPUS, self.output_corpus)

        if self.num_retrieved == min(self.all_hits, 1000):
            self.retrieve_other_button.setText('All available records retrieved')
            self.retrieve_other_button.setEnabled(False)
        else:
            self.retrieve_other_button.setText('Retrieve remaining {} records'
                                               .format(min(self.all_hits, 1000)-self.num_retrieved))
            self.retrieve_other_button.setFocus()

        self.open_set_api_key_dialog_button.setEnabled(True)
        self.run_query_button.setEnabled(True)

    def update_info_label(self):
        info_label = "Records: {}\nRetrieved: {}".format(self.all_hits, self.num_retrieved)
        if self.all_hits > 1000:
            info_label += " (max 1000)"
        self.query_info_label.setText(info_label)

    def display_error_response(self, res, error):
        failure = (not res) or ('response' not in res) or ('docs' not in res['response'])
        if failure and error is not None:
            if isinstance(error, HTTPError):
                self.error(1, "An error occurred (HTTP {})".format(error.code))
            elif isinstance(error, URLError):
                self.error(1, "An error occurred (URL {})".format(error.reason))
        return failure

    def validate_date(self, input_string):
        if not input_string:
            return None
        try:
            return datetime.strptime(input_string, self.PY_DATE_FORMAT).date()
        except ValueError:
            self.warning(1, "Invalid date interval endpoint format (must be YYYY/MM/DD).")
            return None

    def check_api_key(self, api_key):
        nyt_api = NYT(api_key)
        key_valid_flag = nyt_api.check_api_key()

        if key_valid_flag:
            self.api_key = api_key

        self.api_key_updated(key_valid_flag)

    def api_key_updated(self, is_valid):
        self.api_key_is_valid = is_valid
        self.enable_controls()

        self.update_validity_icon()
        if is_valid:
            self.nyt_api = NYT(self.api_key)    # Set the NYT API object, if key is valid.

    def enable_controls(self):
        for control in self.nyt_controls:
            control.setEnabled(self.api_key_is_valid)
        if not self.api_key_is_valid:
            self.open_set_api_key_dialog_button.setFocus()

    def open_set_api_key_dialog(self):
        api_dlg = APIKeyDialog(self, "New York Times API key")
        api_dlg.exec_()


class APIKeyDialog(QDialog):
    def __init__(self, parent, windowTitle="New York Times API key"):
        super().__init__(parent, windowTitle=windowTitle)

        self.parent = parent

        self.setLayout(QVBoxLayout())
        self.layout().setMargin(10)
        self.mainArea = gui.widgetBox(self)
        self.layout().addWidget(self.mainArea)

        # Combo box.
        self.api_key_combo = QComboBox(self.mainArea)
        self.api_key_combo.setEditable(True)
        self.api_key_combo.activated[int].connect(self.select_api_key)
        self.mainArea.layout().addWidget(self.api_key_combo)

        # Buttons
        self.button_box = gui.widgetBox(self.mainArea, orientation="horizontal")
        gui.button(self.button_box, self, "Check", self.check_api_key)
        gui.button(self.button_box, self, "OK", self.accept_changes)
        gui.button(self.button_box, self, "Cancel", self.reject_changes)

        # Label
        self.label_box = gui.widgetBox(self, orientation="horizontal")
        self.api_key_check_label = gui.label(self.label_box, self, "")
        self.update_validity_label()

        # Load the most recent API keys.
        self.set_key_list()

    def set_key_list(self):
        self.api_key_combo.clear()
        for key in self.parent.recent_api_keys:
            self.api_key_combo.addItem(key)

    def update_validity_label(self):
        if self.parent.api_key_is_valid:
            self.api_key_check_label.setText("API key is valid.")
        else:
            self.api_key_check_label.setText("API key not validated!")

    def check_api_key(self):
        self.parent.check_api_key(self.api_key_combo.currentText())
        self.update_validity_label()

    def accept_changes(self):
        self.parent.check_api_key(self.api_key_combo.currentText())  # On OK check the API key also.

        if self.api_key_combo.currentText() not in self.parent.recent_api_keys \
                and self.parent.api_key_is_valid:
            self.parent.recent_api_keys.append(self.api_key_combo.currentText())
        QDialog.accept(self)

    def reject_changes(self):
        QDialog.reject(self)

    def select_api_key(self, n):
        if n < len(self.parent.recent_api_keys):
            key = self.parent.recent_api_keys[n]
            del self.parent.recent_api_keys[n]
            self.parent.recent_api_keys.insert(0, key)

        if len(self.parent.recent_api_keys) > 0:
            self.set_key_list()
