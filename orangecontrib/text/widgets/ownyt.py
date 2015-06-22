import os
import math
from time import sleep
from PyQt4 import QtCore
from PyQt4.QtGui import *

from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget
from Orange.widgets import gui
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.nyt import NYT


def _i(name, icon_path="icons"):
    widget_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(widget_path, icon_path, name)

class Output:
    CORPUS = "Corpus"

class OWNYT(OWWidget):
    name = "New York Times"
    description = "Load data from the New York Times article search API."
    icon = "icons/TextFile.svg"
    priority = 20

    outputs = [(Output.CORPUS, Corpus)]
    want_main_area = False

    output_corpus = None
    # Response info.
    all_hits = 0
    num_retrieved = 0
    # NYT info:
    nyt_api = None
    api_key = ""
    api_key_is_valid = False
    query_running = False

    # Settings.
    recent_queries = Setting([])
    recent_api_keys = Setting([])
    year_from = Setting("")
    year_to = Setting("")
    # Text includes checkboxes.
    includes_headline = Setting(True)
    includes_lead_paragraph = Setting(True)
    includes_snippet = Setting(False)
    includes_abstract = Setting(False)
    includes_print_page = Setting(False)
    includes_keywords = Setting(False)

    def __init__(self):
        super().__init__()

        # Refresh recent queries.
        self.recent_queries = [query for query in self.recent_queries]

        # To hold all the controls. Makes access easier.
        self.nyt_controls = []

        # Root box.
        parameter_box = gui.widgetBox(self.controlArea, addSpace=True)

        # API key box.
        api_key_box = gui.widgetBox(parameter_box, orientation=0)
        # Valid API key feedback.
        self.api_key_valid_label = gui.label(api_key_box, self, "")
        if self.api_key_is_valid:
            self.api_key_valid_label.setPixmap(QPixmap(_i("valid.svg"))
                                               .scaled(15, 15, QtCore.Qt.KeepAspectRatio))
        else:
            self.api_key_valid_label.setPixmap(QPixmap(_i("invalid.svg"))
                                           .scaled(15, 15, QtCore.Qt.KeepAspectRatio))
        self.api_key_valid_label.setMaximumSize(self.api_key_valid_label.sizeHint())
        # Set API key button.
        self.open_set_api_key_dialog_button = gui.button(api_key_box, self, 'Article API key',
                                                         callback=self.open_set_api_key_dialog,
                                                         tooltip="Set the API key for this widget.")

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
        self.nyt_controls.append(self.query_combo)

        # Year box.
        # TODO Add calendar widget and month+day support.
        year_box = gui.widgetBox(parameter_box, orientation=0)
        # Inputs for years.
        gui.label(year_box, self, "From year")
        self.year_from_input = gui.lineEdit(year_box, self, "year_from")
        gui.label(year_box, self, "to year")
        self.year_to_input = gui.lineEdit(year_box, self, "year_to")
        self.nyt_controls.append(self.year_from_input)
        self.nyt_controls.append(self.year_to_input)

        # Text includes box.
        self.text_includes_box = gui.widgetBox(self.controlArea,
                                               "Text includes", addSpace=True)
        self.headline_chbox = gui.checkBox(self.text_includes_box, self,
                                           "includes_headline", "Headline")
        self.lead_paragraph_chbox = gui.checkBox(self.text_includes_box, self,
                                                 "includes_lead_paragraph", "Lead Paragraph")
        self.snippet_chbox = gui.checkBox(self.text_includes_box, self,
                                          "includes_snippet", "Snippet")
        self.abstract_chbox = gui.checkBox(self.text_includes_box, self,
                                           "includes_abstract", "Abstract")
        self.print_page_chbox = gui.checkBox(self.text_includes_box, self,
                                             "includes_print_page", "Print page")
        self.keywords_chbox = gui.checkBox(self.text_includes_box, self,
                                           "includes_keywords", "Keywords")
        self.nyt_controls.append(self.headline_chbox)
        self.nyt_controls.append(self.lead_paragraph_chbox)
        self.nyt_controls.append(self.snippet_chbox)
        self.nyt_controls.append(self.abstract_chbox)
        self.nyt_controls.append(self.print_page_chbox)
        self.nyt_controls.append(self.keywords_chbox)

        # Run query button.
        self.run_query_button = gui.button(self.controlArea, self, 'Run query',
                                           callback=self.run_initial_query,
                                           tooltip="Run the chosen NYT article query.")
        self.nyt_controls.append(self.run_query_button)
        h_line = QFrame()
        h_line.setFrameShape(QFrame.HLine)
        h_line.setFrameShadow(QFrame.Sunken)
        self.controlArea.layout().addWidget(h_line)

        # Query info.
        query_info_box = gui.widgetBox(self.controlArea, addSpace=True)
        self.query_info_label = gui.label(query_info_box, self,
                                          "Records: {}\nRetrieved: {}".format("/", "/"))

        # Retrieve other records.
        self.retrieve_other_button = gui.button(self.controlArea, self, 'Retrieve remaining records (max 1000)',
                                                callback=self.retrieve_remaining_records,
                                                tooltip="Retrieve the remaining records obtained in the query.")
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
            self.recent_queries.append("slovenia maze")
            self.recent_queries.append("slovenia zavec")
            self.recent_queries.append("fc bayern")
        for query in self.recent_queries:
            self.query_combo.addItem(query)

    def select_query(self, n):
        if n < len(self.recent_queries):
            name = self.recent_queries[n]
            del self.recent_queries[n]
            self.recent_queries.insert(0, name)

        if len(self.recent_queries) > 0:
            self.set_query_list()

    def run_initial_query(self):
        # Only execute if the NYT object is present(safety lock).
        # Otherwise this method cannot be called anyway.
        if self.nyt_api:
            # Query keywords.
            qkw = self.query_combo.currentText()

            # Text fields.
            text_includes_params = [self.includes_headline, self.includes_lead_paragraph, self.includes_snippet,
                                    self.includes_abstract, self.includes_print_page, self.includes_keywords]

            # Set the query url.
            q = self.nyt_api.set_query_url(qkw, self.year_from, self.year_to, text_includes_params)

            # Execute the query.
            res, cached = self.nyt_api.execute_query(0)

            # Construct a corpus for the output.
            self.output_corpus = Corpus(self.nyt_api.parse_record_json(res))
            self.send(Output.CORPUS, self.output_corpus)

            # Update the response info.
            self.all_hits = res["response"]["meta"]["hits"]
            self.num_retrieved = len(res["response"]["docs"])
            self.query_info_label.setText("Records: {}\nRetrieved: {}"
                                          .format(self.all_hits, self.num_retrieved))

            # Enable 'retrieve remaining' button.
            if self.num_retrieved < self.all_hits:
                self.retrieve_other_button.setText('Retrieve remaining records ({})'
                                                   .format(self.all_hits-self.num_retrieved))
                self.retrieve_other_button.setEnabled(True)
            else:
                self.retrieve_other_button.setText('All records retrieved')
                self.retrieve_other_button.setEnabled(False)

            # Add the query to history.
            if qkw not in self.recent_queries:
                self.recent_queries.insert(0, qkw)

    def retrieve_remaining_records(self):
        # If a query is running, stop it.
        if self.query_running:
            self.query_running = False
            return

        if self.nyt_api:
            num_steps = math.ceil(self.all_hits/10)

            # Update buttons.
            self.retrieve_other_button.setText('Stop retrieving')
            self.open_set_api_key_dialog_button.setEnabled(False)
            self.run_query_button.setEnabled(False)

            # Accumulate remaining results in these lists.
            remaining_docs = []

            self.query_running = True
            self.progressBarInit()
            for i in range(int(self.num_retrieved/10), num_steps):
                # Stop querying if the flag is not set.
                if not self.query_running:
                    break

                # Update the progress bar.
                self.progressBarSet(100.0 * (i/num_steps))

                res, cached = self.nyt_api.execute_query(i)

                docs = self.nyt_api.parse_record_json(res)
                remaining_docs += docs

                # Update the info label.
                self.num_retrieved += len(res["response"]["docs"])
                self.query_info_label.setText("Records: {}\nRetrieved: {}"
                                              .format(self.all_hits, self.num_retrieved))

                if not cached:  # Only wait if an actual request was made.
                    sleep(1)
            self.progressBarFinished()
            self.query_running = False

            # Update the corpus.
            self.output_corpus.extend_corpus(remaining_docs)
            self.send(Output.CORPUS, self.output_corpus)

            if self.num_retrieved == self.all_hits:
                self.retrieve_other_button.setText('All records retrieved')
                self.retrieve_other_button.setEnabled(False)
            else:
                self.retrieve_other_button.setText('Retrieve remaining records ({})'
                                                   .format(self.all_hits-self.num_retrieved))

            self.open_set_api_key_dialog_button.setEnabled(True)
            self.run_query_button.setEnabled(True)

    def check_api_key(self, api_key):
        nyt_api = NYT(api_key)
        self.api_key_updated(nyt_api.check_api_key())

    def api_key_updated(self, is_valid):
        self.api_key_is_valid = is_valid
        self.enable_controls()

        if is_valid:
            self.api_key_valid_label.setPixmap(QPixmap(_i("valid.svg"))
                                               .scaled(15, 15, QtCore.Qt.KeepAspectRatio))
            self.nyt_api = NYT(self.api_key)    # Set the NYT API object, if key is valid.
        else:
            self.api_key_valid_label.setPixmap(QPixmap(_i("invalid.svg"))
                                               .scaled(15, 15, QtCore.Qt.KeepAspectRatio))

    def enable_controls(self):
        for control in self.nyt_controls:
            control.setEnabled(self.api_key_is_valid)

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
        if self.parent.api_key_is_valid:
            self.api_key_check_label.setText("API key is valid.")
        else:
            self.api_key_check_label.setText("API key NOT validated!")

        # Load the most recent API keys.
        self.set_key_list()

    def set_key_list(self):
        self.api_key_combo.clear()
        for key in self.parent.recent_api_keys:
            self.api_key_combo.addItem(key)

    def select_key(self, n):
        if n < len(self.parent.recent_api_keys):
            key = self.parent.recent_api_keys[n]
            del self.parent.recent_api_keys[n]
            self.parent.recent_api_keys.insert(0, key)

        if len(self.parent.recent_api_keys) > 0:
            self.set_key_list()

    def check_api_key(self):
        self.parent.check_api_key(self.api_key_combo.currentText())
        if self.parent.api_key_is_valid:
            self.api_key_check_label.setText("API key is valid.")
        else:
            self.api_key_check_label.setText("API key NOT valid!")

    def accept_changes(self):
        self.parent.check_api_key(self.api_key_combo.currentText())  # On OK check the API key also.

        if self.api_key_combo.currentText() not in self.parent.recent_api_keys:
            self.parent.recent_api_keys.append(self.api_key_combo.currentText())
        self.parent.api_key = self.api_key_combo.currentText()
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
