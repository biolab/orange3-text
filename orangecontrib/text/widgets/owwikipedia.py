from PyQt4 import QtGui, QtCore

from Orange.widgets import gui
from Orange.widgets import settings
from Orange.widgets.widget import OWWidget
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.language_codes import lang2code
from orangecontrib.text.widgets.utils import ComboBox, ListEdit, CheckListLayout


class Output:
    CORPUS = "Corpus"


class OWWikipedia(OWWidget):
    """ Get articles from wikipedia. """

    name = 'Wikipedia'
    priority = 40
    icon = 'icons/Wikipedia.svg'

    outputs = [(Output.CORPUS, Corpus)]
    want_main_area = False
    resizing_enabled = False

    label_width = 1
    widgets_width = 2

    attributes = [
        ('Content', 'content'),
        ('Title', 'title'),
        ('URL', 'url'),
        ('Page ID', 'pageid'),
        ('Revision ID', 'revision_id'),
        ('Summary', 'summary'),
    ]

    query_list = settings.Setting([])
    language = settings.Setting('en')
    corpus_variables = settings.Setting([attr[1] for attr in attributes])

    info_label = 'Articles count {}'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layout = QtGui.QGridLayout()

        row = 0
        query_edit = ListEdit(self, 'query_list', self)
        layout.addWidget(QtGui.QLabel('Query word list:'), row, 0, 1, self.label_width)
        layout.addWidget(query_edit, row, self.label_width, 1, self.widgets_width)

        # Language
        row += 1
        language_edit = ComboBox(self, 'language',
                                 (('Any', None),) + tuple(sorted(lang2code.items())))
        layout.addWidget(QtGui.QLabel('Language:'), row, 0, 1, self.label_width)
        layout.addWidget(language_edit, row, self.label_width, 1, self.widgets_width)

        self.controlArea.layout().addLayout(layout)

        self.controlArea.layout().addWidget(
            CheckListLayout('Text includes', self, 'corpus_variables', self.attributes, cols=2))

        self.info_box = gui.hBox(self.controlArea, 'Info')
        self.result_label = gui.label(self.info_box, self, self.info_label.format(0))

        self.button_box = gui.hBox(self.controlArea)
        self.button_box.layout().addWidget(self.report_button)

        self.search_button = gui.button(self.button_box, self, "Search", self.search)
        self.search_button.setFocusPolicy(QtCore.Qt.NoFocus)

    def send_report(self):
        self.report_items('Query', (('Language', self.language), ('Query', self.query_list)))

    def search(self):
        pass


if __name__ == '__main__':
    app = QtGui.QApplication([])
    widget = OWWikipedia()
    widget.show()
    app.exec()
    widget.saveSettings()
