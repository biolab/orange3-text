import os
from PyQt4 import QtGui

from Orange.widgets.widget import OWWidget
from Orange.widgets import gui
from orangecontrib.text.corpus import Corpus


class Output:
    CORPUS = "Corpus"

class OWLoadCorpus(OWWidget):
    name = "Corpus"
    description = "Load a corpus of text documents, (optionally) tagged with categories."
    icon = "icons/TextFile.svg"

    outputs = [(Output.CORPUS, Corpus)]
    want_main_area = False

    dlgFormats = "Only text files (*.txt)"

    def __init__(self):
        super().__init__()

        # Browse.
        browse_file_box = gui.widgetBox(self.controlArea, "Corpus file",
                                        orientation=0, addSpace=True)

        # Set the label.
        self.file_label = gui.label(browse_file_box, self, 'Browse for files ...')
        self.file_label.setMinimumWidth(150)

        browse_button = gui.button(browse_file_box, self, '...', callback=self.browse_file)
        browse_button.setIcon(self.style().standardIcon(QtGui.QStyle.SP_DirOpenIcon))
        browse_button.setSizePolicy(QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.Fixed)

        # Corpus info.
        corpus_info_box = gui.widgetBox(self.controlArea, "Corpus info", addSpace=True)

        corp_info = "Corpus consists of 0 documents from 0 different categories."
        self.corp_info_label = gui.label(corpus_info_box, self, corp_info)

    def browse_file(self):
        start_file = os.path.expanduser("~/")
        filename = QtGui.QFileDialog.getOpenFileName(
            self, 'Open Orange Data File', start_file, self.dlgFormats)
        if not filename:
            return
        self.file_label.setText(filename)
        self.open_file(filename)

    def open_file(self, path):
        self.corpus = Corpus(path)

        # Update corpus info.
        categories = set([d.category for d in self.corpus.documents])
        self.corp_info_label.setText("Corpus consists of {} documents from {} different categories."
                                     .format(len(self.corpus.documents), len(categories)))

        self.send(Output.CORPUS, self.corpus)