import os
from PyQt4 import QtGui

from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget
from Orange.widgets import gui
from orangecontrib.text.corpus import Corpus, get_sample_corpora_dir


class Output:
    CORPUS = "Corpus"

class OWLoadCorpus(OWWidget):
    name = "Corpus"
    description = "Load a corpus of text documents, (optionally) tagged with categories."
    icon = "icons/TextFile.svg"

    outputs = [(Output.CORPUS, Corpus)]
    want_main_area = False

    dlgFormats = "Only text files (*.txt)"

    recent_files = Setting(["(none)"])

    def __init__(self):
        super().__init__()

        # Refresh recent files
        self.recent_files = [fn for fn in self.recent_files
                             if os.path.exists(fn)]

        # Browse file box
        fbox = gui.widgetBox(self.controlArea, "Corpus file", orientation=0)

        # Drop-down for recent files
        self.file_combo = QtGui.QComboBox(fbox)
        self.file_combo.setMinimumWidth(300)
        fbox.layout().addWidget(self.file_combo)
        self.file_combo.activated[int].connect(self.select_file)

        # Browse button
        browse = gui.button(fbox, self, 'Browse', callback=self.browse_file)
        browse.setIcon(self.style().standardIcon(QtGui.QStyle.SP_DirOpenIcon))
        browse.setSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)

        # Reload button
        reload = gui.button(fbox, self, "Reload", callback=self.reload, default=True)
        reload.setIcon(self.style().standardIcon(QtGui.QStyle.SP_BrowserReload))
        reload.setSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)

        # Corpus info
        ibox = gui.widgetBox(self.controlArea, "Corpus info", addSpace=True)

        corp_info = "Corpus of 0 documents with 0 optional attributes."
        self.info_label = gui.label(ibox, self, corp_info)

        # Load the most recent file
        self.set_file_list()
        if len(self.recent_files) > 0:
            self.open_file(self.recent_files[0])

    def set_file_list(self):
        self.file_combo.clear()
        if not self.recent_files:
            self.file_combo.addItem("(none)")
        for file in self.recent_files:
            if file == "(none)":
                self.file_combo.addItem("(none)")
            else:
                self.file_combo.addItem(os.path.split(file)[1])
        self.file_combo.addItem("Browse documentation corpora ...")

    def reload(self):
        if self.recent_files:
            return self.open_file(self.recent_files[0])

    def select_file(self, n):
        if n < len(self.recent_files) :
            name = self.recent_files[n]
            del self.recent_files[n]
            self.recent_files.insert(0, name)
        elif n:
            self.browse_file(True)

        if len(self.recent_files) > 0:
            self.set_file_list()
            self.open_file(self.recent_files[0])

    def browse_file(self, demos_loc=False):
        start_file = os.path.expanduser("~/")
        if demos_loc:
            start_file = get_sample_corpora_dir()
        filename = QtGui.QFileDialog.getOpenFileName(
            self, 'Open Orange Document Corpus', start_file, self.dlgFormats)
        if not filename:
            return
        self.recent_files.insert(0, filename)
        self.open_file(filename)

    def open_file(self, path):
        corpus = Corpus.from_file(path)
        self.info_label.setText("Corpus of {} documents with {} optional attribute{}.".format(
            len(corpus),
            len(corpus.domain.metas)-1,
            '' if len(corpus.domain.metas) == 2 else 's',
        ))
        self.send(Output.CORPUS, corpus)
