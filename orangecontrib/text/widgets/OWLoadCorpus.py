import os
from PyQt4 import QtGui

from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget
from Orange.widgets import gui
from Orange.widgets.data.owselectcolumns import VariablesListItemModel, VariablesListItemView
from orangecontrib.text.corpus import Corpus, get_sample_corpora_dir


class Output:
    CORPUS = "Corpus"

class OWLoadCorpus(OWWidget):
    name = "Corpus"
    description = "Load a corpus of text documents, (optionally) tagged with categories."
    icon = "icons/TextFile.svg"
    priority = 10

    outputs = [(Output.CORPUS, Corpus)]
    want_main_area = False

    dlgFormats = "Only tab files (*.tab)"

    recent_files = Setting(["(none)"])

    def __init__(self):
        super().__init__()

        self.corpus = None

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
        corp_info = "Corpus of 0 documents."
        self.info_label = gui.label(ibox, self, corp_info)

        # Used Text Features
        fbox = gui.widgetBox(self.controlArea, orientation=0)
        ubox = gui.widgetBox(fbox, "Used Text Features", addSpace=True)
        self.used_attrs = VariablesListItemModel()
        self.used_attrs_view = VariablesListItemView()
        self.used_attrs_view.setModel(self.used_attrs)
        ubox.layout().addWidget(self.used_attrs_view)

        aa = self.used_attrs
        aa.dataChanged.connect(self.update_feature_selection)
        aa.rowsInserted.connect(self.update_feature_selection)
        aa.rowsRemoved.connect(self.update_feature_selection)

        # Ignored Text Features
        ibox = gui.widgetBox(fbox, "Ignored Text Features", addSpace=True)
        self.unused_attrs = VariablesListItemModel()
        self.unused_attrs_view = VariablesListItemView()
        self.unused_attrs_view.setModel(self.unused_attrs)
        ibox.layout().addWidget(self.unused_attrs_view)

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
        if filename in self.recent_files:
            self.recent_files.remove(filename)
        self.recent_files.insert(0, filename)
        self.set_file_list()
        self.open_file(filename)

    def open_file(self, path):
        self.error(1, '')
        self.used_attrs[:] = []
        self.unused_attrs[:] = []

        try:
            self.corpus = Corpus.from_file(path)
            for i in self.corpus.used_features:
                self.used_attrs.append(i)
            for i in self.corpus.domain.metas:
                if i not in self.corpus.used_features:
                    self.unused_attrs.append(i)

            self.info_label.setText("Corpus of {} documents.".format(len(self.corpus)))
            self.send(Output.CORPUS, self.corpus)
        except BaseException as err:
            self.error(1, str(err))

    def update_feature_selection(self):
        if self.corpus is not None:
            self.corpus.regenerate_documents(self.used_attrs)
            self.send(Output.CORPUS, self.corpus)