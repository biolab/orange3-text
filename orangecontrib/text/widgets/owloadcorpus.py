import os

from Orange.data.io import FileFormat
from Orange.widgets import gui
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.data.owselectcolumns import VariablesListItemView
from Orange.widgets.settings import Setting, ContextSetting, PerfectDomainContextHandler
from Orange.widgets.widget import OWWidget, Msg
from orangecontrib.text.corpus import Corpus, get_sample_corpora_dir
from orangecontrib.text.widgets.utils import widgets


class Output:
    CORPUS = "Corpus"


class OWLoadCorpus(OWWidget):
    name = "Corpus"
    description = "Load a corpus of text documents, (optionally) tagged with categories."
    icon = "icons/TextFile.svg"
    priority = 10

    outputs = [(Output.CORPUS, Corpus)]
    want_main_area = False
    resizing_enabled = False

    dlgFormats = (
        "All readable files ({});;".format(
            '*' + ' *'.join(FileFormat.readers.keys())) +
        ";;".join("{} (*{})".format(f.DESCRIPTION, ' *'.join(f.EXTENSIONS))
                  for f in sorted(set(FileFormat.readers.values()),
                                  key=list(FileFormat.readers.values()).index)))

    settingsHandler = PerfectDomainContextHandler(
        match_values=PerfectDomainContextHandler.MATCH_VALUES_ALL
    )

    recent_files = Setting([])
    used_vars = ContextSetting([])

    class Error(OWWidget.Error):
        read_file = Msg("Can't read file {} ({})")

    def __init__(self):
        super().__init__()

        self.corpus = None

        # Browse file box
        fbox = gui.widgetBox(self.controlArea, "Corpus file", orientation=0)
        widget = widgets.FileWidget(recent_files=self.recent_files, icon_size=(16, 16), on_open=self.open_file,
                                    directory_aliases={"Browse documentation corpora ...": get_sample_corpora_dir()},
                                    dialog_format=self.dlgFormats, dialog_title='Open Orange Document Corpus',
                                    allow_empty=False, reload_label='Reload', browse_label='Browse')
        fbox.layout().addWidget(widget)

        # Corpus info
        ibox = gui.widgetBox(self.controlArea, "Corpus info", addSpace=True)
        corp_info = "Corpus of 0 documents."
        self.info_label = gui.label(ibox, self, corp_info)

        # Used Text Features
        fbox = gui.widgetBox(self.controlArea, orientation=0)
        ubox = gui.widgetBox(fbox, "Used text features", addSpace=True)
        self.used_attrs = VariableListModel(enable_dnd=True)
        self.used_attrs_view = VariablesListItemView()
        self.used_attrs_view.setModel(self.used_attrs)
        ubox.layout().addWidget(self.used_attrs_view)

        aa = self.used_attrs
        aa.dataChanged.connect(self.update_feature_selection)
        aa.rowsInserted.connect(self.update_feature_selection)
        aa.rowsRemoved.connect(self.update_feature_selection)

        # Ignored Text Features
        ibox = gui.widgetBox(fbox, "Ignored text features", addSpace=True)
        self.unused_attrs = VariableListModel(enable_dnd=True)
        self.unused_attrs_view = VariablesListItemView()
        self.unused_attrs_view.setModel(self.unused_attrs)
        ibox.layout().addWidget(self.unused_attrs_view)

        # load first file
        widget.select(0)

    def open_file(self, path):
        self.closeContext()
        self.Error.read_file.clear()
        self.used_attrs[:] = []
        self.unused_attrs[:] = []
        if path:
            try:
                self.corpus = Corpus.from_file(path)
                self.corpus.name = os.path.splitext(os.path.basename(path))[0]
                self.info_label.setText("Corpus of {} documents.".format(len(self.corpus)))
                self.used_vars = list(self.corpus.text_features)
                self.openContext(self.corpus)
                self.used_attrs.extend(self.used_vars)
                self.unused_attrs.extend([f for f in self.corpus.domain.metas
                                          if f.is_string and f not in self.used_attrs])
            except BaseException as err:
                self.Error.read_file(path, str(err))

    def update_feature_selection(self):
        # TODO fix VariablesListItemView so it does not emit
        # duplicated data when reordering inside a single window
        def remove_duplicates(l):
            unique = []
            for i in l:
                if i not in unique:
                    unique.append(i)
            return unique

        if self.corpus is not None:
            self.corpus.set_text_features(remove_duplicates(self.used_attrs))
            self.send(Output.CORPUS, self.corpus)
            self.used_vars = list(self.used_attrs)


if __name__ == '__main__':
    from AnyQt.QtWidgets import QApplication
    app = QApplication([])
    widget = OWLoadCorpus()
    widget.show()
    app.exec()
    widget.saveSettings()
