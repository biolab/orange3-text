import os

from Orange.data.io import FileFormat
from Orange.widgets import gui
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.data.owselectcolumns import VariablesListItemView
from Orange.widgets.settings import Setting, ContextSetting, PerfectDomainContextHandler
from Orange.widgets.widget import OWWidget, Msg, Output
from orangecontrib.text.corpus import Corpus, get_sample_corpora_dir
from orangecontrib.text.widgets.utils import widgets


class OWLoadCorpus(OWWidget):
    name = "Corpus"
    description = "Load a corpus of text documents."
    icon = "icons/TextFile.svg"
    priority = 10

    class Outputs:
        corpus = Output("Corpus", Corpus)

    want_main_area = False
    resizing_enabled = True

    dlgFormats = (
        "All readable files ({});;".format(
            '*' + ' *'.join(FileFormat.readers.keys())) +
        ";;".join("{} (*{})".format(f.DESCRIPTION, ' *'.join(f.EXTENSIONS))
                  for f in sorted(set(FileFormat.readers.values()),
                                  key=list(FileFormat.readers.values()).index)))

    settingsHandler = PerfectDomainContextHandler(
        match_values=PerfectDomainContextHandler.MATCH_VALUES_ALL
    )

    recent_files = Setting([
        "bookexcerpts.tab",
        "Grimm-tales-selected.tab",
        "Election-2016-Tweets.tab",
        "friends-transcripts.tab",
        "Andersen.tab",
    ])
    used_attrs = ContextSetting([])

    class Error(OWWidget.Error):
        read_file = Msg("Can't read file {} ({})")

    def __init__(self):
        super().__init__()

        self.corpus = None

        # Browse file box
        fbox = gui.widgetBox(self.controlArea, "Corpus file", orientation=0)
        self.file_widget = widgets.FileWidget(
            recent_files=self.recent_files, icon_size=(16, 16),
            on_open=self.open_file, dialog_format=self.dlgFormats,
            dialog_title='Open Orange Document Corpus',
            reload_label='Reload', browse_label='Browse',
            allow_empty=False, minimal_width=250,
        )
        fbox.layout().addWidget(self.file_widget)

        # Corpus info
        ibox = gui.widgetBox(self.controlArea, "Corpus info", addSpace=True)
        self.info_label = gui.label(ibox, self, "")
        self.update_info()

        # Used Text Features
        fbox = gui.widgetBox(self.controlArea, orientation=0)
        ubox = gui.widgetBox(fbox, "Used text features", addSpace=False)
        self.used_attrs_model = VariableListModel(enable_dnd=True)
        self.used_attrs_view = VariablesListItemView()
        self.used_attrs_view.setModel(self.used_attrs_model)
        ubox.layout().addWidget(self.used_attrs_view)

        aa = self.used_attrs_model
        aa.dataChanged.connect(self.update_feature_selection)
        aa.rowsInserted.connect(self.update_feature_selection)
        aa.rowsRemoved.connect(self.update_feature_selection)

        # Ignored Text Features
        ibox = gui.widgetBox(fbox, "Ignored text features", addSpace=False)
        self.unused_attrs_model = VariableListModel(enable_dnd=True)
        self.unused_attrs_view = VariablesListItemView()
        self.unused_attrs_view.setModel(self.unused_attrs_model)
        ibox.layout().addWidget(self.unused_attrs_view)

        # Documentation Data Sets & Report
        box = gui.hBox(self.controlArea)
        gui.button(box, self, "Browse documentation corpora",
                   callback=lambda: self.file_widget.browse(
                       get_sample_corpora_dir()),
                   autoDefault=False,
                   )
        box.layout().addWidget(self.report_button)

        # load first file
        self.file_widget.select(0)

    def open_file(self, path):
        self.closeContext()
        self.Error.read_file.clear()
        self.used_attrs_model[:] = []
        self.unused_attrs_model[:] = []
        if path:
            try:
                self.corpus = Corpus.from_file(path)
                self.corpus.name = os.path.splitext(os.path.basename(path))[0]
                self.update_info()
                self.used_attrs = list(self.corpus.text_features)
                self.openContext(self.corpus)
                self.used_attrs_model.extend(self.used_attrs)
                self.unused_attrs_model.extend([f for f in self.corpus.domain.metas
                                          if f.is_string and f not in self.used_attrs_model])
            except BaseException as err:
                self.Error.read_file(path, str(err))

    def update_info(self):
        def describe(corpus):
            dom = corpus.domain
            text_feats = sum(m.is_string for m in dom.metas)
            other_feats = len(dom.attributes) + len(dom.metas) - text_feats
            text = \
                "{} document(s), {} text features(s), {} other feature(s).". \
                format(len(corpus), text_feats, other_feats)
            if dom.has_continuous_class:
                text += "<br/>Regression; numerical class."
            elif dom.has_discrete_class:
                text += "<br/>Classification; discrete class with {} values.". \
                    format(len(dom.class_var.values))
            elif corpus.domain.class_vars:
                text += "<br/>Multi-target; {} target variables.".format(
                    len(corpus.domain.class_vars))
            else:
                text += "<br/>Data has no target variable."
            text += "</p>"
            return text

        if self.corpus is None:
            self.info_label.setText("No corpus loaded.")
        else:
            self.info_label.setText(describe(self.corpus))

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
            self.corpus.set_text_features(
                remove_duplicates(self.used_attrs_model))
            self.used_attrs = list(self.used_attrs_model)

            # prevent sending "empty" corpora
            dom = self.corpus.domain
            empty = not (dom.variables or dom.metas) or len(self.corpus) == 0
            self.Outputs.corpus.send(self.corpus if not empty else None)

    def send_report(self):
        def describe(features):
            if len(features):
                return ', '.join([f.name for f in features])
            else:
                return '(none)'

        if self.corpus is not None:
            domain = self.corpus.domain
            self.report_items('Corpus', (
                ("File", self.file_widget.get_selected_filename()),
                ("Documents", len(self.corpus)),
                ("Used text features", describe(self.used_attrs_model)),
                ("Ignored text features", describe(self.unused_attrs_model)),
                ('Other features', describe(domain.attributes)),
                ('Target', describe(domain.class_vars)),
            ))


if __name__ == '__main__':
    from AnyQt.QtWidgets import QApplication
    app = QApplication([])
    widget = OWLoadCorpus()
    widget.show()
    app.exec()
    widget.saveSettings()
