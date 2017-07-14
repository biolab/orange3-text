from PyQt4 import QtGui
from PyQt4.QtGui import QGridLayout
from PyQt4.QtGui import QLabel

from Orange.data import StringVariable
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.widget import OWWidget, Msg, Input, Output
from orangecontrib.text.tweet_profiler import TweetProfiler
from orangecontrib.text.corpus import Corpus


class OWTweetProfiler(OWWidget):
    name = "Tweet Profiler"
    description = "Detect Ekman's, Plutchik's or Profile of Mood States's " \
                  "emotions in tweets."
    icon = "icons/TweetProfiler.svg"
    priority = 46

    class Inputs:
        corpus = Input("Corpus", Corpus)

    class Outputs:
        corpus = Output("Corpus", Corpus)

    want_main_area = False
    resizing_enabled = False

    token = Setting('')
    model_name = Setting('')
    output_mode = Setting('')
    tweet_attr = Setting(0)
    auto_commit = Setting(True)

    class Error(OWWidget.Error):
        server_down = Msg('Our servers are not responding. '
                          'Please try again later.')
        invalid_token = Msg('This token is invalid')
        no_credit = Msg('Too little credits for this data set')

    def __init__(self):
        super().__init__()
        self.corpus = None
        self.strings_attrs = []
        self.profiler = TweetProfiler(
            token=self.token,
            on_server_down=self.Error.server_down,
            on_invalid_token=self.Error.invalid_token,
            on_too_little_credit=self.Error.no_credit,
        )

        # Info box
        self.n_documents = ''
        self.credit = 0
        box = gui.widgetBox(self.controlArea, "Info")
        gui.label(box, self, 'Documents: %(n_documents)s')
        gui.label(box, self, 'Credits: %(credit)s')

        # Settings
        self.controlArea.layout().addWidget(self.generate_grid_layout())

        # Server token
        box = gui.vBox(self.controlArea, 'Server Token')
        gui.lineEdit(box, self, 'token', callback=self.token_changed,
                     controlWidth=300)
        gui.button(box, self, 'Get Token', callback=self.get_new_token)

        # Auto commit
        buttons_layout = QtGui.QHBoxLayout()
        buttons_layout.addWidget(self.report_button)
        buttons_layout.addSpacing(15)
        buttons_layout.addWidget(
            gui.auto_commit(None, self, 'auto_commit', 'Commit', box=False)
        )
        self.controlArea.layout().addLayout(buttons_layout)

        self.refresh_token_info()

    def generate_grid_layout(self):
        box = QtGui.QGroupBox(title='Options')

        layout = QGridLayout()
        layout.setSpacing(10)
        row = 0

        self.tweet_attr_combo = gui.comboBox(None, self, 'tweet_attr',
                                             callback=self.apply)
        layout.addWidget(QLabel('Attribute:'))
        layout.addWidget(self.tweet_attr_combo, row, 1)

        row += 1
        self.model_name_combo = gui.comboBox(None, self, 'model_name',
                                             items=self.profiler.model_names,
                                             sendSelectedValue=True,
                                             callback=self.apply)
        if self.profiler.model_names:
            self.model_name = self.profiler.model_names[0]  # select 0th
        layout.addWidget(QLabel('Emotions:'))
        layout.addWidget(self.model_name_combo, row, 1)

        row += 1
        self.output_mode_combo = gui.comboBox(None, self, 'output_mode',
                                              items=self.profiler.output_modes,
                                              sendSelectedValue=True,
                                              callback=self.apply)
        if self.profiler.output_modes:
            self.output_mode = self.profiler.output_modes[0]    # select 0th
        layout.addWidget(QLabel('Output:'))
        layout.addWidget(self.output_mode_combo, row, 1)

        box.setLayout(layout)
        return box

    @Inputs.corpus
    def set_corpus(self, corpus):
        self.corpus = corpus

        if corpus is not None:
            self.strings_attrs = [a for a in self.corpus.domain.metas
                                  if isinstance(a, StringVariable)]
            self.tweet_attr_combo.setModel(VariableListModel(self.strings_attrs))
            self.tweet_attr_combo.currentIndexChanged.emit(self.tweet_attr)

            # select the first feature from 'text_features' if present
            ind = [self.strings_attrs.index(tf)
                   for tf in corpus.text_features
                   if tf in self.strings_attrs]
            if ind:
                self.tweet_attr = ind[0]

            self.n_documents = len(corpus)
        self.commit()

    def apply(self):
        self.commit()

    def commit(self):
        self.Error.clear()

        if self.corpus is not None:
            with self.progressBar(iterations=len(self.corpus)) as pb:
                out = self.profiler.transform(
                    self.corpus, self.strings_attrs[self.tweet_attr],
                    self.model_name, self.output_mode,
                    on_advance=pb.advance)
            self.Outputs.corpus.send(out)
        else:
            self.Outputs.corpus.send(None)

        self.refresh_token_info()

    def get_new_token(self):
        self.Warning.clear()
        self.profiler.new_token()
        self.token = self.profiler.token
        self.refresh_token_info()
        self.commit()

    def token_changed(self):
        self.profiler.token = self.token
        self.refresh_token_info()
        self.commit()

    def refresh_token_info(self):
        self.credit = str(self.profiler.get_credit())

    def send_report(self):
        self.report_items([
            ('Documents', self.n_documents),
            ('Attribute', self.strings_attrs[self.tweet_attr]
                if len(self.strings_attrs) > self.tweet_attr else ''),
            ('Emotions', self.model_name),
            ('Output', self.output_mode),
        ])


if __name__ == '__main__':
    app = QtGui.QApplication([])
    corpus = Corpus.from_file('election-tweets-2016.tab')
    widget = OWTweetProfiler()
    widget.set_corpus(corpus[:100])
    widget.show()
    app.exec()
