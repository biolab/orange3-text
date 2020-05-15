import numpy as np

from AnyQt.QtWidgets import QApplication, QGridLayout, QLabel, QGroupBox, \
    QHBoxLayout, QPushButton, QStyle

from Orange.data import StringVariable
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin
from Orange.widgets.widget import OWWidget, Msg, Input, Output
from orangecontrib.text.tweet_profiler import TweetProfiler
from orangecontrib.text.corpus import Corpus


def run_profiler(profiler, corpus, meta_var, model_name, output_mode, state):

    ticks = iter(np.linspace(0., 100.,
                             int(np.ceil(len(corpus) / profiler.BATCH_SIZE) + 1)))

    def advance():
        if state.is_interruption_requested():
            raise InterruptedError
        state.set_progress_value(next(ticks))

    out = profiler.transform(corpus, meta_var, model_name, output_mode,
                             on_advance=advance)
    return out


class OWTweetProfiler(OWWidget, ConcurrentWidgetMixin):
    name = "Tweet Profiler"
    description = "Detect Ekman's, Plutchik's or Profile of Mood States's " \
                  "emotions in tweets."
    icon = "icons/TweetProfiler.svg"
    priority = 330
    keywords = ["Twitter"]

    class Inputs:
        corpus = Input("Corpus", Corpus)

    class Outputs:
        corpus = Output("Corpus", Corpus)

    want_main_area = False
    resizing_enabled = False

    model_name = Setting('')
    output_mode = Setting('')
    tweet_attr = Setting(0)
    auto_commit = Setting(True)

    class Error(OWWidget.Error):
        server_down = Msg('Our servers are not responding. '
                          'Please try again later.')
        unexpected_error = Msg('Unknown error: {}')

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        self.corpus = None
        self.last_config = None     # to avoid reruns with the same params
        self.strings_attrs = []
        self.profiler = TweetProfiler(on_server_down=self.Error.server_down)

        # Settings
        self.controlArea.layout().addWidget(self.generate_grid_layout())

        # Auto commit
        buttons_layout = QHBoxLayout()
        buttons_layout.addSpacing(15)
        buttons_layout.addWidget(
            gui.auto_commit(None, self, 'auto_commit', 'Commit', box=False)
        )
        self.controlArea.layout().addLayout(buttons_layout)

        self.cancel_button = QPushButton(
            'Cancel',
            icon=self.style()
            .standardIcon(QStyle.SP_DialogCancelButton))

        self.cancel_button.clicked.connect(self.cancel)

        hbox = gui.hBox(self.controlArea)
        hbox.layout().addWidget(self.cancel_button)
        self.cancel_button.setDisabled(True)

    def generate_grid_layout(self):
        box = QGroupBox(title='Options')

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
        self.cancel()
        self.corpus = corpus
        self.last_config = None

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

        self.commit()

    def apply(self):
        self.commit()

    def _get_config(self):
        return self.tweet_attr, self.model_name, self.output_mode

    def commit(self):
        self.Error.clear()

        if self.last_config == self._get_config():
            return

        if self.corpus is not None:
            self.cancel_button.setDisabled(False)
            self.start(run_profiler, self.profiler, self.corpus,
                       self.strings_attrs[self.tweet_attr],
                       self.model_name, self.output_mode)
        else:
            self.Outputs.corpus.send(None)

    def on_done(self, result):
        self.cancel_button.setDisabled(True)
        self.last_config = self._get_config()
        self.Outputs.corpus.send(result)

    def on_partial_result(self, result):
        self.cancel()

    def on_exception(self, ex):
        self.Error.unexpected_error(type(ex).__name__)
        self.cancel()

    def cancel(self):
        self.cancel_button.setDisabled(True)
        super().cancel()

    def send_report(self):
        self.report_items([
            ('Attribute', self.strings_attrs[self.tweet_attr]
             if len(self.strings_attrs) > self.tweet_attr else ''),
            ('Emotions', self.model_name),
            ('Output', self.output_mode),
        ])


if __name__ == '__main__':
    app = QApplication([])
    corpus = Corpus.from_file('election-tweets-2016.tab')
    widget = OWTweetProfiler()
    widget.set_corpus(corpus[:100])
    widget.show()
    app.exec()
