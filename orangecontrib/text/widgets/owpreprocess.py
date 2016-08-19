import os

from PyQt4 import QtGui, QtCore

from PyQt4.QtCore import (pyqtSignal as Signal, pyqtSlot as Slot)
from PyQt4.QtGui import (QWidget, QLabel, QHBoxLayout, QVBoxLayout,
                         QButtonGroup, QRadioButton, QSizePolicy, QFrame,
                         QApplication, QCheckBox)
from nltk.downloader import Downloader

from Orange.widgets import gui, settings, widget
from Orange.widgets.widget import OWWidget, Msg
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.widgets.utils import widgets

from orangecontrib.text import preprocess
from orangecontrib.text.widgets.utils.concurrent import asynchronous, OWConcurrentWidget


def _i(name, icon_path='icons'):
    widget_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(widget_path, icon_path, name)


class Input:
    CORPUS = 'Corpus'


class Output:
    PP_CORPUS = 'Corpus'


class OnOffButton(QtGui.QPushButton):
    stateChanged = QtCore.pyqtSignal()

    def __init__(self, enabled=True, on_button='on_button.png', off_button='off_button.png', size=(26, 26), *__args):
        super().__init__(*__args)
        self.on_icon = QtGui.QIcon(_i(on_button))
        self.off_icon = QtGui.QIcon(_i(off_button))
        self.setAutoDefault(False)      # do not toggle on Enter
        self.setFlat(True)
        self.setIconSize(QtCore.QSize(*size))
        self.setStyleSheet('border:none')
        self.state = enabled
        self.clicked.connect(self.change_state)
        self.update_icon()

    def change_state(self):
        self.state = not self.state
        self.update_icon()
        self.stateChanged.emit()

    def update_icon(self):
        self.setIcon(self.on_icon if self.state else self.off_icon)

    def sizeHint(self):
        return QtCore.QSize(26, 26)


class PreprocessorModule(gui.OWComponent, QWidget):
    """The base widget for the pre-processing modules."""

    change_signal = Signal()  # Emitted when the settings are changed.
    title = NotImplemented
    attribute = NotImplemented
    methods = NotImplemented
    single_method = True
    toggle_enabled = True
    enabled = settings.Setting(True)
    disabled_value = None
    Layout = QtGui.QGridLayout

    def __init__(self, master):
        QWidget.__init__(self)
        gui.OWComponent.__init__(self, master)
        self.master = master
        self.preprocessor = master.preprocessor
        self.value = getattr(self.preprocessor, self.attribute)

        # Title bar.
        title_holder = QWidget()
        title_holder.setSizePolicy(QSizePolicy.MinimumExpanding,
                                   QSizePolicy.Fixed)
        title_holder.setStyleSheet("""
        .QWidget {
        background: qlineargradient( x1:0 y1:0, x2:0 y2:1,
        stop:0 #F8F8F8, stop:1 #C8C8C8);
        border-bottom: 1px solid #B3B3B3;
        }
        """)
        self.titleArea = QHBoxLayout()
        self.titleArea.setContentsMargins(15, 10, 15, 10)
        self.titleArea.setSpacing(0)
        title_holder.setLayout(self.titleArea)

        self.title_label = QLabel(self.title)
        self.title_label.setStyleSheet('font-size: 12px;')
        self.titleArea.addWidget(self.title_label)

        self.off_label = QLabel('[disabled]')
        self.off_label.setStyleSheet('color: #B0B0B0; margin-left: 5px;')
        self.titleArea.addWidget(self.off_label)
        self.off_label.hide()

        self.titleArea.addStretch()

        # Root.
        self.rootArea = QVBoxLayout()
        self.rootArea.setContentsMargins(0, 0, 0, 0)
        self.rootArea.setSpacing(0)
        self.setLayout(self.rootArea)
        self.rootArea.addWidget(title_holder)

        self.contents = QWidget()
        contentArea = QVBoxLayout()
        contentArea.setContentsMargins(15, 10, 15, 10)
        self.contents.setLayout(contentArea)
        self.rootArea.addWidget(self.contents)

        self.method_layout = self.Layout()
        self.setup_method_layout()
        self.contents.layout().addLayout(self.method_layout)

        if self.toggle_enabled:
            self.on_off_button = OnOffButton(enabled=self.enabled)
            self.on_off_button.stateChanged.connect(self.on_toggle)
            self.on_off_button.setContentsMargins(0, 0, 0, 0)
            self.titleArea.addWidget(self.on_off_button)
            self.display_widget()

    @staticmethod
    def get_tooltip(method):
        return method.__doc__.strip().strip('.') if method.__doc__ else None

    @staticmethod
    def textify(text):
        return text.replace('&', '&&')

    @property
    def value(self):
        return getattr(self.preprocessor, self.attribute)

    @value.setter
    def value(self, value):
        setattr(self.preprocessor, self.attribute, value)
        self.change_signal.emit()

    def setup_method_layout(self):
        raise NotImplementedError

    def on_toggle(self):
        # Activated when the widget is enabled/disabled.
        self.enabled = not self.enabled
        self.display_widget()

    def display_widget(self):
        if self.enabled:
            self.value = self.get_value()
            self.off_label.hide()
            self.contents.show()
            self.title_label.setStyleSheet('color: #000000;')
        else:
            self.value = self.disabled_value
            self.off_label.show()
            self.contents.hide()
            self.title_label.setStyleSheet('color: #B0B0B0;')

    def get_value(self):
        raise NotImplemented

    def update_value(self):
        self.value = self.get_value()


class SingleMethodModule(PreprocessorModule):
    method_index = settings.Setting(0)

    def setup_method_layout(self):
        self.group = QButtonGroup(self, exclusive=True)
        self.methods = [method() for method in self.methods]

        for i, method in enumerate(self.methods):
            rb = QRadioButton(self, text=self.textify(method.name))
            rb.setChecked(i == self.method_index)
            rb.setToolTip(self.get_tooltip(method))
            self.group.addButton(rb, i)
            self.method_layout.addWidget(rb, i, 0)

        self.group.buttonClicked.connect(self.update_value)

    def get_value(self):
        self.method_index = self.group.checkedId()
        return self.methods[self.method_index]


class MultipleMethodModule(PreprocessorModule):
    disabled_value = []
    checked = settings.Setting([])

    def setup_method_layout(self):
        self.methods = [method() for method in self.methods]

        self.buttons = []
        for i, method in enumerate(self.methods):
            cb = QCheckBox(self.textify(method.name))
            cb.setChecked(i in self.checked)
            cb.stateChanged.connect(self.update_value)
            cb.setToolTip(self.get_tooltip(method))
            self.method_layout.addWidget(cb)
            self.buttons.append(cb)

    def get_value(self):
        self.checked = [i for i, button in enumerate(self.buttons) if button.isChecked()]
        return [self.methods[i] for i in self.checked]


class TokenizerModule(SingleMethodModule):
    attribute = 'tokenizer'
    title = 'Tokenization'
    toggle_enabled = True

    methods = [
        preprocess.tokenize.WordPunctTokenizer,
        preprocess.tokenize.WhitespaceTokenizer,
        preprocess.tokenize.PunktSentenceTokenizer,
        preprocess.tokenize.RegexpTokenizer,
        preprocess.tokenize.TweetTokenizer,
    ]
    REGEXP = 3
    pattern = settings.Setting('\w+')

    method_index = settings.Setting(REGEXP)

    def __init__(self, master):
        super().__init__(master)

        label = gui.label(self, self, 'Pattern:')
        line_edit = widgets.ValidatedLineEdit(self, 'pattern',
                                              validator=preprocess.RegexpTokenizer.validate_regexp)
        line_edit.editingFinished.connect(self.pattern_changed)
        self.method_layout.addWidget(label, self.REGEXP, 1)
        self.method_layout.addWidget(line_edit, self.REGEXP, 2)
        self.pattern_changed()

    def pattern_changed(self):
        if self.methods[self.REGEXP].pattern != self.pattern:
            self.methods[self.REGEXP].pattern = self.pattern

            if self.REGEXP == self.method_index:
                self.change_signal.emit()


class NormalizationModule(SingleMethodModule):
    attribute = 'normalizer'
    title = 'Normalization'
    toggle_enabled = True
    enabled = settings.Setting(False)

    methods = [
        preprocess.PorterStemmer,
        preprocess.SnowballStemmer,
        preprocess.WordNetLemmatizer,
    ]
    SNOWBALL = 1

    snowball_language = settings.Setting('english')

    def __init__(self, master):
        super().__init__(master)

        label = gui.label(self, self, 'Language:')
        label.setAlignment(QtCore.Qt.AlignRight)
        self.method_layout.addWidget(label, self.SNOWBALL, 1)
        box = widgets.ComboBox(self, 'snowball_language',
                               items=preprocess.SnowballStemmer.supported_languages)
        box.currentIndexChanged.connect(self.change_language)
        self.method_layout.addWidget(box, self.SNOWBALL, 2)

    def change_language(self):
        if self.methods[self.SNOWBALL].language != self.snowball_language:
            self.methods[self.SNOWBALL].language = self.snowball_language

            if self.method_index == self.SNOWBALL:
                self.change_signal.emit()


class TransformationModule(MultipleMethodModule):
    attribute = 'transformers'
    title = 'Transformation'

    methods = [
        preprocess.LowercaseTransformer,
        preprocess.StripAccentsTransformer,
        preprocess.HtmlTransformer,
        preprocess.UrlRemover,
    ]
    checked = settings.Setting([0])
    Layout = QHBoxLayout


class DummyKeepN:
    """ Keeps top N tokens by document frequency. """
    name = 'Most frequent tokens.'


class FilteringModule(MultipleMethodModule):
    attribute = 'filters'
    title = 'Filtering'

    methods = [
        preprocess.StopwordsFilter,
        preprocess.LexiconFilter,
        preprocess.RegexpFilter,
        preprocess.FrequencyFilter,
        DummyKeepN,
    ]
    checked = settings.Setting([0])
    STOPWORDS = 0
    LEXICON = 1
    REGEXP = 2
    FREQUENCY = 3
    KEEP_N = 4
    dlgFormats = 'Only text files (*.txt)'

    stopwords_language = settings.Setting('english')

    recent_sw_files = settings.Setting([])
    recent_lexicon_files = settings.Setting([])

    pattern = settings.Setting('\.|,|:|!|\?')
    min_df = settings.Setting(0.)
    max_df = settings.Setting(1.)
    keep_n = settings.Setting(10**5)
    use_keep_n = settings.Setting(False)
    use_df = settings.Setting(False)

    def __init__(self, master):
        super().__init__(master)

        box = widgets.ComboBox(self, 'stopwords_language',
                               items=[None] + preprocess.StopwordsFilter.supported_languages)
        box.currentIndexChanged.connect(self.stopwords_changed)
        self.stopwords_changed()
        self.method_layout.addWidget(box, self.STOPWORDS, 1)

        box = widgets.FileWidget(recent_files=self.recent_sw_files,
                                 dialog_title='Open a stop words source',
                                 dialog_format=self.dlgFormats,
                                 on_open=self.read_stopwords_file,
                                 browse_label='',
                                 reload_label='')
        box.select(0)
        self.method_layout.addWidget(box, self.STOPWORDS, 2, 1, 1)

        box = widgets.FileWidget(recent_files=self.recent_lexicon_files,
                                 dialog_title='Open a lexicon words source',
                                 dialog_format=self.dlgFormats,
                                 on_open=self.read_lexicon_file,
                                 browse_label='',
                                 reload_label='')
        box.select(0)
        self.method_layout.addWidget(box, self.LEXICON, 2, 1, 1)

        pattern_edit = widgets.ValidatedLineEdit(self, 'pattern',
                                              validator=preprocess.RegexpFilter.validate_regexp)
        pattern_edit.editingFinished.connect(self.pattern_changed)
        self.method_layout.addWidget(pattern_edit, self.REGEXP, 1, 1, 2)

        range_widget = widgets.RangeWidget(None, self, ('min_df', 'max_df'),
                                           minimum=0., maximum=1., step=0.05,
                                           allow_absolute=True)
        range_widget.setToolTip(self.get_tooltip(preprocess.FrequencyFilter))
        range_widget.editingFinished.connect(self.df_changed)
        self.method_layout.addWidget(range_widget, self.FREQUENCY, 1, 1, 1)

        spin = gui.spin(self.contents, self, 'keep_n', box=False, minv=1, maxv=10**6)
        spin.editingFinished.connect(self.keep_n_changed)
        self.method_layout.addWidget(spin, self.KEEP_N, 1, 1, 1)

    @property
    def frequency_filter(self):
        return self.methods[self.FREQUENCY]

    def pattern_changed(self):
        if self.methods[self.REGEXP].pattern != self.pattern:
            self.methods[self.REGEXP].pattern = self.pattern

            if self.REGEXP in self.checked:
                self.change_signal.emit()

    def get_value(self):
        self.checked = [i for i, button in enumerate(self.buttons) if button.isChecked()]
        checked = self.checked[:]
        if self.FREQUENCY in self.checked:
            self.frequency_filter.min_df = self.min_df
            self.frequency_filter.max_df = self.max_df
        else:
            checked.append(self.FREQUENCY)
            self.frequency_filter.min_df = 0.
            self.frequency_filter.max_df = 1.

        if self.KEEP_N in checked:
            checked.pop(checked.index(self.KEEP_N))
            self.frequency_filter.keep_n = self.keep_n
        else:
            self.frequency_filter.keep_n = None

        return [self.methods[i] for i in checked]

    def stopwords_changed(self):
        if self.methods[self.STOPWORDS].language != self.stopwords_language:
            self.methods[self.STOPWORDS].language = self.stopwords_language
            if self.STOPWORDS in self.checked:
                self.change_signal.emit()

    def df_changed(self):
        if self.FREQUENCY in self.checked:
            self.frequency_filter.min_df = self.min_df
            self.frequency_filter.max_df = self.max_df
            self.change_signal.emit()

    def keep_n_changed(self):
        if self.KEEP_N in self.checked:
            self.frequency_filter.keep_n = self.keep_n
            self.change_signal.emit()

    def read_stopwords_file(self, path):
        self.methods[self.STOPWORDS].from_file(path)
        if self.STOPWORDS in self.checked:
            self.change_signal.emit()

    def read_lexicon_file(self, path):
        self.methods[self.LEXICON].from_file(path)
        if self.LEXICON in self.checked:
            self.change_signal.emit()


class OWPreprocess(OWConcurrentWidget):

    name = 'Preprocess Text'
    description = 'Construct a text pre-processing pipeline.'
    icon = 'icons/TextPreprocess.svg'
    priority = 30

    inputs = [(Input.CORPUS, Corpus, 'set_data')]
    outputs = [(Output.PP_CORPUS, Corpus)]

    autocommit = settings.Setting(True)

    preprocessors = [
        TransformationModule,
        TokenizerModule,
        NormalizationModule,
        FilteringModule,
    ]

    transformation = settings.SettingProvider(TransformationModule)
    tokenization = settings.SettingProvider(TokenizerModule)
    normalization = settings.SettingProvider(NormalizationModule)
    filtering = settings.SettingProvider(FilteringModule)

    control_area_width = 250
    buttons_area_orientation = QtCore.Qt.Vertical

    UserAdviceMessages = [
        widget.Message(
            "Some preprocessing methods require data (like word relationships, stop words, "
            "punctuation rules etc.) from the NLTK package. This data, if you didn't have it "
            "already, was downloaded to: {}".format(Downloader().default_download_dir()),
            "nltk_data")]

    class Warning(OWWidget.Warning):
        no_token_left = Msg('No tokens on output! Please, change configuration.')

    def __init__(self, parent=None):
        super().__init__(parent)
        self.corpus = None
        self.preprocessor = preprocess.Preprocessor()

        # -- INFO --
        info_box = gui.widgetBox(self.controlArea, 'Info')
        info_box.setFixedWidth(self.control_area_width)
        self.controlArea.layout().addStretch()
        self.info_label = gui.label(info_box, self, '')
        self.update_info()

        # -- PIPELINE --
        frame = QFrame()
        frame.setContentsMargins(0, 0, 0, 0)
        frame.setFrameStyle(QFrame.Box)
        frame.setStyleSheet('.QFrame { border: 1px solid #B3B3B3; }')
        frame_layout = QVBoxLayout()
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(0)
        frame.setLayout(frame_layout)

        for stage in self.preprocessors:
            widget = stage(self)
            setattr(self, stage.title.lower(), widget)
            frame_layout.addWidget(widget)
            widget.change_signal.connect(self.settings_invalidated)

        frame_layout.addStretch()
        scroll = QtGui.QScrollArea()
        scroll.setWidget(frame)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll.resize(frame_layout.sizeHint())
        scroll.setMinimumWidth(frame_layout.sizeHint().width() + 20)  # + scroll bar
        scroll.setMinimumHeight(500)
        self.mainArea.layout().addWidget(scroll)

        # Buttons area
        self.report_button.setFixedWidth(self.control_area_width)

        commit_button = gui.auto_commit(self.buttonsArea, self, 'autocommit',
                                        'Commit', box=False)
        commit_button.setFixedWidth(self.control_area_width - 5)

        self.buttonsArea.layout().addWidget(commit_button)

    def set_data(self, data=None):
        self.corpus = data.copy() if data else None
        self.commit()

    def update_info(self, corpus=None):
        if corpus is not None:
            info = 'Document count: {}\n' \
                   'Total tokens: {}\n'\
                   'Unique tokens: {}'\
                   .format(len(corpus), sum(map(len, corpus.tokens)), len(corpus.dictionary))
        else:
            info = 'No output corpus.'
        self.info_label.setText(info)

    def commit(self):
        self.Warning.no_token_left.clear()
        if self.corpus is not None:
            self.apply()

    def apply(self):
        self.preprocess()

    @asynchronous
    def preprocess(self, **kwargs):
        return self.preprocessor(self.corpus, inplace=True, **kwargs)

    def on_result(self, result):
        self.update_info(result)
        if result is not None and len(result.dictionary) == 0:
            self.Warning.no_token_left()
            result = None
        self.send(Output.PP_CORPUS, result)

    @Slot()
    def settings_invalidated(self):
        self.commit()

    def send_report(self):
        self.report_items('Preprocessor', self.preprocessor.report())


if __name__ == '__main__':
    app = QApplication([])
    widget = OWPreprocess()
    widget.show()
    corpus = Corpus.from_file('bookexcerpts')
    widget.set_data(corpus)
    app.exec()
    widget.saveSettings()
