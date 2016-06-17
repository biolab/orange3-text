import os
from PyQt4 import QtGui, QtCore

from PyQt4.QtCore import (pyqtSignal as Signal, pyqtSlot as Slot)
from PyQt4.QtGui import (QWidget, QLabel, QHBoxLayout, QVBoxLayout,
                         QButtonGroup, QRadioButton, QSizePolicy, QFrame,
                         QApplication, QCheckBox)
from nltk.downloader import Downloader

from Orange.widgets import gui, settings, widget
from Orange.widgets.widget import OWWidget
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.widgets.utils import widgets

from orangecontrib.text import preprocess


def _i(name, icon_path='icons'):
    widget_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(widget_path, icon_path, name)


class Input:
    CORPUS = 'Corpus'


class Output:
    PP_CORPUS = 'Corpus'


class PreprocessorModule(gui.OWComponent, QWidget):
    """The base widget for the pre-processing modules."""

    change_signal = Signal()  # Emitted when the settings are changed.
    # Emitted when the module has a message to display in the main widget.
    error_signal = Signal(str)
    title = NotImplemented
    attribute = NotImplemented
    methods = NotImplemented
    single_method = True
    toggle_enabled = False
    enabled = settings.Setting(True)
    disabled_value = None

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

        self.method_layout = QtGui.QGridLayout()
        self.setup_method_layout()
        self.contents.layout().addLayout(self.method_layout)

        if self.toggle_enabled:
            self.toggle_module_switch = QCheckBox()
            switch_icon_on_resource = _i('on_button.png')
            switch_icon_off_resource = _i('off_button.png')
            style_sheet = '''
            QCheckBox::indicator {
                width: 23px;
                height: 23px;
            }
            QCheckBox::indicator:checked {
                image: url(%s);
            }
            QCheckBox::indicator:unchecked {
                image: url(%s);
            }
            ''' % (switch_icon_on_resource, switch_icon_off_resource)
            self.toggle_module_switch.setStyleSheet(style_sheet)
            self.toggle_module_switch.setChecked(self.enabled)
            self.toggle_module_switch.stateChanged.connect(self.on_toggle)
            self.titleArea.addWidget(self.toggle_module_switch)

            self.display_widget()

    @staticmethod
    def textify(text):
        return text.replace('&', '&&')

    @property
    def value(self):
        return getattr(self.preprocessor, self.attribute)

    @value.setter
    def value(self, value):
        setattr(self.preprocessor, self.attribute, value)
        self.notify_on_change()

    def setup_method_layout(self):
        raise NotImplementedError

    def notify_on_change(self):
        # Emits signals corresponding to the changes done.
        self.change_signal.emit()

    def on_toggle(self):
        # Activated when the widget is enabled/disabled.
        self.enabled = not self.enabled
        self.display_widget()
        self.change_signal.emit()

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
            self.group.addButton(rb, i)
            self.method_layout.addWidget(rb, i, 0)

        self.group.buttonClicked.connect(self.update_value)

    def get_value(self):
        self.method_index = self.group.checkedId()
        return self.methods[self.method_index]


class MultipleMethodModule(PreprocessorModule):
    toggle_enabled = True
    disabled_value = []
    checked = settings.Setting([])

    def setup_method_layout(self):
        self.methods = [method() for method in self.methods]

        self.buttons = []
        for i, method in enumerate(self.methods):
            cb = QCheckBox(self.textify(method.name))
            cb.setChecked(i in self.checked)
            cb.stateChanged.connect(self.update_value)
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

    def __init__(self, master):
        super().__init__(master)

        label = gui.label(self, self, 'Pattern:')
        line_edit = widgets.ValidatedLineEdit(self, 'pattern',
                                              validator=preprocess.RegexpTokenizer.validate_regexp)
        line_edit.editingFinished.connect(self.pattern_changed)
        self.method_layout.addWidget(label, self.REGEXP, 1)
        self.method_layout.addWidget(line_edit, self.REGEXP, 2)

    def pattern_changed(self):
        self.methods[self.REGEXP].pattern = self.pattern
        if self.REGEXP == self.method_index:
            self.notify_on_change()


class NormalizationModule(SingleMethodModule):
    attribute = 'normalizer'
    title = 'Normalization'
    toggle_enabled = True

    method_index = 0

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
        self.methods[self.SNOWBALL].language = self.snowball_language
        if self.method_index == self.SNOWBALL:
            self.notify_on_change()


class TransformationModule(MultipleMethodModule):
    attribute = 'transformers'
    title = 'Transformation'

    methods = [
        preprocess.LowercaseTransformer,
        preprocess.StripAccentsTransformer,
        preprocess.HtmlTransformer,
    ]


class FilteringModule(MultipleMethodModule):
    attribute = 'filters'
    title = 'Filtering'

    methods = [
        preprocess.StopwordsFilter,
        preprocess.LexiconFilter,
        preprocess.FrequencyFilter,
    ]
    STOPWORDS = 0
    LEXICON = 1
    FREQUENCY = 2
    dlgFormats = 'Only text files (*.txt)'

    stopwords_language = settings.Setting('english')

    recent_sw_files = settings.Setting([])
    recent_lexicon_files = settings.Setting([])

    min_df = settings.Setting(0.)
    max_df = settings.Setting(1.)
    keep_n = settings.Setting(10**5)
    use_keep_n = settings.Setting(False)

    def __init__(self, master):
        super().__init__(master)

        box = widgets.ComboBox(self, 'stopwords_language',
                               items=preprocess.StopwordsFilter.supported_languages)
        box.currentIndexChanged.connect(self.notify_on_change)
        self.method_layout.addWidget(box, self.STOPWORDS, 1)

        box = widgets.FileWidget(self.recent_sw_files,
                                 dialog_title='Open a stop words source',
                                 dialog_format=self.dlgFormats,
                                 callback=self.read_stopwords_file)
        self.method_layout.addWidget(box, self.STOPWORDS, 2, 1, 1)

        box = widgets.FileWidget(self.recent_lexicon_files,
                                 dialog_title='Open a lexicon words source',
                                 dialog_format=self.dlgFormats,
                                 callback=self.read_lexicon_file)
        self.method_layout.addWidget(box, self.LEXICON, 2, 1, 1)

        row = self.FREQUENCY
        range_widget = widgets.RangeWidget(self, ('min_df', 'max_df'),
                                           minimum=0., maximum=1., step=0.05,
                                           allow_absolute=True)
        range_widget.editingFinished.connect(self.df_changed)
        self.method_layout.addWidget(range_widget, row, 1, 1, 1)

        # row += 1
        keep_layout = QHBoxLayout()
        keep_layout.addWidget(QLabel('Keep most frequent tokens:'))
        check, spin = gui.spin(self.contents, self, 'keep_n',
                               minv=10, maxv=10**6,
                               checked='use_keep_n', checkCallback=self.keep_n_changed)
        spin.editingFinished.connect(self.keep_n_changed)
        keep_layout.addWidget(check)
        keep_layout.addWidget(spin)
        keep_layout.addStretch()
        self.method_layout.addLayout(keep_layout, row, 2, 1, 3)

    def stopwords_changed(self):
        self.methods[self.STOPWORDS].language = self.stopwords_language
        if self.STOPWORDS in self.checked:
            self.notify_on_change()

    def df_changed(self):
        self.methods[self.FREQUENCY].min_df = self.min_df
        self.methods[self.FREQUENCY].max_df = self.max_df
        if self.FREQUENCY in self.checked:
            self.notify_on_change()

    def keep_n_changed(self):
        if self.use_keep_n:
            self.methods[self.FREQUENCY].keep_n = self.keep_n
        else:
            self.methods[self.FREQUENCY].keep_n = None
        if self.FREQUENCY in self.checked:
            self.notify_on_change()

    def read_stopwords_file(self, path):
        with open(path, 'rt') as f:
            words = [line.strip() for line in f]
            self.methods[self.STOPWORDS].word_list = words
        if self.STOPWORDS in self.methods:
            self.notify_on_change()

    def read_lexicon_file(self, path):
        with open(path, 'rt') as f:
            words = [line.strip() for line in f]
            self.methods[self.LEXICON].lexicon = words
        if self.LEXICON in self.methods:
            self.notify_on_change()


class OWPreprocess(OWWidget):

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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.corpus = None
        self.preprocessor = preprocess.Preprocessor()
        self.preprocessor.on_progress = self.on_progress

        # -- INFO --
        info_box = gui.widgetBox(self.controlArea, 'Info')
        info_box.setFixedWidth(self.control_area_width)
        self.controlArea.layout().addStretch()
        self.info_label = gui.label(info_box, self,
                                    'No input corpus detected.')

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
            widget.error_signal.connect(self.display_message)

        frame_layout.addStretch()
        self.mainArea.layout().addWidget(frame)

        # Buttons area
        self.report_button.setFixedWidth(self.control_area_width)

        commit_button = gui.auto_commit(self.buttonsArea, self, 'autocommit',
                                        'Commit', box=False)
        commit_button.setFixedWidth(self.control_area_width - 5)

        self.buttonsArea.layout().addWidget(commit_button)
        self.progress_bar = None  # Progress bar initialization.

    def set_data(self, data=None):
        self.corpus = data
        self.update_info()
        self.commit()

    def update_info(self):
        if self.corpus is not None:
            info = 'Document count: {}'.format(len(self.corpus))
        else:
            info = 'No input corpus detected.'
        self.info_label.setText(info)

    def commit(self):
        if self.corpus is not None:
            self.apply()

    def apply(self):
        self.progressBarInit()
        output = self.preprocessor(self.corpus)
        self.progressBarFinished()
        self.send(Output.PP_CORPUS, output)

    def on_progress(self, progress):
        self.progressBarSet(progress)

    @Slot()
    def settings_invalidated(self):
        self.commit()

    @Slot(str)
    def display_message(self, message):
        self.error(0, message)

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
