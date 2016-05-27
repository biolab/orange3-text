import os

from PyQt4.QtCore import (pyqtSignal as Signal, pyqtSlot as Slot)
from PyQt4.QtGui import (QWidget, QLabel, QHBoxLayout, QVBoxLayout,
                         QButtonGroup, QRadioButton, QSizePolicy, QFrame,
                         QComboBox, QPushButton, QStyle, QApplication,
                         QFileDialog, QLineEdit, QCheckBox)
from nltk.corpus import stopwords

from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.preprocess import (PorterStemmer as PS,
                                           Lemmatizer as LM,
                                           SnowballStemmer as SS)
from orangecontrib.text.preprocess import Preprocessor


def _i(name, icon_path='icons'):
    widget_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(widget_path, icon_path, name)


class Input:
    CORPUS = 'Corpus'


class Output:
    PP_CORPUS = 'Corpus'


class PreprocessorModule(QWidget):
    """The base widget for the pre-processing modules."""

    change_signal = Signal()  # Emitted when the settings are changed.
    # Emitted when the module has a message to display in the main widget.
    error_signal = Signal(str)
    enabled = False  # If the module is enabled.

    def __init__(self, title, toggle_enabled, is_enabled):
        super().__init__()

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

        self.title_label = QLabel(title)
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

        self.enabled = is_enabled
        if toggle_enabled:
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

            # Change the view according to the flag.
            if self.enabled:
                self.off_label.hide()
                self.contents.show()
                self.title_label.setStyleSheet('color: #000000;')
            else:
                self.off_label.show()
                self.contents.hide()
                self.title_label.setStyleSheet('color: #B0B0B0;')

            self.titleArea.addWidget(self.toggle_module_switch)

    def add_to_content_area(self, new_widget):
        self.contents.layout().addWidget(new_widget)

    def add_layout_to_content_area(self, new_layout):
        self.contents.layout().addLayout(new_layout)

    def notify_on_change(self):
        # Emits signals corresponding to the changes done.
        self.change_signal.emit()

    def on_toggle(self):
        # Activated when the widget is enabled/disabled.
        self.enabled = not self.enabled
        if self.enabled:
            self.off_label.hide()
            self.contents.show()
            self.title_label.setStyleSheet('color: #000000;')
        else:
            self.off_label.show()
            self.contents.hide()
            self.title_label.setStyleSheet('color: #B0B0B0;')
        self.change_signal.emit()

    def restore_data(self, data):
        # Restores the widget state from the input data.
        raise NotImplementedError

    def export_data(self):
        # Export the settings for this module instance.
        return NotImplementedError

    @staticmethod
    def get_pp_settings():
        # Returns the dict representation of this portion of a pre-processor.
        return NotImplementedError


class TokenizerModule(PreprocessorModule):
    DEFAULT_SETTINGS = {
        'is_enabled': True,
        'method': 0,
    }

    NLTKTokenizer, TwitterTokenizer = 0, 1
    tokenizer_values = {
        NLTKTokenizer: False,
        TwitterTokenizer: True
    }
    tokenizer_names = {
        NLTKTokenizer: 'NLTK tokenizer',
        TwitterTokenizer: 'Twitter tokenizer',
    }

    tokenizer_method = 0

    def __init__(self, data):
        data = data or self.DEFAULT_SETTINGS
        PreprocessorModule.__init__(
                self, 'Tokenizer', False,
                data.get('is_enabled')
        )

        self.group = QButtonGroup(self, exclusive=True)
        for method in [self.NLTKTokenizer, self.TwitterTokenizer]:
            rb = QRadioButton(self, text=self.tokenizer_names[method])
            if method == self.TwitterTokenizer:
                # Disable until the Twitter tokenizer is available.
                rb.setEnabled(False)
            self.add_to_content_area(rb)
            self.group.addButton(rb, method)
        self.group.buttonClicked.connect(self.group_button_clicked)

        # Restore the previous state, after starting off the layout.
        self.restore_data(data)

    def group_button_clicked(self):
        self.tokenizer_method = self.group.checkedId()
        self.notify_on_change()

    def restore_data(self, data):
        self.tokenizer_method = data.get('method')
        b = self.group.button(self.tokenizer_method)
        b.setChecked(True)

    def export_data(self):
        return {
            'is_enabled': self.enabled,
            'method': self.tokenizer_method,
        }

    def get_pp_setting(self):
        return {
            'use_twitter_tokenizer': self.tokenizer_values.get(
                    self.tokenizer_method)
        }


class TransformationModule(PreprocessorModule):
    DEFAULT_SETTINGS = {
        'is_enabled': True,
        'method': 0,
    }

    PorterStemmer, SnowballStemmer, Lemmatizer = 0, 1, 2
    transformation_values = {
        PorterStemmer: PS,
        SnowballStemmer: SS,
        Lemmatizer: LM,
    }

    transformation_method = 0

    def __init__(self, data):
        data = data or self.DEFAULT_SETTINGS
        PreprocessorModule.__init__(
                self, 'Stemming', True,
                data.get('is_enabled')
        )

        self.group = QButtonGroup(self, exclusive=True)
        for method in [
            self.PorterStemmer,
            self.SnowballStemmer,
            self.Lemmatizer
        ]:
            rb = QRadioButton(
                    self,
                    text=self.transformation_values[method].name
            )
            self.add_to_content_area(rb)
            self.group.addButton(rb, method)
        self.group.buttonClicked.connect(self.group_button_clicked)

        # Restore the previous state, after starting off the layout.
        self.restore_data(data)

    def group_button_clicked(self):
        self.transformation_method = self.group.checkedId()
        self.notify_on_change()

    def restore_data(self, data):
        self.transformation_method = data.get('method')
        b = self.group.button(self.transformation_method)
        b.setChecked(True)

    def export_data(self):
        return {
            'is_enabled': self.enabled,
            'method': self.transformation_method,
        }

    def get_pp_setting(self):
        return {
            'transformation': self.transformation_values.get(
                    self.transformation_method
            )
        }


class CasingModule(PreprocessorModule):
    DEFAULT_SETTINGS = {
        'is_enabled': True,
    }

    def __init__(self, data):
        data = data or self.DEFAULT_SETTINGS
        PreprocessorModule.__init__(
                self, 'Case folding', True,
                data.get('is_enabled')
        )

        self.add_to_content_area(QLabel('All tokens will be lower-cased.'))

    def restore_data(self, data):
        pass  # Nothing to restore here.

    def export_data(self):
        return {
            'is_enabled': self.enabled,
        }

    def get_pp_setting(self):
        return {
            'lowercase': self.enabled,
        }


class FilteringModule(PreprocessorModule):
    DEFAULT_SETTINGS = {
        'is_enabled': True,
        'methods': [True, False, False],
        'recent_sw_files': [],
        'min_df': None,
        'max_df': None,
    }

    English, Custom, DocumentFrequency = 0, 1, 2
    filtering_values = {
        English: 'english',
        Custom: [],
        DocumentFrequency: (None, None),
    }
    filter_names = {
        English: 'English stop words',
        Custom: 'Custom stop words',
        DocumentFrequency: 'Filter by token frequency',
    }

    filtering_methods = [True, False, False]

    dlgFormats = 'Only text files (*.txt)'
    recent_sw_files = []

    def __init__(self, data):
        data = data or self.DEFAULT_SETTINGS
        PreprocessorModule.__init__(
                self, 'Token filtering', True,
                data.get('is_enabled')
        )

        self.group = QButtonGroup(self, exclusive=False)

        # --- English ---
        cb = QCheckBox(self, text=self.filter_names[self.English])
        self.add_to_content_area(cb)
        self.group.addButton(cb, self.English)

        # --- Custom ---
        cb = QCheckBox(self, text=self.filter_names[self.Custom])
        self.add_to_content_area(cb)
        self.group.addButton(cb, self.Custom)

        # File browser.
        file_browser_layout = QHBoxLayout()
        file_browser_layout.setContentsMargins(20, 0, 0, 0)
        self.sw_file_combo = QComboBox()
        self.sw_file_combo.setMinimumWidth(200)
        file_browser_layout.addWidget(self.sw_file_combo)
        self.sw_file_combo.activated[int].connect(self.select_file)

        self.browse_button = QPushButton(self)
        self.browse_button.clicked.connect(self.browse_file)
        self.browse_button.setIcon(self.style()
                                   .standardIcon(QStyle.SP_DirOpenIcon))
        self.browse_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        file_browser_layout.addWidget(self.browse_button)

        # Reload button
        self.reload_button = QPushButton(self)
        self.reload_button.clicked.connect(self.on_reload_button_clicked)
        self.reload_button.setIcon(self.style()
                                   .standardIcon(QStyle.SP_BrowserReload))
        self.reload_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        file_browser_layout.addWidget(self.reload_button)

        self.add_layout_to_content_area(file_browser_layout)

        # --- DF ---
        df_info_text = """
        Remove all tokens that appear in less than 'min-df' documents.
        Remove all tokens that appear in more than 'max-df' documents.
        Values can be either integers or floats (ratio of documents).
        """
        cb = QCheckBox(self, text=self.filter_names[self.DocumentFrequency])
        self.add_to_content_area(cb)
        self.group.addButton(cb, self.DocumentFrequency)
        df_info_text = QLabel(df_info_text)
        df_info_text.setContentsMargins(0,0,0,0)
        df_info_text.setStyleSheet("""
        font-size: 11px;
        font-style: italic;
        """)
        self.add_to_content_area(df_info_text)
        # Min/Max-Df setter.
        df_setter_layout = QHBoxLayout()
        df_setter_layout.setContentsMargins(20, 0, 0, 0)
        self.min_df_input = QLineEdit()
        self.min_df_input.textChanged.connect(self.update_df_parameters)
        self.max_df_input = QLineEdit()
        self.max_df_input.textChanged.connect(self.update_df_parameters)
        df_setter_layout.addWidget(QLabel('Min-df:'))
        df_setter_layout.addWidget(self.min_df_input)
        df_setter_layout.addWidget(QLabel('Max-df:'))
        df_setter_layout.addWidget(self.max_df_input)

        self.add_layout_to_content_area(df_setter_layout)
        self.group.buttonClicked.connect(self.group_button_clicked)

        # Restore the widget to its previous state.
        self.restore_data(data)

    def str_to_num(self, s):
        if not s:
            return None

        try:
            return int(s)
        except ValueError:
            pass  # Not an int. Continue.
        try:
            return float(s)
        except ValueError:  # Not a float either.
            self.send_message('Input "{}" cannot be cast into a number.'
                              .format(s))
            return None

    def send_message(self, message):
        # Sends a message with the "message" signal, to the main widget.
        self.error_signal.emit(message)

    # --- File selection.
    def select_file(self, n):
        if n < len(self.recent_sw_files):
            name = self.recent_sw_files[n]
            del self.recent_sw_files[n]
            self.recent_sw_files.insert(0, name)

        if len(self.recent_sw_files) > 0:
            self.set_file_list()
            self.open_file(self.recent_sw_files[0])

    def set_file_list(self):
        self.sw_file_combo.clear()
        if not self.recent_sw_files:
            self.sw_file_combo.addItem('(none)')
        else:
            for file in self.recent_sw_files:
                self.sw_file_combo.addItem(os.path.split(file)[1])

    def browse_file(self):
        # Opens the file browser, starting at the home directory.
        start_file = os.path.expanduser('~/')
        # Get the file path from the browser window.
        path = QFileDialog.getOpenFileName(self, 'Open a stop words source',
                                           start_file, self.dlgFormats)
        if not path:
            return

        if path in self.recent_sw_files:
            self.recent_sw_files.remove(path)
        self.recent_sw_files.insert(0, path)
        self.set_file_list()
        self.open_file(path)

    def update_df_parameters(self):
        min_df = None if not self.min_df_input.text() else self.min_df_input.text()
        max_df = None if not self.max_df_input.text() else self.max_df_input.text()
        self.filtering_values[self.DocumentFrequency] = (min_df, max_df)
        self.notify_on_change()

    def open_file(self, path):
        try:
            with open(path) as f:  # Read most recent.
                self.filtering_values[self.Custom] = [sw.strip() for sw in
                                                      f.read().splitlines()]
                self.notify_on_change()
        except Exception:  # Raise an exception otherwise.
            self.send_message('Could not open "{}".'
                              .format(path))

    def on_reload_button_clicked(self):
        if self.recent_sw_files:
            self.select_file(0)
    # END File selection.

    def group_button_clicked(self):
        self.filtering_methods = [ch_box.isChecked() for ch_box in
                                  self.group.buttons()]

        self.enable_choice_settings()

        # Emit the signal.
        self.notify_on_change()

    def enable_choice_settings(self):
        self.sw_file_combo.setEnabled(self.filtering_methods[1])
        self.browse_button.setEnabled(self.filtering_methods[1])
        self.reload_button.setEnabled(self.filtering_methods[1])

        self.min_df_input.setEnabled(self.filtering_methods[2])
        self.max_df_input.setEnabled(self.filtering_methods[2])

    def get_pp_setting(self):
        flag_english = self.filtering_methods[0]
        flag_custom = self.filtering_methods[1]
        flag_df = self.filtering_methods[2]
        if flag_english and flag_custom:  # Use custom.
            stop_words = {
                'stop_words': stopwords.words('english') +
                              self.filtering_values[self.Custom]
            }
        elif flag_english and not flag_custom:
            stop_words = {
                'stop_words': 'english'
            }
        elif flag_custom:
            stop_words = {
                'stop_words': self.filtering_values[self.Custom]
            }
        else:
            stop_words = {}

        if flag_df:
            stop_words.update({
                'min_df': self.str_to_num(self.min_df_input.text()),
                'max_df': self.str_to_num(self.max_df_input.text()),
            })
        return stop_words

    def restore_data(self, data):
        self.recent_sw_files = data.get('recent_sw_files')
        self.min_df_input.setText(data.get('min_df'))
        self.max_df_input.setText(data.get('max_df'))
        self.filtering_methods = data.get('methods')

        for flag, ch_box in zip(self.filtering_methods, self.group.buttons()):
            ch_box.setChecked(flag)

        self.enable_choice_settings()  # Enable the settings if set.
        self.set_file_list()  # Fill the combo box with the recent sw files.
        self.select_file(0)  # Select the first file.

    def export_data(self):
        return {
            'is_enabled': self.enabled,
            'methods': self.filtering_methods,
            'recent_sw_files': self.recent_sw_files,
            'min_df': self.min_df_input.text(),
            'max_df': self.max_df_input.text(),
        }


PREPROCESSOR_MODULES = [
    TokenizerModule,
    TransformationModule,
    CasingModule,
    FilteringModule,
]


class OWPreprocess(OWWidget):

    name = 'Preprocess Text'
    description = 'Construct a text pre-processing pipeline.'
    icon = 'icons/TextPreprocess.svg'
    priority = 30

    inputs = [(Input.CORPUS, Corpus, 'set_data')]
    outputs = [(Output.PP_CORPUS, Corpus)]

    autocommit = settings.Setting(True)
    # Persistent data for each module is stored here.
    persistent_data_tokenizer = settings.Setting({})
    persistent_data_casing = settings.Setting({})
    persistent_data_stemmer = settings.Setting({})
    persistent_data_filter = settings.Setting({})
    preprocessors = []  # Pre-processing modules for the current run.

    def __init__(self, parent=None):
        super().__init__(parent)

        self.corpus = None

        # -- INFO --
        info_box = gui.widgetBox(self.controlArea, 'Info')
        self.controlArea.layout().addStretch()
        self.info_label = gui.label(info_box, self,
                                    'No input corpus detected.')
        # Commit checkbox and commit button.
        output_box = gui.widgetBox(self.controlArea, 'Output')
        auto_commit_box = gui.auto_commit(output_box, self, 'autocommit',
                                          'Commit', box=False)
        auto_commit_box.setMinimumWidth(170)

        # -- PIPELINE --
        frame = QFrame()
        frame.setContentsMargins(0, 0, 0, 0)
        frame.setFrameStyle(QFrame.Box)
        frame.setStyleSheet('.QFrame { border: 1px solid #B3B3B3; }')
        frame_layout = QVBoxLayout()
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(0)
        frame.setLayout(frame_layout)
        # Load the previous states.
        persistent_data = [
            self.persistent_data_tokenizer,
            self.persistent_data_stemmer,
            self.persistent_data_casing,
            self.persistent_data_filter,
        ]
        for ModuleClass, ModuleData in zip(PREPROCESSOR_MODULES,
                                           persistent_data):
            pp_module_widget = ModuleClass(ModuleData)  # Create pp instance.
            self.preprocessors.append(pp_module_widget)
            pp_module_widget.change_signal.connect(self.settings_invalidated)
            pp_module_widget.error_signal.connect(self.display_message)

            frame_layout.addWidget(pp_module_widget)
        self.store_pipeline()  # Store the pipeline after loading it.

        frame_layout.addStretch()
        self.mainArea.layout().addWidget(frame)

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
        self.store_pipeline()  # Store the new pipeline.
        if self.corpus is not None:
            pp = self.assemble_preprocessor()
            if pp is not None:
                self.apply(pp)

    def apply(self, preprocessor):
        with self.progressBar(len(self.corpus)*2) as progress_bar:
            self.progress_bar = progress_bar
            output = preprocessor(self.corpus)
        self.progress_bar = None
        self.send(Output.PP_CORPUS, output)

    def assemble_preprocessor(self):
        self.error(0, '')

        pp_settings = {
            # If disabled, this defaults to True, which is not what we want.
            'lowercase': False,
        }
        for pp in self.preprocessors:
            if pp.enabled:
                pp_settings.update(pp.get_pp_setting())
        pp_settings['callback'] = self.document_finished
        try:
            preprocessor = Preprocessor(**pp_settings)
        except Exception as e:
            self.error(0, str(e))
            return None
        return preprocessor

    def store_pipeline(self):
        for pp in self.preprocessors:
            if isinstance(pp, TokenizerModule):
                self.persistent_data_tokenizer = pp.export_data()
            elif isinstance(pp, CasingModule):
                self.persistent_data_casing = pp.export_data()
            elif isinstance(pp, TransformationModule):
                self.persistent_data_stemmer = pp.export_data()
            elif isinstance(pp, FilteringModule):
                self.persistent_data_filter = pp.export_data()

    def document_finished(self):
        if self.progress_bar is not None:
            self.progress_bar.advance()

    @Slot()
    def settings_invalidated(self):
        self.commit()

    @Slot(str)
    def display_message(self, message):
        self.error(0, message)

if __name__ == '__main__':
    app = QApplication([])
    widget = OWPreprocess()
    widget.show()
    corpus = Corpus.from_file('bookexcerpts')
    widget.set_data(corpus)
    app.exec()
