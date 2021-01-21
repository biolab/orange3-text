from typing import Dict, Optional, List, Callable, Tuple, Type, Union
from types import SimpleNamespace
import os
import random
import pkg_resources

from AnyQt.QtCore import Qt, pyqtSignal
from AnyQt.QtWidgets import QComboBox, QButtonGroup, QLabel, QCheckBox, \
    QRadioButton, QGridLayout, QLineEdit, QSpinBox, QFormLayout, QHBoxLayout, \
    QDoubleSpinBox, QFileDialog, QAbstractSpinBox
from AnyQt.QtWidgets import QWidget, QPushButton, QSizePolicy, QStyle
from AnyQt.QtGui import QBrush

from Orange.util import wrap_callback
from orangewidget.utils.filedialogs import RecentPath

import Orange.widgets.data.owpreprocess
from Orange.widgets import gui
from Orange.widgets.data.owpreprocess import PreprocessAction, Description
from Orange.widgets.data.utils.preprocess import ParametersRole, \
    DescriptionRole, BaseEditor, StandardItemModel
from Orange.widgets.settings import Setting
from Orange.widgets.utils.concurrent import TaskState, ConcurrentWidgetMixin
from Orange.widgets.widget import Input, Output, Msg, Message

from orangecontrib.text import Corpus
from orangecontrib.text.misc import nltk_data_dir
from orangecontrib.text.preprocess import *
from orangecontrib.text.preprocess.normalize import UDPipeStopIteration
from orangecontrib.text.tag import AveragedPerceptronTagger, MaxEntTagger, \
    POSTagger
from orangecontrib.text.tag.pos import StanfordPOSTaggerError

_DEFAULT_NONE = "(none)"


def icon_path(basename):
    return pkg_resources.resource_filename(__name__, "icons/" + basename)


class Result(SimpleNamespace):
    corpus = None  # type: Optional[Corpus]
    msgs = []


class ValidatedLineEdit(QLineEdit):
    def __init__(self, text: str, validator: Callable):
        super().__init__(text)
        self.__validator = validator
        self.textChanged.connect(self.__validate)

    def __validate(self):
        color = "gray" if self.__validator(self.text()) else "red"
        self.setStyleSheet(f"QLineEdit {{ border : 1px solid {color};}}")


class ComboBox(QComboBox):
    def __init__(self, master: BaseEditor, items: List[str], value: str,
                 callback: Callable):
        super().__init__(master)
        self.setMinimumWidth(80)
        self.addItems(items)
        self.setCurrentText(value)
        self.currentTextChanged.connect(callback)


class UDPipeComboBox(QComboBox):
    def __init__(self, master: BaseEditor, value: str, default: str,
                 callback: Callable):
        super().__init__(master)
        self.__items = []  # type: List
        self.__default_lang = default
        self.add_items(value)
        self.currentTextChanged.connect(callback)
        self.setMinimumWidth(80)

    @property
    def items(self) -> List:
        return UDPipeLemmatizer().models.supported_languages

    def add_items(self, value: str):
        self.__items = self.items
        self.addItems(self.__items)
        if value in self.__items:
            self.setCurrentText(value)
        elif self.__default_lang in self.__items:
            self.setCurrentText(self.__default_lang)
        elif self.__items:
            self.setCurrentIndex(0)

    def showPopup(self):
        if self.__items != self.items:
            self.clear()
            self.add_items(self.currentText())
        super().showPopup()


class RangeSpins(QHBoxLayout):
    SpinBox = QSpinBox

    def __init__(self, start: float, step: float, end: float, minimum: int,
                 maximum: int, cb_start: Callable, cb_end: Callable,
                 edited: Callable):
        super().__init__()
        self._min, self._max = minimum, maximum
        self._spin_start = self.SpinBox(minimum=minimum, maximum=end,
                                        value=start, singleStep=step)
        self._spin_end = self.SpinBox(minimum=start, maximum=maximum,
                                      value=end, singleStep=step)
        self._spin_start.editingFinished.connect(edited)
        self._spin_end.editingFinished.connect(edited)
        self._spin_start.valueChanged.connect(self._spin_end.setMinimum)
        self._spin_end.valueChanged.connect(self._spin_start.setMaximum)
        self._spin_start.valueChanged.connect(cb_start)
        self._spin_end.valueChanged.connect(cb_end)
        self._spin_start.setFixedWidth(50)
        self._spin_end.setFixedWidth(50)
        self.addWidget(self._spin_start)
        self.addWidget(self._spin_end)

    def set_range(self, start: float, end: float):
        self._spin_start.setMaximum(self._max)
        self._spin_end.setMinimum(self._min)
        self._spin_start.setValue(start)
        self._spin_end.setValue(end)
        self._spin_start.setMaximum(end)
        self._spin_end.setMinimum(start)

    def spins(self) -> Tuple[QAbstractSpinBox, QAbstractSpinBox]:
        return self._spin_start, self._spin_end


class RangeDoubleSpins(RangeSpins):
    SpinBox = QDoubleSpinBox

    def __init__(self, start: float, step: float, end: float, minimum: int,
                 maximum: int, cb_start: Callable, cb_end: Callable,
                 edited: Callable):
        super().__init__(start, step, end, minimum, maximum, cb_start,
                         cb_end, edited)
        self._spin_start.setMaximumWidth(1000)
        self._spin_end.setMaximumWidth(1000)
        self._spin_start.setMinimumWidth(0)
        self._spin_end.setMinimumWidth(0)


class FileLoader(QWidget):
    activated = pyqtSignal()
    file_loaded = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.recent_paths = []

        self.file_combo = QComboBox()
        self.file_combo.setMinimumWidth(80)
        self.file_combo.activated.connect(self._activate)

        self.browse_btn = QPushButton("...")
        icon = self.style().standardIcon(QStyle.SP_DirOpenIcon)
        self.browse_btn.setIcon(icon)
        self.browse_btn.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.browse_btn.clicked.connect(self.browse)

        self.load_btn = QPushButton("")
        icon = self.style().standardIcon(QStyle.SP_BrowserReload)
        self.load_btn.setIcon(icon)
        self.load_btn.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.load_btn.setAutoDefault(True)
        self.load_btn.clicked.connect(self.file_loaded)

    def browse(self):
        start_file = self.last_path() or os.path.expanduser("~/")
        formats = ["Text files (*.txt)", "All files (*)"]
        file_name, _ = QFileDialog.getOpenFileName(
            None, "Open...", start_file, ";;".join(formats), formats[0])
        if not file_name:
            return
        self.add_path(file_name)
        self._activate()

    def _activate(self):
        self.activated.emit()
        self.file_loaded.emit()

    def set_current_file(self, path: str):
        if path:
            self.add_path(path)
            self.file_combo.setCurrentText(path)
        else:
            self.file_combo.setCurrentText("(none)")

    def get_current_file(self) -> Optional[RecentPath]:
        index = self.file_combo.currentIndex()
        if index >= len(self.recent_paths) or index < 0:
            return None
        path = self.recent_paths[index]
        return path if isinstance(path, RecentPath) else None

    def add_path(self, filename: str):
        recent = RecentPath.create(filename, [])
        if recent in self.recent_paths:
            self.recent_paths.remove(recent)
        self.recent_paths.insert(0, recent)
        self.set_file_list()

    def set_file_list(self):
        self.file_combo.clear()
        for i, recent in enumerate(self.recent_paths):
            self.file_combo.addItem(recent.basename)
            self.file_combo.model().item(i).setToolTip(recent.abspath)
            if not os.path.exists(recent.abspath):
                self.file_combo.setItemData(i, QBrush(Qt.red),
                                            Qt.TextColorRole)
        self.file_combo.addItem(_DEFAULT_NONE)

    def last_path(self) -> Optional[str]:
        return self.recent_paths[0].abspath if self.recent_paths else None


class PreprocessorModule(BaseEditor):
    @staticmethod
    def get_tooltip(method: Type) -> Optional[str]:
        if not method.__doc__:
            return None
        return " ".join([l.strip() for l in
                         method.__doc__.split("\n")]).strip(".")


class SingleMethodModule(PreprocessorModule):
    Methods = NotImplemented
    DEFAULT_METHOD = NotImplemented

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.__method = self.DEFAULT_METHOD

        self.setLayout(QGridLayout())
        self.__group = QButtonGroup(self, exclusive=True)
        self.__group.buttonClicked.connect(self.__method_rb_clicked)
        for method_id in range(len(self.Methods)):
            method = self.Methods[method_id]
            rb = QRadioButton(method.name)
            rb.setChecked(self.__method == method_id)
            rb.setToolTip(self.get_tooltip(method))
            self.__group.addButton(rb, method_id)
            self.layout().addWidget(rb)

    @property
    def method(self) -> int:
        return self.__method

    def setParameters(self, params: Dict):
        self._set_method(params.get("method", self.DEFAULT_METHOD))

    def _set_method(self, method: int):
        if self.__method != method:
            self.__method = method
            self.__group.button(method).setChecked(True)
            self.changed.emit()

    def __method_rb_clicked(self):
        self._set_method(self.__group.checkedId())
        self.edited.emit()

    def parameters(self) -> Dict:
        return {"method": self.__method}

    def __repr__(self):
        return self.Methods[self.__method].name


class MultipleMethodModule(PreprocessorModule):
    Methods = NotImplemented
    DEFAULT_METHODS = NotImplemented

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.__methods = self.DEFAULT_METHODS

        self.setLayout(QGridLayout())
        self.__cbs = []
        for method in range(len(self.Methods)):
            cb = QCheckBox(self.Methods[method].name, self)
            cb.setChecked(method in self.__methods)
            cb.clicked.connect(self.__method_check_clicked)
            cb.setToolTip(self.get_tooltip(self.Methods[method]))
            self.__cbs.append((method, cb))
            self.layout().addWidget(cb)

    @property
    def methods(self) -> List[int]:
        return self.__methods

    def setParameters(self, params: Dict):
        self.__set_methods(params.get("methods", self.DEFAULT_METHODS))

    def __set_methods(self, methods: List[int]):
        if self.__methods != methods:
            self.__methods = methods
            for i, cb in self.__cbs:
                cb.setChecked(i in self.__methods)
            self.changed.emit()

    def __method_check_clicked(self):
        self.__set_methods([i for i, cb in self.__cbs if cb.isChecked()])
        self.edited.emit()

    def parameters(self) -> Dict:
        return {"methods": self.__methods}

    def __repr__(self) -> str:
        names = [self.Methods[method].name for method in self.__methods]
        return ", ".join(names)


class TransformationModule(MultipleMethodModule):
    Lowercase, Accents, Parse, Urls = range(4)
    Methods = {Lowercase: LowercaseTransformer,
               Accents: StripAccentsTransformer,
               Parse: HtmlTransformer,
               Urls: UrlRemover}
    DEFAULT_METHODS = [Lowercase]

    @staticmethod
    def createinstance(params: Dict) -> List[BaseTransformer]:
        methods = params.get("methods", TransformationModule.DEFAULT_METHODS)
        return [TransformationModule.Methods[method]() for method in methods]


class TokenizerModule(SingleMethodModule):
    Word, Whitespace, Sentence, Regexp, Tweet = range(5)
    Methods = {Word: WordPunctTokenizer,
               Whitespace: WhitespaceTokenizer,
               Sentence: PunktSentenceTokenizer,
               Regexp: RegexpTokenizer,
               Tweet: TweetTokenizer}
    DEFAULT_METHOD = Regexp
    DEFAULT_PATTERN = r"\w+"

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.__pattern = self.DEFAULT_PATTERN
        validator = RegexpTokenizer.validate_regexp
        self.__edit = ValidatedLineEdit(self.__pattern, validator)
        self.__edit.editingFinished.connect(self.__edit_finished)
        self.layout().addWidget(QLabel("Pattern:"), 3, 1)
        self.layout().addWidget(self.__edit, 3, 2)

    def setParameters(self, params: Dict):
        super().setParameters(params)
        self.__set_pattern(params.get("pattern", self.DEFAULT_PATTERN))

    def __set_pattern(self, pattern: str):
        if self.__pattern != pattern:
            self.__pattern = pattern
            self.__edit.setText(pattern)
            self.changed.emit()

    def __edit_finished(self):
        pattern = self.__edit.text()
        if self.__pattern != pattern:
            self.__set_pattern(pattern)
            if self.method == self.Regexp:
                self.edited.emit()

    def parameters(self) -> Dict:
        params = super().parameters()
        params.update({"pattern": self.__pattern})
        return params

    @staticmethod
    def createinstance(params: Dict) -> BaseTokenizer:
        method = params.get("method", TokenizerModule.DEFAULT_METHOD)
        pattern = params.get("pattern", TokenizerModule.DEFAULT_PATTERN)
        args = {"pattern": pattern} if method == TokenizerModule.Regexp else {}
        return TokenizerModule.Methods[method](**args)

    def __repr__(self):
        text = super().__repr__()
        if self.method == self.Regexp:
            text = f"{text} ({self.__pattern})"
        return text


class NormalizationModule(SingleMethodModule):
    Porter, Snowball, WordNet, UDPipe = range(4)
    Methods = {Porter: PorterStemmer,
               Snowball: SnowballStemmer,
               WordNet: WordNetLemmatizer,
               UDPipe: UDPipeLemmatizer}
    DEFAULT_METHOD = Porter
    DEFAULT_LANGUAGE = "English"
    DEFAULT_USE_TOKE = False

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.__snowball_lang = self.DEFAULT_LANGUAGE
        self.__udpipe_lang = self.DEFAULT_LANGUAGE
        self.__use_tokenizer = self.DEFAULT_USE_TOKE

        self.__combo_sbl = ComboBox(
            self, SnowballStemmer.supported_languages,
            self.__snowball_lang, self.__set_snowball_lang
        )
        self.__combo_udl = UDPipeComboBox(
            self, self.__udpipe_lang, self.DEFAULT_LANGUAGE,
            self.__set_udpipe_lang
        )
        self.__check_use = QCheckBox("UDPipe tokenizer",
                                     checked=self.DEFAULT_USE_TOKE)
        self.__check_use.clicked.connect(self.__set_use_tokenizer)

        label = QLabel("Language:")
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.layout().addWidget(label, self.Snowball, 1)
        self.layout().addWidget(self.__combo_sbl, self.Snowball, 2)

        label = QLabel("Language:")
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.layout().addWidget(label, self.UDPipe, 1)
        self.layout().addWidget(self.__combo_udl, self.UDPipe, 2)

        self.layout().addWidget(self.__check_use, self.UDPipe, 3)
        self.layout().setColumnStretch(2, 1)
        self.__enable_udpipe()

    def __enable_udpipe(self):
        enable = bool(self.__combo_udl.items)
        layout = self.layout()  # type: QGridLayout
        for i in range(4):
            layout.itemAtPosition(self.UDPipe, i).widget().setEnabled(enable)
        if self.method == self.UDPipe and not enable:
            self._set_method(self.Porter)

    def setParameters(self, params: Dict):
        super().setParameters(params)
        snowball_lang = params.get("snowball_language", self.DEFAULT_LANGUAGE)
        self.__set_snowball_lang(snowball_lang)
        udpipe_lang = params.get("udpipe_language", self.DEFAULT_LANGUAGE)
        self.__set_udpipe_lang(udpipe_lang)
        use_tokenizer = params.get("udpipe_tokenizer", self.DEFAULT_USE_TOKE)
        self.__set_use_tokenizer(use_tokenizer)

    def _set_method(self, method: int):
        super()._set_method(method)
        self.__enable_udpipe()

    def __set_snowball_lang(self, language: str):
        if self.__snowball_lang != language:
            self.__snowball_lang = language
            self.__combo_sbl.setCurrentText(language)
            self.changed.emit()
            if self.method == self.Snowball:
                self.edited.emit()

    def __set_udpipe_lang(self, language: str):
        if self.__udpipe_lang != language:
            self.__udpipe_lang = language
            self.__combo_udl.setCurrentText(language)
            self.changed.emit()
            if self.method == self.UDPipe:
                self.edited.emit()

    def __set_use_tokenizer(self, use: bool):
        if self.__use_tokenizer != use:
            self.__use_tokenizer = use
            self.__check_use.setChecked(use)
            self.changed.emit()
            if self.method == self.UDPipe:
                self.edited.emit()

    def parameters(self) -> Dict:
        params = super().parameters()
        params.update({"snowball_language": self.__snowball_lang,
                       "udpipe_language": self.__udpipe_lang,
                       "udpipe_tokenizer": self.__use_tokenizer})
        return params

    @staticmethod
    def createinstance(params: Dict) -> BaseNormalizer:
        method = params.get("method", NormalizationModule.DEFAULT_METHOD)
        args = {}
        def_lang = NormalizationModule.DEFAULT_LANGUAGE
        if method == NormalizationModule.Snowball:
            args = {"language": params.get("snowball_language", def_lang)}
        elif method == NormalizationModule.UDPipe:
            def_use = NormalizationModule.DEFAULT_USE_TOKE
            args = {"language": params.get("udpipe_language", def_lang),
                    "use_tokenizer": params.get("udpipe_tokenizer", def_use)}
        return NormalizationModule.Methods[method](**args)

    def __repr__(self):
        text = super().__repr__()
        if self.method == self.Snowball:
            text = f"{text} ({self.__snowball_lang})"
        elif self.method == self.UDPipe:
            text = f"{text} ({self.__udpipe_lang}, " \
                   f"Tokenize: {['No', 'Yes'][self.__use_tokenizer]})"
        return text


class FilteringModule(MultipleMethodModule):
    Stopwords, Lexicon, Regexp, DocFreq, DummyDocFreq, MostFreq = range(6)
    Methods = {Stopwords: StopwordsFilter,
               Lexicon: LexiconFilter,
               Regexp: RegexpFilter,
               DocFreq: FrequencyFilter,
               DummyDocFreq: FrequencyFilter,
               MostFreq: MostFrequentTokensFilter}
    DEFAULT_METHODS = [Stopwords]
    DEFAULT_LANG = "English"
    DEFAULT_NONE = None
    DEFAULT_PATTERN = "\.|,|:|;|!|\?|\(|\)|\||\+|\'|\"|‘|’|“|”|\'|" \
                      "\’|…|\-|–|—|\$|&|\*|>|<|\/|\[|\]"
    DEFAULT_FREQ_TYPE = 0  # 0 - relative freq, 1 - absolute freq
    DEFAULT_REL_START, DEFAULT_REL_END, REL_MIN, REL_MAX = 0.1, 0.9, 0, 1
    DEFAULT_ABS_START, DEFAULT_ABS_END, ABS_MIN, ABS_MAX = 1, 10, 0, 10000
    DEFAULT_N_TOKEN = 100

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.__sw_lang = self.DEFAULT_LANG
        self.__sw_file = self.DEFAULT_NONE
        self.__lx_file = self.DEFAULT_NONE
        self.__pattern = self.DEFAULT_PATTERN
        self.__freq_type = self.DEFAULT_FREQ_TYPE
        self.__rel_freq_st = self.DEFAULT_REL_START
        self.__rel_freq_en = self.DEFAULT_REL_END
        self.__abs_freq_st = self.DEFAULT_ABS_START
        self.__abs_freq_en = self.DEFAULT_ABS_END
        self.__n_token = self.DEFAULT_N_TOKEN
        self.__invalidated = False

        self.__combo = ComboBox(
            self, [_DEFAULT_NONE] + StopwordsFilter.supported_languages(),
            self.__sw_lang, self.__set_language
        )
        self.__sw_loader = FileLoader()
        self.__sw_loader.set_file_list()
        self.__sw_loader.activated.connect(self.__sw_loader_activated)
        self.__sw_loader.file_loaded.connect(self.__sw_invalidate)

        self.__lx_loader = FileLoader()
        self.__lx_loader.set_file_list()
        self.__lx_loader.activated.connect(self.__lx_loader_activated)
        self.__lx_loader.file_loaded.connect(self.__lx_invalidate)

        validator = RegexpFilter.validate_regexp
        self.__edit = ValidatedLineEdit(self.__pattern, validator)
        self.__edit.editingFinished.connect(self.__edit_finished)

        rel_freq_rb = QRadioButton("Relative:")
        abs_freq_rb = QRadioButton("Absolute:")
        self.__freq_group = group = QButtonGroup(self, exclusive=True)
        group.addButton(rel_freq_rb, 0)
        group.addButton(abs_freq_rb, 1)
        group.buttonClicked.connect(self.__freq_group_clicked)
        group.button(self.__freq_type).setChecked(True)

        self.__rel_range_spins = RangeDoubleSpins(
            self.__rel_freq_st, 0.05, self.__rel_freq_en, self.REL_MIN,
            self.REL_MAX, self.__set_rel_freq_start, self.__set_rel_freq_end,
            self.__rel_spins_edited
        )
        self.__abs_range_spins = RangeSpins(
            self.__abs_freq_st, 1, self.__abs_freq_en, self.ABS_MIN,
            self.ABS_MAX, self.__set_abs_freq_start, self.__set_abs_freq_end,
            self.__abs_spins_edited
        )

        self.__spin_n = QSpinBox(
            minimum=1, maximum=10 ** 6, value=self.__n_token)
        self.__spin_n.editingFinished.connect(self.__spin_n_edited)
        self.__spin_n.valueChanged.connect(self.changed)

        self.layout().addWidget(self.__combo, self.Stopwords, 1)
        self.layout().addWidget(self.__sw_loader.file_combo,
                                self.Stopwords, 2, 1, 2)
        self.layout().addWidget(self.__sw_loader.browse_btn, self.Stopwords, 4)
        self.layout().addWidget(self.__sw_loader.load_btn, self.Stopwords, 5)
        self.layout().addWidget(self.__lx_loader.file_combo,
                                self.Lexicon, 2, 1, 2)
        self.layout().addWidget(self.__lx_loader.browse_btn, self.Lexicon, 4)
        self.layout().addWidget(self.__lx_loader.load_btn, self.Lexicon, 5)
        self.layout().addWidget(self.__edit, self.Regexp, 1, 1, 5)
        spins = self.__rel_range_spins.spins()
        self.layout().addWidget(rel_freq_rb, self.DocFreq, 1)
        self.layout().addWidget(spins[0], self.DocFreq, 2)
        self.layout().addWidget(spins[1], self.DocFreq, 3)
        spins = self.__abs_range_spins.spins()
        self.layout().addWidget(abs_freq_rb, self.DummyDocFreq, 1)
        self.layout().addWidget(spins[0], self.DummyDocFreq, 2)
        self.layout().addWidget(spins[1], self.DummyDocFreq, 3)
        title = self.layout().itemAtPosition(self.DummyDocFreq, 0).widget()
        title.hide()
        self.layout().addWidget(self.__spin_n, self.MostFreq, 1)
        self.layout().setColumnStretch(3, 1)

    def __sw_loader_activated(self):
        self.__sw_file = self.__sw_loader.get_current_file()
        self.changed.emit()
        if self.Stopwords in self.methods:
            self.edited.emit()

    def __sw_invalidate(self):
        if self.Stopwords in self.methods and self.__sw_file:
            self.__invalidated = random.random()
            self.edited.emit()

    def __lx_loader_activated(self):
        self.__lx_file = self.__lx_loader.get_current_file()
        self.changed.emit()
        if self.Lexicon in self.methods:
            self.edited.emit()

    def __lx_invalidate(self):
        if self.Lexicon in self.methods and self.__lx_file:
            self.__invalidated = random.random()
            self.edited.emit()

    def __edit_finished(self):
        pattern = self.__edit.text()
        if self.__pattern != pattern:
            self.__set_pattern(pattern)
            if self.Regexp in self.methods:
                self.edited.emit()

    def __freq_group_clicked(self):
        i = self.__freq_group.checkedId()
        if self.__freq_type != i:
            self.__set_freq_type(i)
            if self.DocFreq in self.methods:
                self.edited.emit()

    def __rel_spins_edited(self):
        if self.DocFreq in self.methods and self.__freq_type == 0:
            self.edited.emit()

    def __abs_spins_edited(self):
        if self.DocFreq in self.methods and self.__freq_type == 1:
            self.edited.emit()

    def __spin_n_edited(self):
        n = self.__spin_n.value()
        if self.__n_token != n:
            self.__set_n_tokens(n)
            if self.MostFreq in self.methods:
                self.edited.emit()

    def setParameters(self, params: Dict):
        super().setParameters(params)
        self.__set_language(params.get("language", self.DEFAULT_LANG))
        self.__set_sw_path(params.get("sw_path", self.DEFAULT_NONE),
                           params.get("sw_list", []))
        self.__set_lx_path(params.get("lx_path", self.DEFAULT_NONE),
                           params.get("lx_list", []))
        self.__set_pattern(params.get("pattern", self.DEFAULT_PATTERN))
        self.__set_freq_type(params.get("freq_type", self.DEFAULT_FREQ_TYPE))
        self.__set_rel_freq_range(
            params.get("rel_start", self.DEFAULT_REL_START),
            params.get("rel_end", self.DEFAULT_REL_END)
        )
        self.__set_abs_freq_range(
            params.get("abs_start", self.DEFAULT_ABS_START),
            params.get("abs_end", self.DEFAULT_ABS_END)
        )
        self.__set_n_tokens(params.get("n_tokens", self.DEFAULT_N_TOKEN))
        self.__invalidated = False

    def __set_language(self, language: str):
        if self.__sw_lang != language:
            self.__sw_lang = language
            self.__combo.setCurrentText(language)
            self.changed.emit()
            if self.Stopwords in self.methods:
                self.edited.emit()

    def __set_sw_path(self, path: RecentPath, paths: List[RecentPath] = []):
        self.__sw_loader.recent_paths = paths
        self.__sw_loader.set_file_list()
        self.__sw_loader.set_current_file(_to_abspath(path))
        self.__sw_file = self.__sw_loader.get_current_file()

    def __set_lx_path(self, path: RecentPath, paths: List[RecentPath] = []):
        self.__lx_loader.recent_paths = paths
        self.__lx_loader.set_file_list()
        self.__lx_loader.set_current_file(_to_abspath(path))
        self.__lx_file = self.__lx_loader.get_current_file()

    def __set_pattern(self, pattern: str):
        if self.__pattern != pattern:
            self.__pattern = pattern
            self.__edit.setText(pattern)
            self.changed.emit()

    def __set_freq_type(self, freq_type: int):
        if self.__freq_type != freq_type:
            self.__freq_type = freq_type
            self.__freq_group.button(self.__freq_type).setChecked(True)
            self.changed.emit()

    def __set_rel_freq_range(self, start: float, end: float):
        self.__set_rel_freq_start(start)
        self.__set_rel_freq_end(end)
        self.__rel_range_spins.set_range(start, end)

    def __set_rel_freq_start(self, n: float):
        if self.__rel_freq_st != n:
            self.__rel_freq_st = n
            self.changed.emit()

    def __set_rel_freq_end(self, n: float):
        if self.__rel_freq_en != n:
            self.__rel_freq_en = n
            self.changed.emit()

    def __set_abs_freq_range(self, start: int, end: int):
        self.__set_abs_freq_start(start)
        self.__set_abs_freq_end(end)
        self.__abs_range_spins.set_range(start, end)

    def __set_abs_freq_start(self, n: int):
        if self.__abs_freq_st != n:
            self.__abs_freq_st = n
            self.changed.emit()

    def __set_abs_freq_end(self, n: int):
        if self.__abs_freq_en != n:
            self.__abs_freq_en = n
            self.changed.emit()

    def __set_n_tokens(self, n: int):
        if self.__n_token != n:
            self.__n_token = n
            self.__spin_n.setValue(n)
            self.changed.emit()

    def parameters(self) -> Dict:
        params = super().parameters()
        params.update({"language": self.__sw_lang,
                       "sw_path": self.__sw_file,
                       "sw_list": self.__sw_loader.recent_paths,
                       "lx_path": self.__lx_file,
                       "lx_list": self.__lx_loader.recent_paths,
                       "pattern": self.__pattern,
                       "freq_type": self.__freq_type,
                       "rel_start": self.__rel_freq_st,
                       "rel_end": self.__rel_freq_en,
                       "abs_start": self.__abs_freq_st,
                       "abs_end": self.__abs_freq_en,
                       "n_tokens": self.__n_token,
                       "invalidated": self.__invalidated})
        return params

    @staticmethod
    def createinstance(params: Dict) -> List[BaseTokenFilter]:
        def map_none(s):
            return "" if s == _DEFAULT_NONE else s

        methods = params.get("methods", FilteringModule.DEFAULT_METHODS)
        filters = []
        if FilteringModule.Stopwords in methods:
            lang = params.get("language", FilteringModule.DEFAULT_LANG)
            path = params.get("sw_path", FilteringModule.DEFAULT_NONE)
            filters.append(StopwordsFilter(language=map_none(lang),
                                           path=_to_abspath(path)))
        if FilteringModule.Lexicon in methods:
            path = params.get("lx_path", FilteringModule.DEFAULT_NONE)
            filters.append(LexiconFilter(path=_to_abspath(path)))
        if FilteringModule.Regexp in methods:
            pattern = params.get("pattern", FilteringModule.DEFAULT_PATTERN)
            filters.append(RegexpFilter(pattern=pattern))
        if FilteringModule.DocFreq in methods:
            if params.get("freq_type", FilteringModule.DEFAULT_FREQ_TYPE) == 0:
                st = params.get("rel_start", FilteringModule.DEFAULT_REL_START)
                end = params.get("rel_end", FilteringModule.DEFAULT_REL_END)
            else:
                st = params.get("abs_start", FilteringModule.DEFAULT_ABS_START)
                end = params.get("abs_end", FilteringModule.DEFAULT_ABS_END)
            filters.append(FrequencyFilter(min_df=st, max_df=end))
        if FilteringModule.MostFreq in methods:
            n = params.get("n_tokens", FilteringModule.DEFAULT_N_TOKEN)
            filters.append(MostFrequentTokensFilter(keep_n=n))
        return filters

    def __repr__(self):
        texts = []
        for method in self.methods:
            if method == self.Stopwords:
                append = f"Language: {self.__sw_lang}, " \
                         f"File: {_to_abspath(self.__sw_file)}"
            elif method == self.Lexicon:
                append = f"File: {_to_abspath(self.__lx_file)}"
            elif method == self.Regexp:
                append = f"{self.__pattern}"
            elif method == self.DocFreq:
                if self.__freq_type == 0:
                    append = f"[{self.__rel_freq_st}, {self.__rel_freq_en}]"
                else:
                    append = f"[{self.__abs_freq_st}, {self.__abs_freq_en}]"
            elif method == self.MostFreq:
                append = f"{self.__n_token}"
            texts.append(f"{self.Methods[method].name} ({append})")
        return ", ".join(texts)


def _to_abspath(path: RecentPath) -> str:
    return path.abspath if path else None


class NgramsModule(PreprocessorModule):
    DEFAULT_START = 1
    DEFAULT_END = 2
    MIN, MAX = 1, 10

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.__start = self.DEFAULT_START
        self.__end = self.DEFAULT_END

        self.setLayout(QFormLayout())
        self.__range_spins = RangeSpins(
            self.__start, 1, self.__end, self.MIN, self.MAX,
            self.__set_start, self.__set_end, self.edited)
        self.layout().addRow("Range:", self.__range_spins)

    def setParameters(self, params: Dict):
        self.__set_range(params.get("start", self.DEFAULT_START),
                         params.get("end", self.DEFAULT_END))

    def __set_range(self, start: int, end: int):
        self.__set_start(start)
        self.__set_end(end)
        self.__range_spins.set_range(start, end)

    def __set_start(self, n: int):
        if self.__start != n:
            self.__start = n
            self.changed.emit()

    def __set_end(self, n: int):
        if self.__end != n:
            self.__end = n
            self.changed.emit()

    def parameters(self) -> Dict:
        return {"start": self.__start, "end": self.__end}

    def __repr__(self):
        return f"({self.__start}, {self.__end})"

    @staticmethod
    def createinstance(params: Dict) -> NGrams:
        return NGrams((params.get("start", NgramsModule.DEFAULT_START),
                       params.get("end", NgramsModule.DEFAULT_END)))


class POSTaggingModule(SingleMethodModule):
    Averaged, MaxEnt, Stanford = range(3)
    Methods = {Averaged: AveragedPerceptronTagger,
               MaxEnt: MaxEntTagger}
    DEFAULT_METHOD = Averaged

    @staticmethod
    def createinstance(params: Dict) -> POSTagger:
        method = params.get("method", POSTaggingModule.DEFAULT_METHOD)
        return POSTaggingModule.Methods[method]()


PREPROCESS_ACTIONS = [
    PreprocessAction(
        "Transformation", "preprocess.transform", "",
        Description("Transformation", icon_path("Transform.svg")),
        TransformationModule
    ),
    PreprocessAction(
        "Tokenization", "preprocess.tokenize", "",
        Description("Tokenization", icon_path("Tokenize.svg")),
        TokenizerModule
    ),
    PreprocessAction(
        "Normalization", "preprocess.normalize", "",
        Description("Normalization", icon_path("Normalize.svg")),
        NormalizationModule
    ),
    PreprocessAction(
        "Filtering", "preprocess.filter", "",
        Description("Filtering", icon_path("Filter.svg")),
        FilteringModule
    ),
    PreprocessAction(
        "N-grams Range", "preprocess.ngrams", "",
        Description("N-grams Range", icon_path("NGrams.svg")),
        NgramsModule
    ),
    PreprocessAction(
        "POS Tagger", "tag.pos", "",
        Description("POS Tagger", icon_path("POSTag.svg")),
        POSTaggingModule
    )
]


class OWPreprocess(Orange.widgets.data.owpreprocess.OWPreprocess,
                   ConcurrentWidgetMixin):
    name = "Preprocess Text"
    description = "Construct a text pre-processing pipeline."
    icon = "icons/TextPreprocess.svg"
    priority = 200
    keywords = []

    settings_version = 3

    class Inputs:
        corpus = Input("Corpus", Corpus)

    class Outputs:
        corpus = Output("Corpus", Corpus)

    class Error(Orange.widgets.data.owpreprocess.OWPreprocess.Error):
        unknown_error = Msg("{}")
        udpipe_offline_no_models = Msg("No UDPipe model is selected.")
        file_not_found = Msg("File not found.")
        invalid_encoding = Msg("Invalid file encoding. Please save the "
                               "file as UTF-8 and try again.")
        stanford_tagger = Msg("Problem loading Stanford POS Tagger:\n{}")

    class Warning(Orange.widgets.data.owpreprocess.OWPreprocess.Warning):
        no_token_left = Msg("No tokens on the output.")
        udpipe_offline = Msg("No internet connection. UDPipe"
                             " works with local models.")
        udpipe_offline_no_models = Msg("No internet connection. "
                                       "UDPipe model is not available.")
        tokenizer_propagated = Msg("Tokenization should be placed before "
                                   "Normalization, Filtering, n-grams and "
                                   "POS Tagger.")
        tokenizer_ignored = Msg("Tokenization has been ignored due to "
                                "UDPipe tokenizer.")
        filtering_ignored = Msg("Filtering has been ignored due to "
                                "UDPipe tokenizer.")

    UserAdviceMessages = [
        Message(f"Some preprocessing methods require data (like word "
                f"relationships, stop words, punctuation rules etc.) "
                f"from the NLTK package. This data was downloaded to:"
                f" {nltk_data_dir()}", "nltk_data")]

    PREPROCESSORS = PREPROCESS_ACTIONS
    DEFAULT_PP = {"preprocessors": [("preprocess.transform", {}),
                                    ("preprocess.tokenize", {}),
                                    ("preprocess.filter", {})]
                  }  # type: Dict[str, List[Tuple[str, Dict]]]
    storedsettings = Setting(DEFAULT_PP)

    def __init__(self):
        ConcurrentWidgetMixin.__init__(self)
        Orange.widgets.data.owpreprocess.OWPreprocess.__init__(self)
        box = gui.vBox(self.controlArea, "Preview")
        self.preview = ""
        gui.label(box, self, "%(preview)s", wordWrap=True)
        self.controlArea.layout().insertWidget(1, box)
        self.controlArea.setFixedWidth(220)

    def load(self, saved: Dict) -> StandardItemModel:
        for i, (name, params) in enumerate(saved.get("preprocessors", [])):
            if name == "preprocess.filter" and params:
                self.__update_filtering_params(params)
                saved["preprocessors"][i] = (name, params)
        return super().load(saved)

    def __update_filtering_params(self, params: Dict):
        params["sw_path"] = self.__relocate_file(params.get("sw_path"))
        params["sw_list"] = self.__relocate_files(params.get("sw_list", []))
        params["lx_path"] = self.__relocate_file(params.get("lx_path"))
        params["lx_list"] = self.__relocate_files(params.get("lx_list", []))

    def __relocate_files(self, paths: List[RecentPath]) -> List[RecentPath]:
        return [self.__relocate_file(path) for path in paths]

    def __relocate_file(self, path: RecentPath) -> RecentPath:
        basedir = self.workflowEnv().get("basedir", None)
        if basedir is None or path is None:
            return path

        search_paths = [("basedir", basedir)]
        resolved = path.resolve(search_paths)
        kwargs = dict(title=path.title, sheet=path.sheet,
                      file_format=path.file_format)
        if resolved is not None:
            return RecentPath.create(resolved.abspath, search_paths, **kwargs)
        elif path.search(search_paths) is not None:
            return RecentPath.create(path.search(search_paths),
                                     search_paths, **kwargs)
        return path

    @Inputs.corpus
    def set_data(self, data: Corpus):
        self.cancel()
        super().set_data(data)

    def buildpreproc(self) -> PreprocessorList:
        plist = []
        for i in range(self.preprocessormodel.rowCount()):
            item = self.preprocessormodel.item(i)
            desc = item.data(DescriptionRole)
            params = item.data(ParametersRole)
            assert isinstance(params, dict)

            inst = desc.viewclass.createinstance(params)
            self._check_preprocessors(inst, plist)
            plist.extend(inst if isinstance(inst, list) else [inst])

        return PreprocessorList(plist)

    def _check_preprocessors(self, preprocessors: Union[Preprocessor, List],
                             plist: List[Preprocessor]):
        if isinstance(preprocessors, BaseTokenizer):
            if any(isinstance(pp, TokenizedPreprocessor) for pp in plist):
                self.Warning.tokenizer_propagated()
        elif isinstance(preprocessors, UDPipeLemmatizer):
            if not preprocessors.models.online:
                if not preprocessors.models.model_files:
                    self.Warning.udpipe_offline_no_models()
                else:
                    self.Warning.udpipe_offline()
            if preprocessors.use_tokenizer:
                if any(isinstance(pp, BaseTokenizer) for pp in plist):
                    self.Warning.tokenizer_ignored()
                if any(isinstance(pp, BaseTokenFilter) for pp in plist):
                    self.Warning.filtering_ignored()

    def apply(self):
        self.storeSpecificSettings()
        self.clear_messages()
        preprocessor = None
        try:
            preprocessor = self.buildpreproc()
        except FileNotFoundError:
            self.Error.file_not_found()
        except UnicodeError as e:
            self.Error.invalid_encoding(e)
        except StanfordPOSTaggerError as e:
            self.Error.stanford_tagger(e)
        except Exception as e:
            self.Error.unknown_error(str(e))

        self.start(self.apply_preprocessor, self.data, preprocessor)

    def apply_preprocessor(self, data: Optional[Corpus],
                           preprocessor: Optional[PreprocessorList],
                           state: TaskState) -> Result:
        def callback(i: float, status=""):
            state.set_progress_value(i * 100)
            if status:
                state.set_status(status)
            if state.is_interruption_requested():
                raise Exception

        pp_data = None
        msgs = []
        if data and preprocessor is not None:
            pp_data = preprocessor(data, wrap_callback(callback, end=0.9))
            if not pp_data.has_tokens():
                pp_data = BASE_TOKENIZER(
                    pp_data, wrap_callback(callback, start=0.9))
            if pp_data is not None and len(pp_data.dictionary) == 0:
                msgs.append(self.Warning.no_token_left)
                pp_data = None
        return Result(corpus=pp_data, msgs=msgs)

    def on_partial_result(self, result: Result):
        pass

    def on_done(self, result: Result):
        data, msgs = result.corpus, result.msgs
        for msg in msgs:
            msg()
        summary = len(data) if data else self.info.NoOutput
        detail = self.get_corpus_info(data) if data else ""
        self.info.set_output_summary(summary, detail)
        self.Outputs.corpus.send(data)
        self.update_preview(data)

    def on_exception(self, ex: Exception):
        if isinstance(ex, UDPipeStopIteration):
            self.Error.udpipe_offline_no_models()
        else:
            self.Error.unknown_error(ex)

    def update_preview(self, data):
        if data:
            try:
                tokens = next(data.ngrams_iterator(include_postags=True))
                self.preview = ", ".join(tokens[:5])
            except StopIteration:
                self.preview = ""

        else:
            self.preview = ""

    def workflowEnvChanged(self, key: str, *_):
        if key == "basedir":
            for i in range(self.preprocessormodel.rowCount()):
                item = self.preprocessormodel.item(i)
                params = item.data(ParametersRole)
                if params and "sw_path" in params and "lx_path" in params:
                    self.__update_filtering_params(params)
                    item.setData(params, ParametersRole)

        self.storedsettings = self.save(self.preprocessormodel)

    def onDeleteWidget(self):
        self.shutdown()
        super().onDeleteWidget()

    @staticmethod
    def get_corpus_info(corpus: Corpus) -> str:
        return f"Document count: {len(corpus)}\n" \
               f"Total tokens: {sum(map(len, corpus.tokens))}\n" \
               f"Total types: {len(corpus.dictionary)}"

    @classmethod
    def migrate_settings(cls, settings: Dict, version: int):
        if version < 2:
            settings["storedsettings"] = {"preprocessors": []}
            preprocessors = []

            # transformers
            if "transformers" in settings:
                transformers = settings.pop("transformers")
                if transformers["enabled"]:
                    params = {"methods": transformers["checked"]}
                    preprocessors.append(("preprocess.transform", params))

            # tokenizer
            if "tokenizer" in settings:
                tokenizer = settings.pop("tokenizer")
                if tokenizer["enabled"]:
                    params = {"method": tokenizer["method_index"],
                              "pattern": tokenizer["pattern"]}
                    preprocessors.append(("preprocess.tokenize", params))

            # normalizer
            if "normalizer" in settings:
                normalizer = settings.pop("normalizer")
                if normalizer["enabled"]:
                    params = {
                        "method": normalizer["method_index"],
                        "snowball_language": normalizer["snowball_language"],
                        "udpipe_language": normalizer["udpipe_language"],
                        "udpipe_tokenizer": normalizer["udpipe_tokenizer"]
                    }
                    preprocessors.append(("preprocess.normalize", params))

            # filtering
            if "filters" in settings:
                filters = settings.pop("filters")
                if filters["enabled"]:
                    def str_into_paths(label):
                        files = [RecentPath.create(path, []) for path in
                                 filters[label] if path != _DEFAULT_NONE]
                        return files[0] if files else None, files

                    sw_path, sw_list = str_into_paths("recent_sw_files")
                    lx_path, lx_list = str_into_paths("recent_lexicon_files")
                    params = {"methods": filters["checked"],
                              "language": filters["stopwords_language"],
                              "sw_path": sw_path, "sw_list": sw_list,
                              "lx_path": lx_path, "lx_list": lx_list,
                              "pattern": filters["pattern"],
                              "start": filters["min_df"],
                              "end": filters["max_df"],
                              "n_tokens": filters["keep_n"]}
                    preprocessors.append(("preprocess.filter", params))

            # n-grams
            if "ngrams_range" in settings:
                ngrams_range = settings.pop("ngrams_range")
                if ngrams_range["enabled"]:
                    params = {"start": ngrams_range["ngrams_range"][0],
                              "end": ngrams_range["ngrams_range"][1]}
                    preprocessors.append(("preprocess.ngrams", params))

            # POS tagger
            if "pos_tagger" in settings:
                pos_tagger = settings.pop("pos_tagger")
                if pos_tagger["enabled"]:
                    params = {"method": pos_tagger["method_index"]}
                    preprocessors.append(("tag.pos", params))

            settings["storedsettings"]["preprocessors"] = preprocessors

        if version < 3:
            preprocessors = settings["storedsettings"]["preprocessors"]
            for pp_name, pp_settings in preprocessors:
                if pp_name == "preprocess.filter":
                    start = pp_settings["start"]
                    end = pp_settings["end"]
                    if end <= 1:
                        pp_settings["rel_start"] = start
                        pp_settings["rel_end"] = end
                    else:
                        pp_settings["abs_start"] = start
                        pp_settings["abs_end"] = end
                    del pp_settings["start"]
                    del pp_settings["end"]


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWPreprocess).run(set_data=Corpus.from_file("deerwester"))
