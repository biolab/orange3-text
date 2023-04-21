from typing import List, Type, Tuple

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QGridLayout, QLabel

from Orange.widgets import gui, settings
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.utils.signals import Input, Output
from Orange.widgets.widget import OWWidget, Msg
from orangecontrib.text import Corpus, preprocess
from orangecontrib.text.language import ISO2LANG, LANG2ISO
from orangecontrib.text.sentiment import (
    VaderSentiment,
    LiuHuSentiment,
    MultiSentiment,
    CustomDictionaries,
    SentiArt,
    LilahSentiment,
    DictionaryNotFound,
    Sentiment,
)
from orangecontrib.text.widgets.owpreprocess import FileLoader, _to_abspath
from orangecontrib.text.preprocess import PreprocessorList
from orangewidget.utils.filedialogs import RecentPath


class OWSentimentAnalysis(OWWidget, ConcurrentWidgetMixin):
    name = "Sentiment Analysis"
    description = "Compute sentiment from text."
    icon = "icons/SentimentAnalysis.svg"
    priority = 320
    keywords = ["emotion"]

    class Inputs:
        corpus = Input("Corpus", Corpus)

    class Outputs:
        corpus = Output("Corpus", Corpus)

    settings_version = 1
    want_main_area = False
    resizing_enabled = False

    method_idx: int = settings.Setting(1)
    autocommit: bool = settings.Setting(True)
    liu_language: str = settings.Setting(
        ISO2LANG[LiuHuSentiment.DEFAULT_LANG], schema_only=True
    )
    multi_language: str = settings.Setting(
        ISO2LANG[MultiSentiment.DEFAULT_LANG], schema_only=True
    )
    senti_language: str = settings.Setting(
        ISO2LANG[SentiArt.DEFAULT_LANG], schema_only=True
    )
    lilah_language: str = settings.Setting(
        ISO2LANG[LilahSentiment.DEFAULT_LANG], schema_only=True
    )

    METHODS = [
        LiuHuSentiment,
        VaderSentiment,
        MultiSentiment,
        SentiArt,
        LilahSentiment,
        CustomDictionaries
    ]

    class Warning(OWWidget.Warning):
        one_dict_only = Msg(f"Only one dictionary loaded.")
        no_dicts_loaded = Msg("No dictionaries loaded.")

    class Error(OWWidget.Error):
        offline = Msg(
            "Sentiment cannot be computed since you are offline and the "
            "required dictionary is unavailable locally."
        )

    def __init__(self):
        super().__init__()
        ConcurrentWidgetMixin.__init__(self)
        self.corpus = None
        self.pp_corpus = None
        self.pos_file = None
        self.neg_file = None
        self.senti_dict = None

        # languages from workflow should be retained when data on input
        self.__pend_liu_lang = self.liu_language
        self.__pend_multi_lang = self.multi_language
        self.__pend_senti_lang = self.senti_language
        self.__pend_lilah_lang = self.lilah_language

        self.form = QGridLayout()
        self.method_box = box = gui.radioButtonsInBox(
            self.controlArea, self, "method_idx", [], box="Method",
            orientation=self.form, callback=self._method_changed)
        self.liu_hu = gui.appendRadioButton(box, "Liu Hu", addToLayout=False)
        self.liu_lang = gui.comboBox(
            None,
            self,
            "liu_language",
            sendSelectedValue=True,
            contentsLength=10,
            items=[ISO2LANG[lg] for lg in LiuHuSentiment.LANGUAGES],
            callback=self._method_changed,
        )
        self.vader = gui.appendRadioButton(box, "Vader", addToLayout=False)
        self.multi_sent = gui.appendRadioButton(
            box, "Multilingual " "sentiment", addToLayout=False
        )
        self.multi_box = gui.comboBox(
            None,
            self,
            "multi_language",
            sendSelectedValue=True,
            contentsLength=10,
            items=[ISO2LANG[lg] for lg in MultiSentiment.LANGUAGES],
            callback=self._method_changed,
        )
        self.senti_art = gui.appendRadioButton(box, "SentiArt", addToLayout=False)
        self.senti_box = gui.comboBox(
            None,
            self,
            "senti_language",
            sendSelectedValue=True,
            contentsLength=10,
            items=[ISO2LANG[lg] for lg in SentiArt.LANGUAGES],
            callback=self._method_changed,
        )
        self.lilah_sent = gui.appendRadioButton(
            box, "Lilah sentiment", addToLayout=False
        )
        self.lilah_box = gui.comboBox(
            None,
            self,
            "lilah_language",
            sendSelectedValue=True,
            contentsLength=10,
            items=[ISO2LANG[lg] for lg in LilahSentiment.LANGUAGES],
            callback=self._method_changed,
        )
        self.custom_list = gui.appendRadioButton(
            box, "Custom dictionary", addToLayout=False
        )
        self.__posfile_loader = FileLoader()
        self.__posfile_loader.set_file_list()
        self.__posfile_loader.activated.connect(self.__pos_loader_activated)
        self.__posfile_loader.file_loaded.connect(self.__pos_loader_activated)

        self.__negfile_loader = FileLoader()
        self.__negfile_loader.set_file_list()
        self.__negfile_loader.activated.connect(self.__neg_loader_activated)
        self.__negfile_loader.file_loaded.connect(self.__neg_loader_activated)

        self.form.addWidget(self.liu_hu, 0, 0, Qt.AlignLeft)
        self.form.addWidget(QLabel("Language:"), 0, 1, Qt.AlignRight)
        self.form.addWidget(self.liu_lang, 0, 2, Qt.AlignRight)
        self.form.addWidget(self.vader, 1, 0, Qt.AlignLeft)
        self.form.addWidget(QLabel("Language:"), 1, 1, Qt.AlignRight)
        self.form.addWidget(QLabel("   English"), 1, 2, Qt.AlignLeft)
        self.form.addWidget(self.multi_sent, 2, 0, Qt.AlignLeft)
        self.form.addWidget(QLabel("Language:"), 2, 1, Qt.AlignRight)
        self.form.addWidget(self.multi_box, 2, 2, Qt.AlignRight)
        self.form.addWidget(self.senti_art, 3, 0, Qt.AlignLeft)
        self.form.addWidget(QLabel("Language:"), 3, 1, Qt.AlignRight)
        self.form.addWidget(self.senti_box, 3, 2, Qt.AlignRight)
        self.form.addWidget(self.lilah_sent, 4, 0, Qt.AlignLeft)
        self.form.addWidget(QLabel("Language:"), 4, 1, Qt.AlignRight)
        self.form.addWidget(self.lilah_box, 4, 2, Qt.AlignRight)
        self.form.addWidget(self.custom_list, 5, 0, Qt.AlignLeft)
        self.filegrid = QGridLayout()
        self.form.addLayout(self.filegrid, 6, 0, 1, 3)
        self.filegrid.addWidget(QLabel("Positive:"), 0, 0, Qt.AlignRight)
        self.filegrid.addWidget(self.__posfile_loader.file_combo, 0, 1)
        self.filegrid.addWidget(self.__posfile_loader.browse_btn, 0, 2)
        self.filegrid.addWidget(self.__posfile_loader.load_btn, 0, 3)
        self.filegrid.addWidget(QLabel("Negative:"), 1, 0, Qt.AlignRight)
        self.filegrid.addWidget(self.__negfile_loader.file_combo, 1, 1)
        self.filegrid.addWidget(self.__negfile_loader.browse_btn, 1, 2)
        self.filegrid.addWidget(self.__negfile_loader.load_btn, 1, 3)

        gui.auto_apply(self.buttonsArea, self, "autocommit")

    def __pos_loader_activated(self):
        cf = self.__posfile_loader.get_current_file()
        self.pos_file = cf.abspath if cf else None
        self.self.commit.deferred()

    def __neg_loader_activated(self):
        cf = self.__negfile_loader.get_current_file()
        self.neg_file = cf.abspath if cf else None
        self.self.commit.deferred()

    def __set_pos_path(self, path: RecentPath, paths: List[RecentPath] = []):
        self._posfile_loader.recent_paths = paths
        self.__posfile_loader.set_file_list()
        self.__posfile_loader.set_current_file(_to_abspath(path))
        self.pos_file = self.__posfile_loader.get_current_file()

    def __set_lx_path(self, path: RecentPath, paths: List[RecentPath] = []):
        self.__negfile_loader.recent_paths = paths
        self.__negfile_loader.set_file_list()
        self.__negfile_loader.set_current_file(_to_abspath(path))
        self.neg_file = self.__negfile_loader.get_current_file()

    @Inputs.corpus
    def set_corpus(self, corpus):
        self.pp_corpus = None
        if corpus is not None:
            if not corpus.has_tokens():
                # create preprocessed corpus upon setting data to avoid
                # preprocessing at each method run
                pp_list = [preprocess.LowercaseTransformer(),
                           preprocess.WordPunctTokenizer()]
                self.pp_corpus = PreprocessorList(pp_list)(corpus)
            else:
                self.pp_corpus = corpus
        self.__set_language_settings()
        self.commit.now()

    def __set_language_settings(self):
        settings_ = (
            (self.__pend_liu_lang, "liu_language", LiuHuSentiment),
            (self.__pend_multi_lang, "multi_language", MultiSentiment),
            (self.__pend_senti_lang, "senti_language", SentiArt),
            (self.__pend_lilah_lang, "lilah_language", LilahSentiment),
        )

        for l_pending, l_setting, model in settings_:
            if self.pp_corpus and self.pp_corpus.language in model.LANGUAGES:
                setattr(self, l_setting, ISO2LANG[self.pp_corpus.language])
            else:
                # if Corpus's language not supported use default language
                setattr(self, l_setting, ISO2LANG[model.DEFAULT_LANG])

            # when workflow loaded use language saved in workflow
            if l_pending is not None:
                setattr(self, l_setting, l_pending)

        self.__pend_liu_lang = None
        self.__pend_multi_lang = None
        self.__pend_senti_lang = None
        self.__pend_lilah_lang = None

    def _method_changed(self):
        self.commit.deferred()

    def _compute_sentiment(self):
        method = self.METHODS[self.method_idx]
        kwargs = {}
        if method.name == "Liu Hu":
            kwargs = dict(language=LANG2ISO[self.liu_language])
        elif method.name == "Multilingual Sentiment":
            kwargs = dict(language=LANG2ISO[self.multi_language])
        elif method.name == "SentiArt":
            kwargs = dict(language=LANG2ISO[self.senti_language])
        elif method.name == "LiLaH Sentiment":
            kwargs = dict(language=LANG2ISO[self.lilah_language])
        elif method.name == "Custom Dictionaries":
            kwargs = dict(pos=self.pos_file, neg=self.neg_file)
            if bool(self.pos_file) != bool(self.neg_file):  # xor: one of them None
                self.Warning.one_dict_only()
            elif not self.pos_file and not self.neg_file:
                self.Warning.no_dicts_loaded()

        self.start(self._run, self.pp_corpus, method, kwargs)

    @staticmethod
    def _run(
        corpus: Corpus,
        method: Type[Sentiment],
        kwargs: Tuple[str],
        task_state: TaskState,
    ) -> Corpus:
        def callback(i: float):
            task_state.set_progress_value(i * 100)
            if task_state.is_interruption_requested():
                raise Exception

        method = method(**kwargs)
        return method.transform(corpus, callback=callback)

    def on_done(self, corpus: Corpus):
        self.Outputs.corpus.send(corpus)

    def on_exception(self, ex):
        if isinstance(ex, DictionaryNotFound):
            self.Error.offline()
        else:
            raise ex
        self.Outputs.corpus.send(None)

    @gui.deferred
    def commit(self):
        self.Error.offline.clear()
        self.Warning.one_dict_only.clear()
        self.Warning.no_dicts_loaded.clear()

        if self.pp_corpus is not None:
            self._compute_sentiment()
        else:
            self.Outputs.corpus.send(None)

    def send_report(self):
        self.report_items((
            ('Method', self.METHODS[self.method_idx].name),
        ))

    @classmethod
    def migrate_settings(cls, settings, version):
        if version is None:
            # in old version Custom Dictionaries were at id 4
            method_idx = settings["method_idx"]
            if method_idx == 4:
                settings["metric_idx"] = 5


if __name__ == '__main__':
    from orangewidget.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWSentimentAnalysis).run(Corpus.from_file("book-excerpts")[:3])
