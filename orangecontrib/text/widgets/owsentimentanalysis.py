from typing import List

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QGridLayout, QLabel

from Orange.widgets import gui, settings
from Orange.widgets.utils.signals import Input, Output
from Orange.widgets.widget import OWWidget, Msg
from orangecontrib.text import Corpus, preprocess
from orangecontrib.text.sentiment import VaderSentiment, LiuHuSentiment, \
    MultiSentiment, CustomDictionaries, SentiArt, MultisentimentDictionaries, \
    SentiArtDictionaries, LilahSentiment, LilahDictionaries
from orangecontrib.text.widgets.owpreprocess import FileLoader, _to_abspath
from orangecontrib.text.preprocess import PreprocessorList
from orangewidget.utils.filedialogs import RecentPath


class OWSentimentAnalysis(OWWidget):
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

    method_idx = settings.Setting(1)
    autocommit = settings.Setting(True)
    want_main_area = False
    resizing_enabled = False

    METHODS = [
        LiuHuSentiment,
        VaderSentiment,
        MultiSentiment,
        SentiArt,
        LilahSentiment,
        CustomDictionaries
    ]
    DEFAULT_NONE = None

    class Warning(OWWidget.Warning):
        senti_offline = Msg(
            "No internet connection! Sentiment now only works with local models."
        )
        senti_offline_no_lang = Msg(
            "No internet connection and no local language resources are available."
        )
        one_dict_only = Msg(f"Only one dictionary loaded.")
        no_dicts_loaded = Msg("No dictionaries loaded.")

    class Error(OWWidget.Error):
        lang_unsupported = Msg(
            "{} does not support the Corpus's language. Please select another method."
        )

    def __init__(self):
        super().__init__()
        self.corpus = None
        self.pp_corpus = None
        self.pos_file = None
        self.neg_file = None
        self.senti_dict = None

        box = gui.radioButtonsInBox(
            self.controlArea,
            self,
            "method_idx",
            [
                "Liu Hu",
                "Vader",
                "Multilingual sentiment",
                "SentiArt",
                "Lilah sentiment",
                "Custom dictionary",
            ],
            box="Method",
            callback=self._method_changed,
        )

        self.__posfile_loader = FileLoader()
        self.__posfile_loader.set_file_list()
        self.__posfile_loader.activated.connect(self.__pos_loader_activated)
        self.__posfile_loader.file_loaded.connect(self.__pos_loader_activated)

        self.__negfile_loader = FileLoader()
        self.__negfile_loader.set_file_list()
        self.__negfile_loader.activated.connect(self.__neg_loader_activated)
        self.__negfile_loader.file_loaded.connect(self.__neg_loader_activated)

        filegrid = QGridLayout()
        gui.indentedBox(box, orientation=filegrid)
        filegrid.addWidget(QLabel("Positive:"), 0, 0, Qt.AlignRight)
        filegrid.addWidget(self.__posfile_loader.file_combo, 0, 1)
        filegrid.addWidget(self.__posfile_loader.browse_btn, 0, 2)
        filegrid.addWidget(self.__posfile_loader.load_btn, 0, 3)
        filegrid.addWidget(QLabel("Negative:"), 1, 0, Qt.AlignRight)
        filegrid.addWidget(self.__negfile_loader.file_combo, 1, 1)
        filegrid.addWidget(self.__negfile_loader.browse_btn, 1, 2)
        filegrid.addWidget(self.__negfile_loader.load_btn, 1, 3)

        self.multi_dict = MultisentimentDictionaries()
        self.senti_dict = SentiArtDictionaries()
        self.lilah_dict = LilahDictionaries()
        self.online = self.multi_dict.online
        self.check_sentiment_online()

        gui.auto_commit(
            self.controlArea, self, "autocommit", "Commit", "Autocommit is on"
        )

    def __pos_loader_activated(self):
        cf = self.__posfile_loader.get_current_file()
        self.pos_file = cf.abspath if cf else None
        self._method_changed()

    def __neg_loader_activated(self):
        cf = self.__negfile_loader.get_current_file()
        self.neg_file = cf.abspath if cf else None
        self._method_changed()

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

    def check_sentiment_online(self):
        current_state = self.multi_dict.online
        if self.online != current_state:
            self.online = current_state

        self.Warning.senti_offline.clear()
        self.Warning.senti_offline_no_lang.clear()
        if not current_state and self.method_idx == 2:
            if self.multi_dict.supported_languages():
                self.Warning.senti_offline()
            else:
                self.Warning.senti_offline_no_lang()
        if not current_state and self.method_idx == 3:
            if self.senti_dict_dict.supported_languages():
                self.Warning.senti_offline()
            else:
                self.Warning.senti_offline_no_lang()
        if not current_state and self.method_idx == 4:
            if self.lilah_dict_dict.supported_languages():
                self.Warning.senti_offline()
            else:
                self.Warning.senti_offline_no_lang()

    @Inputs.corpus
    def set_corpus(self, data=None):
        self.corpus = data
        self.pp_corpus = None
        if self.corpus is not None:
            if not self.corpus.has_tokens():
                # create preprocessed corpus upon setting data to avoid
                # preprocessing at each method run
                pp_list = [preprocess.LowercaseTransformer(),
                           preprocess.WordPunctTokenizer()]
                self.pp_corpus = PreprocessorList(pp_list)(self.corpus)
            else:
                self.pp_corpus = self.corpus
        self.commit.now()

    def _method_changed(self):
        self.commit.deferred()

    def _compute_sentiment(self, method):
        corpus = self.pp_corpus
        if method.name == "Liu Hu":
            out = method(language=corpus.language).transform(corpus)
        elif method.name == "Multilingual Sentiment":
            if not self.senti_dict.online:
                self.Warning.senti_offline()
                return
            else:
                out = method(language=corpus.language).transform(corpus)
        elif method.name == "SentiArt":
            if not self.senti_dict.online:
                self.Warning.senti_offline()
                return
            out = method(language=corpus.language).transform(corpus)
        elif method.name == "Lilah Sentiment":
            if not self.lilah_dict.online:
                self.Warning.senti_offline()
                return
            out = method(language=corpus.language).transform(corpus)
        elif method.name == "Custom Dictionaries":
            out = method(self.pos_file, self.neg_file).transform(corpus)
            if (self.pos_file and not self.neg_file) or (
                self.neg_file and not self.pos_file
            ):
                self.Warning.one_dict_only()
            if not self.pos_file and not self.neg_file:
                self.Warning.no_dicts_loaded()
        else:
            out = method().transform(corpus)
        return out

    @gui.deferred
    def commit(self):
        self.Warning.senti_offline.clear()
        self.Warning.one_dict_only.clear()
        self.Warning.no_dicts_loaded.clear()
        self.Error.lang_unsupported.clear()

        if self.corpus is not None:
            method = self.METHODS[self.method_idx]
            if method.check_language(self.corpus.language):
                self.Outputs.corpus.send(self._compute_sentiment(method))
            else:
                self.Error.lang_unsupported(method.name)
                self.Outputs.corpus.send(None)
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


def main():
    from orangewidget.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWSentimentAnalysis).run(Corpus.from_file("book-excerpts")[:3])


if __name__ == '__main__':
    main()
