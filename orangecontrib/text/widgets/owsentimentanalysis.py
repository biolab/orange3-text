from typing import List

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QApplication, QGridLayout, QLabel

from Orange.widgets import gui, settings
from Orange.widgets.utils.signals import Input, Output
from Orange.widgets.widget import OWWidget, Msg
from orangecontrib.text import Corpus, preprocess
from orangecontrib.text.sentiment import VaderSentiment, LiuHuSentiment, \
    MultiSentiment, CustomDictionaries, MultisentimentDictionaries
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

    method_idx = settings.Setting(1)
    autocommit = settings.Setting(True)
    liu_language = settings.Setting('English')
    multi_language = settings.Setting('English')
    want_main_area = False
    resizing_enabled = False

    METHODS = [
        LiuHuSentiment,
        VaderSentiment,
        MultiSentiment,
        CustomDictionaries
    ]
    LANG = ['English', 'Slovenian']
    MULTI_LANG = MultiSentiment.LANGS.keys()
    DEFAULT_NONE = None

    class Warning(OWWidget.Warning):
        senti_offline = Msg('No internet connection! Sentiment now only works '
                            'with local models.')
        senti_offline_no_lang = Msg('No internet connection and no local '
                                    'language resources are available.')
        one_dict_only = Msg(f'Only one dictionary loaded.')
        no_dicts_loaded = Msg('No dictionaries loaded.')

    def __init__(self):
        super().__init__()
        self.corpus = None
        self.pp_corpus = None
        self.pos_file = None
        self.neg_file = None

        self.form = QGridLayout()
        self.method_box = box = gui.radioButtonsInBox(
            self.controlArea, self, "method_idx", [], box="Method",
            orientation=self.form, callback=self._method_changed)
        self.liu_hu = gui.appendRadioButton(box, "Liu Hu", addToLayout=False)
        self.liu_lang = gui.comboBox(None, self, 'liu_language',
                                     sendSelectedValue=True,
                                     contentsLength=10,
                                     items=self.LANG,
                                     callback=self._method_changed)
        self.vader = gui.appendRadioButton(box, "Vader", addToLayout=False)
        self.multi_sent = gui.appendRadioButton(box, "Multilingual "
                                                     "sentiment",
                                                addToLayout=False)
        self.multi_box = gui.comboBox(None, self, 'multi_language',
                                      sendSelectedValue=True,
                                      contentsLength=10, items=[''],
                                      callback=self._method_changed)
        self.custom_list = gui.appendRadioButton(box, "Custom dictionary",
                                                 addToLayout=False)

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
        self.form.addWidget(self.multi_sent, 2, 0, Qt.AlignLeft)
        self.form.addWidget(QLabel("Language:"), 2, 1, Qt.AlignRight)
        self.form.addWidget(self.multi_box, 2, 2, Qt.AlignRight)
        self.form.addWidget(self.custom_list, 3, 0, Qt.AlignLeft)
        self.filegrid = QGridLayout()
        self.form.addLayout(self.filegrid, 4, 0, 1, 3)
        self.filegrid.addWidget(QLabel("Positive:"), 0, 0, Qt.AlignRight)
        self.filegrid.addWidget(self.__posfile_loader.file_combo, 0, 1)
        self.filegrid.addWidget(self.__posfile_loader.browse_btn, 0, 2)
        self.filegrid.addWidget(self.__posfile_loader.load_btn, 0, 3)
        self.filegrid.addWidget(QLabel("Negative:"), 1, 0, Qt.AlignRight)
        self.filegrid.addWidget(self.__negfile_loader.file_combo, 1, 1)
        self.filegrid.addWidget(self.__negfile_loader.browse_btn, 1, 2)
        self.filegrid.addWidget(self.__negfile_loader.load_btn, 1, 3)

        self.senti_dict = MultisentimentDictionaries()
        self.update_multi_box()
        self.senti_online = self.senti_dict.online
        self.check_sentiment_online()

        ac = gui.auto_commit(self.controlArea, self, 'autocommit', 'Commit',
                             'Autocommit is on')
        ac.layout().insertSpacing(1, 8)

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

    def update_multi_box(self):
        if self.senti_dict.supported_languages():
            self.multi_box.clear()
            items = sorted([key for (key, value) in MultiSentiment.LANGS.items()
                            if value in self.senti_dict.supported_languages()])
            self.multi_box.addItems(items)
            self.multi_box.setCurrentIndex(items.index("English"))

    def check_sentiment_online(self):
        current_state = self.senti_dict.online
        if self.senti_online != current_state:
            self.update_multi_box()
            self.senti_online = current_state

        self.Warning.senti_offline.clear()
        self.Warning.senti_offline_no_lang.clear()
        if not current_state and self.method_idx == 2:
            if self.senti_dict.supported_languages():
                self.Warning.senti_offline()
            else:
                self.Warning.senti_offline_no_lang()


    @Inputs.corpus
    def set_corpus(self, data=None):
        self.corpus = data
        # create preprocessed corpus upon setting data to avoid preprocessing
        # at each method run
        pp_list = [preprocess.LowercaseTransformer(),
                   preprocess.WordPunctTokenizer()]
        self.pp_corpus = PreprocessorList(pp_list)(self.corpus)
        self.commit()

    def _method_changed(self):
        self.commit()

    def commit(self):
        if self.corpus is not None:
            self.Warning.senti_offline.clear()
            self.Warning.one_dict_only.clear()
            self.Warning.no_dicts_loaded.clear()
            method = self.METHODS[self.method_idx]
            corpus = self.pp_corpus
            if method.name == 'Liu Hu':
                out = method(language=self.liu_language).transform(corpus)
            elif method.name == 'Multilingual Sentiment':
                if not self.senti_dict.online:
                    self.Warning.senti_offline()
                    self.update_box(self.multi_box, self.multi_dict, MultiSentiment)
                    return
                else:
                    out = method(language=self.multi_language).transform(corpus)
            elif method.name == 'Custom Dictionaries':
                out = method(self.pos_file, self.neg_file).transform(corpus)
                if (self.pos_file and not self.neg_file) or \
                    (self.neg_file and not self.pos_file):
                    self.Warning.one_dict_only()
                if not self.pos_file and not self.neg_file:
                    self.Warning.no_dicts_loaded()
            else:
                out = method().transform(corpus)
            self.Outputs.corpus.send(out)
        else:
            self.Outputs.corpus.send(None)

    def send_report(self):
        self.report_items((
            ('Method', self.METHODS[self.method_idx].name),
        ))


def main():
    app = QApplication([])
    widget = OWSentimentAnalysis()
    corpus = Corpus.from_file('book-excerpts')
    corpus = corpus[:3]
    widget.set_corpus(corpus)
    widget.show()
    app.exec()


if __name__ == '__main__':
    main()
