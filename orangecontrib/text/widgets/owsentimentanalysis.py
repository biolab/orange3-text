from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QApplication, QGridLayout, QLabel

from Orange.widgets import gui, settings
from Orange.widgets.utils.signals import Input, Output
from Orange.widgets.widget import OWWidget
from orangecontrib.text import Corpus
from orangecontrib.text.sentiment import VaderSentiment, LiuHuSentiment, \
    MultiSentiment, SentimentDictionaries
from orangewidget.widget import Msg


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
        MultiSentiment
    ]
    LANG = ['English', 'Slovenian']
    MULTI_LANG = MultiSentiment.LANGS.keys()

    class Warning(OWWidget.Warning):
        senti_offline = Msg('No internet connection! Sentiment now only works '
                            'with local models.')
        senti_offline_no_lang = Msg('No internet connection and no local '
                                    'language resources are available.')

    def __init__(self):
        super().__init__()
        self.corpus = None

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

        self.form.addWidget(self.liu_hu, 0, 0, Qt.AlignLeft)
        self.form.addWidget(QLabel("Language:"), 0, 1, Qt.AlignRight)
        self.form.addWidget(self.liu_lang, 0, 2, Qt.AlignRight)
        self.form.addWidget(self.vader, 1, 0, Qt.AlignLeft)
        self.form.addWidget(self.multi_sent, 2, 0, Qt.AlignLeft)
        self.form.addWidget(QLabel("Language:"), 2, 1, Qt.AlignRight)
        self.form.addWidget(self.multi_box, 2, 2, Qt.AlignRight)

        self.senti_dict = SentimentDictionaries()
        self.update_multi_box()
        self.senti_online = self.senti_dict.online
        self.check_sentiment_online()

        ac = gui.auto_commit(self.controlArea, self, 'autocommit', 'Commit',
                             'Autocommit is on')
        ac.layout().insertSpacing(1, 8)

    def update_multi_box(self):
        if self.senti_dict.supported_languages:
            self.multi_box.clear()
            items = sorted([key for (key, value) in MultiSentiment.LANGS.items()
                            if value in self.senti_dict.supported_languages])
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
            if self.senti_dict.supported_languages:
                self.Warning.senti_offline()
            else:
                self.Warning.senti_offline_no_lang()

    @Inputs.corpus
    def set_corpus(self, data=None):
        self.corpus = data
        self.commit()

    def _method_changed(self):
        self.commit()

    def commit(self):
        if self.corpus is not None:
            self.Warning.senti_offline.clear()
            method = self.METHODS[self.method_idx]
            if self.method_idx == 0:
                out = method(language=self.liu_language).transform(self.corpus)
            elif self.method_idx == 2:
                if not self.senti_dict.online:
                    self.Warning.senti_offline()
                    self.update_multi_box()
                    return
                else:
                    out = method(language=self.multi_language).transform(self.corpus)
            else:
                out = method().transform(self.corpus)
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
