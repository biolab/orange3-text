from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QApplication, QGridLayout, QLabel

from Orange.widgets import gui, settings
from Orange.widgets.utils.signals import Input, Output
from Orange.widgets.widget import OWWidget
from orangecontrib.text import Corpus
from orangecontrib.text.sentiment import Vader_Sentiment, Liu_Hu_Sentiment


class OWSentimentAnalysis(OWWidget):
    name = "Sentiment Analysis"
    description = "Predict sentiment from text."
    icon = "icons/SentimentAnalysis.svg"
    priority = 320

    class Inputs:
        corpus = Input("Corpus", Corpus)

    class Outputs:
        corpus = Output("Corpus", Corpus)

    method_idx = settings.Setting(1)
    autocommit = settings.Setting(True)
    language = settings.Setting('English')
    want_main_area = False
    resizing_enabled = False

    METHODS = [
        Liu_Hu_Sentiment,
        Vader_Sentiment
    ]
    LANG = ['English', 'Slovenian']

    def __init__(self):
        super().__init__()
        self.corpus = None

        form = QGridLayout()
        self.method_box = box = gui.radioButtonsInBox(
            self.controlArea, self, "method_idx", [], box="Method",
            orientation=form, callback=self._method_changed)
        self.liu_hu = gui.appendRadioButton(box, "Liu Hu", addToLayout=False)
        self.liu_lang = gui.comboBox(None, self, 'language',
                                     sendSelectedValue=True,
                                     items=self.LANG,
                                     callback=self._method_changed)
        self.vader = gui.appendRadioButton(box, "Vader", addToLayout=False)

        form.addWidget(self.liu_hu, 0, 0, Qt.AlignLeft)
        form.addWidget(QLabel("Language:"), 0, 1, Qt.AlignRight)
        form.addWidget(self.liu_lang, 0, 2, Qt.AlignRight)
        form.addWidget(self.vader, 1, 0, Qt.AlignLeft)

        ac = gui.auto_commit(self.controlArea, self, 'autocommit', 'Commit',
                             'Autocommit is on')
        ac.layout().insertSpacing(1, 8)

    @Inputs.corpus
    def set_corpus(self, data=None):
        self.corpus = data
        self.commit()

    def _method_changed(self):
        self.commit()

    def commit(self):
        if self.corpus is not None:
            method = self.METHODS[self.method_idx]
            if self.method_idx == 0:
                out = method(language=self.language).transform(self.corpus)
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
