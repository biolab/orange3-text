from AnyQt.QtWidgets import QApplication

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
    want_main_area = False
    resizing_enabled = False

    def __init__(self):
        super().__init__()
        self.METHODS = [
            Liu_Hu_Sentiment(),
            Vader_Sentiment()
        ]
        self.corpus = None

        gui.radioButtons(self.controlArea, self, "method_idx", box="Method",
                         btnLabels=[m.name for m in self.METHODS],
                         callback=self._method_changed)

        ac = gui.auto_commit(self.controlArea, self, 'autocommit', 'Commit',
                             'Autocommit is on')
        ac.layout().insertWidget(0, self.report_button)
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
            out = method.transform(self.corpus)
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
    corpus = Corpus.from_file('bookexcerpts')
    corpus = corpus[:3]
    widget.set_corpus(corpus)
    widget.show()
    app.exec()

if __name__ == '__main__':
    main()
