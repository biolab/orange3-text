from PyQt4.QtGui import QApplication

from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget
from orangecontrib.text.bagofowords import BagOfWords
from orangecontrib.text.corpus import Corpus


class Input:
    CORPUS = 'Corpus'


class Output:
    CORPUS = 'Corpus'


class OWBagOfWords(OWWidget):
    name = 'Bag of Words'
    description = 'Generates a bag of words from the input corpus.'
    icon = 'icons/BagOfWords.svg'
    priority = 40

    # Input/output
    inputs = [
        (Input.CORPUS, Corpus, 'set_data'),
    ]
    outputs = [
        (Output.CORPUS, Corpus)
    ]

    want_main_area = False

    # Settings
    autocommit = Setting(True)
    use_tfidf = Setting(False)

    def __init__(self):
        super().__init__()

        self.corpus = None

        # Info box.
        info_box = gui.widgetBox(
                self.controlArea,
                'Info',
        )
        bow_info = 'No info available.'
        self.info_label = gui.label(info_box, self, bow_info)

        # TF-IDF.
        tfidf_box = gui.widgetBox(
                self.controlArea,
                'Settings',
        )

        self.tfidf_checkbox = gui.checkBox(
                tfidf_box,
                self,
                'use_tfidf',
                'Use TF-IDF'
        )
        self.tfidf_checkbox.stateChanged.connect(self.tfidf_changed)

        gui.auto_commit(
                self.controlArea,
                self,
                'autocommit',
                'Commit',
                box=False
        )

    def set_data(self, data):
        self.corpus = data
        self.commit()

    def commit(self):
        self.apply()

    def apply(self):
        self.error(0, '')
        if self.corpus is not None:
            # BoW uses 4 steps ATM.
            # Update this when a better solution is available.
            with self.progressBar(4) as progress_bar:

                bag_of_words = BagOfWords(
                        progress_callback=progress_bar.advance,
                        error_callback=self.show_errors
                )
                bow_corpus = bag_of_words(
                        self.corpus,
                        use_tfidf=self.use_tfidf
                )

                self.update_info(bag_of_words.vocabulary)
                self.send(Output.CORPUS, bow_corpus)

    def show_errors(self, error):
        self.error(0, '')
        self.error(0, str(error))

    def update_info(self, new_vocabulary):
        if new_vocabulary is None:
            new_info = 'No info available.'
        else:
            new_info = '{} documents.\n{} unique tokens.'.format(
                new_vocabulary.num_docs,
                len(new_vocabulary),
            )
        self.info_label.setText(new_info)

    def tfidf_changed(self):
        self.commit()

if __name__ == '__main__':
    app = QApplication([])
    widget = OWBagOfWords()
    widget.show()
    corpus = Corpus.from_file('bookexcerpts')
    widget.set_data(corpus)
    app.exec()
