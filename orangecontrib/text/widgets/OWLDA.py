from Orange.widgets.widget import OWWidget
from Orange.widgets.settings import Setting
from Orange.widgets import gui
from Orange.data import Table
from Orange.widgets.data.contexthandlers import DomainContextHandler
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.preprocess import Preprocessor
from orangecontrib.text.lda import LDA


class Output:
    DATA = "Data"
    TOPICS = "Topics"


class OWLDA(OWWidget):
    # Basic widget info
    name = "LDA"
    description = "Latent Dirichlet Allocation topic model."
    icon = "icons/LDA.svg"
    priority = 50

    settingsHandler = DomainContextHandler()

    # Input/output
    inputs = [("Corpus", Table, "set_data"),  # hack to accept input signals of type Table
              ("Preprocessor", Preprocessor, "set_preprocessor")]
    outputs = [(Output.DATA, Table),
               (Output.TOPICS, Table)]
    want_main_area = False

    # Settings
    num_topics = Setting(5)

    def __init__(self):
        super().__init__()

        self.corpus = None
        self.preprocessor = Preprocessor()

        # Info.
        info_box = gui.widgetBox(self.controlArea, "Info")
        self.info_label = gui.label(info_box, self, '')

        # Settings.
        topic_box = gui.widgetBox(self.controlArea, "Settings")
        hbox = gui.widgetBox(topic_box, orientation=0)
        self.topics_label = gui.label(hbox, self, 'Number of topics: ')
        self.topics_label.setMaximumSize(self.topics_label.sizeHint())
        self.topics_input = gui.spin(hbox, self, "num_topics",
                                     minv=1, maxv=2 ** 31 - 1,)

        # Commit button
        self.commit = gui.button(self.controlArea, self, "&Commit",
                                 callback=self.apply, default=True)

        self.refresh_gui()

    def set_preprocessor(self, data):
        if data is None:
            self.preprocessor = Preprocessor()
        else:
            self.preprocessor = data
        self.apply()

    def set_data(self, data=None):
        self.error(1)
        if data is None or isinstance(data, Corpus):
            self.corpus = data
        else:
            self.corpus = None
            self.error(1, 'Input should be of type Corpus')
        self.apply()

    def refresh_gui(self):
        got_corpus = self.corpus is not None
        self.commit.setEnabled(got_corpus)

        ndoc = len(self.corpus) if got_corpus else "(None)"
        self.info_label.setText("Input text entries: {}".format(ndoc))

    def progress(self, p):
        self.progressBarSet(p)

    def apply(self):
        self.refresh_gui()
        if self.corpus:
            preprocessed = self.preprocessor(self.corpus.documents)

            self.progressBarInit()
            lda = LDA(preprocessed, num_topics=self.num_topics, callback=self.progress)
            table = lda.insert_topics_into_corpus(self.corpus)
            topics = lda.get_topics_table()
            self.progressBarFinished()

            self.send(Output.DATA, table)
            self.send(Output.TOPICS, topics)
        else:
            self.send(Output.DATA, None)
            self.send(Output.TOPICS, None)
