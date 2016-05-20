from PyQt4 import QtGui

from Orange.widgets.widget import OWWidget
from Orange.widgets.settings import Setting
from Orange.widgets import gui
from Orange.data import Table
from Orange.widgets.data.contexthandlers import DomainContextHandler
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.preprocess import Preprocessor
from orangecontrib.text.lda import LDA
from orangecontrib.text.topics import Topics


class Output:
    DATA = "Data"
    TOPICS = "Topics"


class OWLDA(OWWidget):
    # Basic widget info
    name = "Topic Modelling"
    description = "Topic modelling with Latent Dirichlet Allocation."
    icon = "icons/LDA.svg"
    priority = 50

    settingsHandler = DomainContextHandler()

    # Input/output
    inputs = [("Corpus", Corpus, "set_data")]
    outputs = [(Output.DATA, Table),
               (Output.TOPICS, Topics)]
    want_main_area = True

    # Settings
    num_topics = Setting(5)

    def __init__(self):
        super().__init__()

        self.lda = None
        self.corpus = None

        # Info.
        info_box = gui.widgetBox(self.controlArea, "Info")
        self.info_label = gui.label(info_box, self, '')

        # Settings.
        topic_box = gui.widgetBox(self.controlArea, "Settings")
        hbox = gui.widgetBox(topic_box, orientation=0)
        self.topics_label = gui.label(hbox, self, 'Number of topics: ')
        self.topics_label.setMaximumSize(self.topics_label.sizeHint())
        self.topics_input = gui.spin(hbox, self, "num_topics",
                                     minv=1, maxv=2 ** 31 - 1,
                                     callback=self.num_topics_changed)

        # Commit button
        self.commit = gui.button(self.controlArea, self, "&Apply",
                                 callback=self.apply, default=True)
        self.commit.setEnabled(False)
        gui.rubber(self.controlArea)

        # Topics description
        self.cols = ['Topic', 'Topic keywords']
        self.topic_desc = QtGui.QTreeWidget()
        self.topic_desc.setColumnCount(len(self.cols))
        self.topic_desc.setHeaderLabels(self.cols)
        #self.topic_desc.setSelectionMode(QtGui.QTreeView.ExtendedSelection)
        self.topic_desc.itemSelectionChanged.connect(self.selected_topic_changed)
        for i in range(len(self.cols)):
            self.topic_desc.resizeColumnToContents(i)
        self.mainArea.layout().addWidget(self.topic_desc)

        self.refresh_gui()

    def set_data(self, data=None):
        self.corpus = data
        self.apply()

    def refresh_gui(self):
        got_corpus = self.corpus is not None
        ndoc = len(self.corpus) if got_corpus else "(None)"
        self.info_label.setText("Input text entries: {}".format(ndoc))

    def update_topics(self):
        self.topic_desc.clear()
        for i in range(self.lda.num_topics):
            words = self.lda.get_top_words_by_id(i)
            it = LDATreeWidgetItem(i, words, self.topic_desc)
            self.topic_desc.addTopLevelItem(it)
        for i in range(2):
            self.topic_desc.resizeColumnToContents(i)

    def num_topics_changed(self):
        if self.corpus is None or \
                (self.lda is not None and self.lda.num_topics == self.num_topics):
            self.commit.setEnabled(False)
        else:
            self.commit.setEnabled(True)

    def selected_topic_changed(self):
        selected = self.topic_desc.selectedItems()
        if selected:
            self.send_topic_by_id(selected[0].topic_id)

    def send_topic_by_id(self, topic_id):
        self.topic_desc.setCurrentItem(self.topic_desc.topLevelItem(topic_id))
        self.send(Output.TOPICS, self.lda.get_topics_table_by_id(topic_id))

    def enabled(self, bool):
        self.topics_input.setEnabled(bool)

    def progress(self, p):
        self.progressBarSet(p)

    def apply(self):
        self.topic_desc.clear()
        self.refresh_gui()
        if self.corpus:
            self.commit.setEnabled(False)
            self.enabled(False)

            with self.progressBar(len(self.corpus)) as bar:
                self.lda = LDA(self.corpus.tokens, num_topics=self.num_topics,
                               callback=bar.advance)
                table = self.lda.insert_topics_into_corpus(self.corpus)
                self.update_topics()

            self.send(Output.DATA, table)
            self.send_topic_by_id(0)

            self.enabled(True)
        else:
            self.send(Output.DATA, None)
            self.send(Output.TOPICS, None)


class LDATreeWidgetItem(QtGui.QTreeWidgetItem):
    def __init__(self, topic_id, words, parent):
        super().__init__(parent)
        self.topic_id = topic_id
        self.words = words

        self.setText(0, '{:d}'.format(topic_id+1))
        self.setText(1, ', '.join(words))
