from gensim import corpora, models, matutils

from Orange.widgets.widget import OWWidget
from Orange.widgets.settings import Setting
from Orange.widgets import gui
from Orange.data import Table, ContinuousVariable, Domain
from Orange.widgets.data.contexthandlers import DomainContextHandler
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.preprocess import Preprocessor


class Output:
    DATA = "Data"


class OWLDA(OWWidget):
    # Basic widget info
    name = "LDA"
    description = "Latent Dirichlet Allocation topic model."
    icon = "icons/mywidget.svg"

    settingsHandler = DomainContextHandler()

    # Input/output
    inputs = [("Corpus", Table, "set_data"),  # hack to accept input signals of type Table
              ("Preprocessor", Preprocessor, "set_preprocessor")]
    outputs = [(Output.DATA, Table)]
    want_main_area = False

    # Settings
    num_topics = Setting(5)
    use_tfidf = Setting(False)

    def __init__(self):
        super().__init__()

        self.corpus = None
        self.preprocessor = Preprocessor()

        # Info.
        info_box = gui.widgetBox(self.controlArea, "Info")
        self.info_label = gui.label(info_box, self, '')

        # Settings.
        topic_box = gui.widgetBox(self.controlArea, "Settings")
        self.num_topics_input = gui.spin(
            topic_box, self, "num_topics", label="Number of topics: ",
            minv=1, maxv=2 ** 31 - 1,)
        self.use_tfidf_button = gui.checkBox(topic_box, self, "use_tfidf",
                     "Use TF-IDF weighting")

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
        self.num_topics_input.setEnabled(got_corpus)
        self.commit.setEnabled(got_corpus)
        self.use_tfidf_button.setEnabled(got_corpus)

        ndoc = len(self.corpus) if got_corpus else "(None)"
        self.info_label.setText("Input text entries: {}".format(ndoc))

    def apply(self):
        self.refresh_gui()
        if self.corpus:
            # Preprocess
            preprocessed = self.preprocessor(self.corpus.documents)
            dictionary = corpora.Dictionary(preprocessed)
            corpus = [dictionary.doc2bow(t) for t in preprocessed]

            # LDA
            if self.use_tfidf:
                tfidf = models.TfidfModel(corpus)
                corpus = tfidf[corpus]

            lda = models.LdaModel(corpus, id2word=dictionary,
                                  num_topics=self.num_topics)
            corpus = lda[corpus]
            corpus_np = matutils.corpus2dense(corpus,
                                              num_terms=self.num_topics).T

            # Generate the new table.
            attr = [ContinuousVariable(f) for f in
                    lda.show_topics(num_topics=self.num_topics, num_words=3)]
            domain = Domain(attr,
                            self.corpus.domain.class_vars,
                            metas=self.corpus.domain.metas)
            print(corpus_np.shape[1], len(attr))
            new_table = Table.from_numpy(domain,
                                         corpus_np,
                                         Y=self.corpus._Y,
                                         metas=self.corpus.metas)

            self.send("Data", new_table)
        else:
            self.send("Data", None)
