import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from Orange.widgets.widget import OWWidget
from Orange.widgets.settings import Setting
from Orange.widgets import gui
from Orange.data import Table, DiscreteVariable, ContinuousVariable, Domain, StringVariable
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.preprocess import Preprocessor


class Output:
    DATA = "Data"


class OWBagOfWords(OWWidget):
    # Basic widget info
    name = "Bag of words"
    description = "Generates a bag of words from the input corpus."
    icon = "icons/BagOfWords.svg"
    priority = 40

    # Input/output
    inputs = [("Corpus", Table, "set_corpus"),  # hack to accept input signals of type Table
              ("Preprocessor", Preprocessor, "set_preprocessor")]
    outputs = [(Output.DATA, Table)]
    want_main_area = False

    _normalization_options = ['(none)', 'L1 (sum of elements)', 'L2 (Euclidean)']

    # Settings
    use_tfidf = Setting(False)
    normalization_type = Setting(0)

    def __init__(self):
        super().__init__()

        self.corpus = None
        self.preprocessor = None
        self.normalization = None

        # Pre-processing info.
        pp_info_box = gui.widgetBox(self.controlArea, "Pre-processing info")

        pp_info = "Includes punctuation: {}\n" \
                  "Lowercase: {}\n" \
                  "Transformation: {}\n" \
                  "Stop words removed: {}\n" \
                  "TF-IDF performed: {}\n" \
                  "Normalization: {}"\
            .format(False, False, None, None,
                    self.use_tfidf, self.normalization)
        self.pp_info_label = gui.label(pp_info_box, self, pp_info)

        # TF-IDF.
        tfidf_box = gui.widgetBox(self.controlArea, "TF-IDF", addSpace=False)

        self.tfidf_chbox = gui.checkBox(tfidf_box, self, "use_tfidf", "Use TF-IDF")
        self.tfidf_chbox.stateChanged.connect(self._tfidf_changed)
        ibox = gui.indentedBox(tfidf_box)
        self.norm_combo = gui.comboBox(ibox, self, 'normalization_type',
                                       items=self._normalization_options,
                                       label="Normalization:")
        self.norm_combo.activated[int].connect(self._select_normalization)
        self.norm_combo.setEnabled(self.use_tfidf)

        gui.button(self.controlArea, self, "&Apply", callback=self.apply, default=True)

    def set_preprocessor(self, data):
        self.preprocessor = data

    def set_corpus(self, data):
        self.error(1)
        if data is None or isinstance(data, Corpus):
            self.corpus = data
        else:
            self.corpus = None
            self.error(1, 'Input should be of type Corpus')
        self.apply()

    def apply(self):
        # TODO Move the logic to the scripting module.
        new_table = None
        if self.corpus:
            if self.preprocessor:
                if self.use_tfidf:
                    cv = TfidfVectorizer(lowercase=self.preprocessor.lowercase,
                                         stop_words=self.preprocessor.stop_words,
                                         preprocessor=self.preprocessor.transformation,
                                         norm=self.normalization)
                else:
                    cv = CountVectorizer(lowercase=self.preprocessor.lowercase,
                                         stop_words=self.preprocessor.stop_words,
                                         preprocessor=self.preprocessor.transformation)

                pp_info_tag = "Includes punctuation: {}\nLowercase: {}\n" \
                              "Transformation: {}\n" \
                              "Stop words removed: {}\n" \
                              "TF-IDF performed: {}\n" \
                              "Normalization: {}"\
                    .format(self.preprocessor.incl_punct, self.preprocessor.lowercase,
                            self.preprocessor.transformation.name,
                            self.preprocessor.stop_words, self.use_tfidf,
                            self._normalization_options[self.normalization_type])
                self.pp_info_label.setText(pp_info_tag)
            else:
                cv = CountVectorizer(lowercase=True,
                                     binary=True)

            documents = self.corpus.documents
            feats = cv.fit(documents)  # Features.
            freqs = feats.transform(documents).toarray()  # Frequencies.

            # Generate the domain attributes.
            attr = [ContinuousVariable(f) for f in feats.get_feature_names()]

            # Construct a new domain.
            domain = Domain(attr, self.corpus.domain.class_vars, metas=self.corpus.domain.metas)

            # Create the table.
            new_table = Table.from_numpy(domain, freqs, Y=self.corpus._Y, metas=self.corpus.metas)
        self.send("Data", new_table)

    def _select_normalization(self, n):
        if n > 0:
            self.normalization = self._normalization_options[n].split(" ")[0].lower()
        else:
            self.normalization = None

    def _tfidf_changed(self):
        self.norm_combo.setEnabled(self.use_tfidf)
