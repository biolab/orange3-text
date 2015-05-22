import numpy as np
from PyQt4 import QtCore, QtGui
from sklearn.feature_extraction.text import CountVectorizer

from Orange.widgets.widget import OWWidget
from Orange.widgets.settings import Setting
from Orange.widgets import gui
from Orange.data import Table, DiscreteVariable, ContinuousVariable, StringVariable, Domain, Instance
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.preprocess import Preprocessor, Stemmer, Lemmatizer


class Output:
    DATA = "Data"

class OWBagOfWords(OWWidget):
    name = "Bag of words"
    description = """Generates a bag of words from the input corpus."""
    icon = "icons/BagOfWords.svg"

    inputs = [("Corpus", Corpus, "set_data")]
    outputs = [(Output.DATA, Table)]
    want_main_area = False

    def __init__(self):
        super().__init__()

        self.data = None

        # Settings.
        settings_box = gui.widgetBox(self.controlArea, "Basic options", addSpace=True)

    def set_data(self, data):
        if data is not None and isinstance(data, Corpus):
            self.data = data
            self.apply()

    def apply(self):
        new_table = None
        if self.data:
            documents = []
            for document in self.data.documents:
                if document.tokens:  # Some preprocessing was done.
                    documents.append(" ".join(document.tokens))
            if documents:
                cv = CountVectorizer()
                feats = cv.fit(documents)
                freqs = feats.transform(documents)
                freqs = freqs.toarray()

                print("Vocab: ", feats.vocabulary_)
                print("Freq: ", freqs)
                print("Features: ", feats.get_feature_names())

                # Generate the domain attributes.
                categories = np.array([d.category for d in self.data.documents])
                attr = [ContinuousVariable.make(f) for f in feats.get_feature_names()]
                class_var = DiscreteVariable.make("category", categories)
                # metas - 'text'(the text data in this input object)
                meta_attr = [StringVariable.make("text")]

                # Construct a new domain.
                domain = Domain(attr, class_var, metas=meta_attr)

                # Create the table.
                #new_table = Table.from_domain(domain, n_rows=0)
                new_table = Table.from_numpy(domain, freqs, class_var)
                new_table.save("/Users/david/Desktop/testna_tabela.tab")
                """
                # Create a new row to put into the table, for every item in the corpus.
                for i, document in enumerate(self.data.documents):
                    new_instance = Instance(domain, freqs[i, :])
                    new_instance["text"] = document.text
                    # Add the new instance.
                    new_table.append(new_instance)
                """
        self.send("Data", new_table)