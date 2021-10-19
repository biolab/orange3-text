import pickle

from orangecontrib.text import Corpus
from orangecontrib.text.widgets.owrelevanterms import OWRelevantTerms
from orangewidget.tests.base import WidgetTest


class TestStatisticsWidget(WidgetTest):
    def setUp(self) -> None:
        self.widget = self.create_widget(OWRelevantTerms)
        self.book_data = Corpus.from_file("book-excerpts")
        self._load_comparable_data()

    def _load_comparable_data(self) -> None:
        """
        Load preset deerwester data and LDA model. Save them to
        `self.deerwester` and `self.model`.
        """
        with open("data/LDAvis/deerwester.pkl", "rb") as c:
            self.deerwester = pickle.load(c)
        with open("data/LDAvis/model.pickle", "rb") as m:
            self.model = pickle.load(m)

    def test_send_data(self):
        """ Test with basic data, and empty data """
        self.send_signal(self.widget.Inputs.topics, self.book_data)
        self.assertEqual(len(self.book_data), len(self.widget.corpus))

        self.send_signal(self.widget.Inputs.corpus, None)
        self.assertIsNone(self.widget.corpus)
        self.widget.apply()
        self.assertIsNone(self.get_output(self.widget.Outputs.corpus))