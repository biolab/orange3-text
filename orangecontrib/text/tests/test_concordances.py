from Orange.widgets.tests.base import WidgetTest
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.widgets.owconcordance import ConcordanceModel, OWConcordance


class TestConcordances(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWConcordance)
        self.corpus = Corpus.from_file('deerwester')

    def test_compute_indices(self):
        conc_model = ConcordanceModel()
        conc_model.set_data(self.corpus)
        conc_model.set_word("interface")
        self.assertEqual(len(conc_model.word_index), 2)

    def test_no_data(self):
        self.send_signal("Corpus", None)
        self.assertIsNone(self.get_output("Corpus"))
