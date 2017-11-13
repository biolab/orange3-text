from Orange.widgets.tests.base import WidgetTest
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.widgets.owpreprocess import OWPreprocess


class TestConcordanceWidget(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWPreprocess)
        self.corpus = Corpus.from_file('deerwester')

    def test_set_corpus(self):
        """
        Just basic test.
        """
        self.send_signal("Corpus", self.corpus)

    def test_multiple_instances(self):
        # GH-327
        widget2 = self.create_widget(OWPreprocess)
        widget3 = self.create_widget(OWPreprocess)
