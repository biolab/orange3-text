import unittest

from AnyQt.QtWidgets import QVBoxLayout

from orangecontrib.text import Corpus
from orangecontrib.text.vectorization import BowVectorizer
from orangecontrib.text.widgets.utils.owbasevectorizer import OWBaseVectorizer
from Orange.widgets.tests.base import WidgetTest


class TestableBaseVectWidget(OWBaseVectorizer):
    name = "TBV"
    Method = BowVectorizer

    def create_configuration_layout(self):
        return QVBoxLayout()

    def init_method(self):
        return self.Method()


class TestOWBaseVectorizer(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(TestableBaseVectWidget)
        self.corpus = Corpus.from_file('deerwester')

    def test_hide_attributes(self):
        self.send_signal("Corpus", self.corpus)
        self.assertTrue(all(f.attributes['hidden'] for f in
                            self.get_output("Corpus").domain.attributes))
        self.widget.controls.hidden_cb.setChecked(False)
        self.assertFalse(any(f.attributes['hidden'] for f in
                            self.get_output("Corpus").domain.attributes))
        new_corpus = Corpus.from_file('book-excerpts')[:10]
        self.send_signal("Corpus", new_corpus)
        self.assertFalse(any(f.attributes['hidden'] for f in
                            self.get_output("Corpus").domain.attributes))

    def test_no_data_on_input(self):
        self.send_signal("Corpus", self.corpus)
        self.assertTrue(self.get_output("Corpus"))

        self.send_signal("Corpus", None)
        self.assertFalse(self.get_output("Corpus"))


if __name__ == "__main__":
    unittest.main()