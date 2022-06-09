import unittest

from Orange.widgets.tests.base import WidgetTest

from orangecontrib.text.corpus import Corpus
from orangecontrib.text.widgets.owsimhash import OWSimhash


class TestOWSimhash(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWSimhash)
        self.corpus = Corpus.from_file("deerwester")

    def test_corpus(self):
        """
        Just basic test.
        GH-247
        """
        self.send_signal("Corpus", self.corpus)
        self.send_signal("Corpus", None)

    def test_output(self):
        self.send_signal("Corpus", self.corpus)
        output = self.get_output(self.widget.Outputs.corpus)
        self.assertEqual(len(self.corpus), len(output))
        self.assertEqual(64, len(output.domain.attributes))


if __name__ == "__main__":
    unittest.main()
