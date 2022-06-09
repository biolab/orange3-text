import unittest

from Orange.widgets.tests.base import WidgetTest

from orangecontrib.text.corpus import Corpus
from orangecontrib.text.widgets.owbagofwords import OWTBagOfWords


class TestBagOfWords(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWTBagOfWords)
        self.corpus = Corpus.from_file('deerwester')

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
        self.assertEqual(42, len(output.domain.attributes))

        self.send_signal("Corpus", self.corpus[:2])
        output = self.get_output(self.widget.Outputs.corpus)
        self.assertEqual(2, len(output))
        self.assertListEqual(
            # fmt: off
            [
                "a", "abc", "applications", "computer", "for", "human", "interface",
                "lab", "machine", "of", "opinion", "response", "survey", "system",
                "time", "user"
            ],
            # fmt: on
            [x.name for x in output.domain.attributes]
        )


if __name__ == "__main__":
    unittest.main()
