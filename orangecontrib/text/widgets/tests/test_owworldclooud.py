import unittest

from Orange.widgets.tests.base import WidgetTest
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.widgets.owwordcloud import OWWordCloud


class TestWorldCloudWidget(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWWordCloud)
        self.corpus = Corpus.from_file('deerwester')

    def test_data(self):
        """
        Just basic test.
        GH-244
        """
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.send_signal(self.widget.Inputs.corpus, None)

    def test_empty_data(self):
        """
        Widget crashes when receives zero length data.
        GH-244
        """
        self.assertTrue(self.widget.documents_info_str == "(no documents on input)")
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.assertTrue(self.widget.documents_info_str == "9 documents with 42 words")
        self.send_signal(self.widget.Inputs.corpus, self.corpus[:0])
        self.assertTrue(self.widget.documents_info_str == "(no documents on input)")


if __name__ == "__main__":
    unittest.main()
