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
        self.send_signal("Corpus", self.corpus)
        self.send_signal("Corpus", None)


if __name__ == "__main__":
    unittest.main()
