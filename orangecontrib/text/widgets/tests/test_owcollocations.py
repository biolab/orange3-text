import unittest

from AnyQt.QtCore import Qt

from Orange.widgets.tests.base import WidgetTest

from orangecontrib.text import Corpus
from orangecontrib.text import preprocess
from orangecontrib.text.widgets.owcollocations import OWCollocations


class TestOWCollocations(WidgetTest):

    def setUp(self) -> None:
        self.widget: OWCollocations = self.create_widget(OWCollocations)

        # create corpus
        self.corpus = Corpus.from_file("deerwester")

    def test_set_data(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        output = self.get_output(self.widget.Outputs.corpus)
        self.assertEqual(len(self.widget.collModel), 20)
        self.assertEqual(len(output), 56)

    def test_preprocessed(self):
        pp_list = [
            preprocess.LowercaseTransformer(),
            preprocess.PunktSentenceTokenizer(),
            preprocess.SnowballStemmer(),
        ]
        for p in pp_list:
            self.pp_corpus = p(self.corpus)

        self.send_signal(self.widget.Inputs.corpus, self.pp_corpus)
        self.assertEqual(len(self.widget.collModel), 20)

    def test_trigrams(self):
        model = self.widget.collModel
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        bigram = len(model[0][0].split(" "))

        # trigrams
        self.widget.controls.type_index.buttons[1].click()
        trigram = len(self.widget.collModel[0][0].split(" "))

        self.assertGreater(trigram, bigram)

    def test_change_scorer(self):
        model = self.widget.collModel
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.assertEqual(len(model[0]), 2)

        for i, _ in enumerate(self.widget.controls.selected_method.buttons):
            self.widget.controls.selected_method.buttons[i].click()
            self.assertTrue(self.widget.Outputs.corpus)

    def test_sort_table(self):
        """Test that sorting the table for one method doesn't crash the
        widget when changing method"""
        view = self.widget.collView
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        score = self.widget.collModel[0][1]

        view.horizontalHeader().setSortIndicator(0, Qt.AscendingOrder)

        # change method
        self.widget.controls.selected_method.buttons[1].click()
        self.assertNotEqual(self.widget.collModel[0][1], score)

    def test_no_scores(self):
        """Test no scores outputs None"""
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.widget.controls.type_index.buttons[1].click()
        view = self.widget.collView
        # test view on no data
        view.horizontalHeader().setSortIndicator(0, Qt.AscendingOrder)

    def test_report(self):
        """Test report"""
        self.widget.report_button.click()  # defaults
        # set trigram which returns an empty table
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.widget.controls.type_index.buttons[1].click()
        self.widget.report_button.click()  # empty


if __name__ == "__main__":
    unittest.main()
