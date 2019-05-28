import unittest

from Orange.widgets.tests.base import WidgetTest

from orangecontrib.text.corpus import Corpus
from orangecontrib.text.vectorization import BowVectorizer
from orangecontrib.text.widgets.owwordenrichment import OWWordEnrichment


class TestWordEnrichment(WidgetTest):
    def setUp(self):
        # type: OWWordEnrichment
        self.widget = self.create_widget(OWWordEnrichment)
        self.corpus = Corpus.from_file('book-excerpts')
        vect = BowVectorizer()
        self.corpus_vect = vect.transform(self.corpus)

    def test_filter_fdr(self):
        widget = self.widget
        subset_corpus = self.corpus_vect[:10]
        self.send_signal(widget.Inputs.data, self.corpus_vect)
        self.send_signal(widget.Inputs.selected_data, subset_corpus)

        # test p-value filter
        widget.filter_by_p = True
        widget.filter_p_value = 1e-9
        widget.filter_by_fdr = False
        widget.filter_fdr_value = 0.01

        widget.filter_and_display()
        self.assertEqual(widget.sig_words.topLevelItemCount(), 3)
        self.assertEqual({widget.sig_words.topLevelItem(i).text(0)
                          for i in (0, 1, 2)}, {'livesey', 'doctor', 'rum'})

        # test fdr filter
        widget.filter_by_p = True
        widget.filter_p_value = 1e-4
        widget.filter_by_fdr = True
        widget.filter_fdr_value = 1e-4

        widget.filter_and_display()
        self.assertEqual(widget.sig_words.topLevelItemCount(), 5)
        self.assertEqual({widget.sig_words.topLevelItem(i).text(0)
                          for i in (0, 1, 2, 3, 4)},
                         {'livesey', 'doctor', 'rum', 'admiral', 'inn'})

        # test if different when fdr false
        widget.filter_by_p = True
        widget.filter_p_value = 1e-4
        widget.filter_by_fdr = False
        widget.filter_fdr_value = 1e-4

        widget.filter_and_display()
        self.assertEqual(widget.sig_words.topLevelItemCount(), 16)

        # test no results
        widget.filter_by_p = True
        widget.filter_p_value = 1e-11
        widget.filter_by_fdr = False
        widget.filter_fdr_value = 1e-5

        widget.filter_and_display()
        self.assertEqual(widget.sig_words.topLevelItemCount(), 0)


if __name__ == "__main__":
    unittest.main()
