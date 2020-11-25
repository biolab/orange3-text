import unittest
from unittest.mock import Mock

import Orange
from Orange.data import Table, Domain
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.text.corpus import Corpus
from orangecontrib.text.vectorization import BowVectorizer
from orangecontrib.text.widgets.owwordenrichment import OWWordEnrichment


class TestWordEnrichment(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWWordEnrichment)
        corpus = Corpus.from_file('book-excerpts')[::3]
        vect = BowVectorizer()
        self.corpus_vect = vect.transform(corpus)
        self.subset_corpus = self.corpus_vect[:5]

    @unittest.skipIf(
        Orange.__version__ < "3.24.0", "wait_until_finished not supported")
    def test_filter_fdr(self):
        widget = self.widget

        self.send_signal(widget.Inputs.data, self.corpus_vect)
        self.send_signal(widget.Inputs.selected_data, self.subset_corpus)
        self.wait_until_finished(timeout=10000)

        # test p-value filter
        widget.filter_by_p = True
        widget.filter_p_value = 1e-3
        widget.filter_by_fdr = False
        widget.filter_fdr_value = 0.01

        widget.filter_and_display()
        self.assertEqual(widget.sig_words.topLevelItemCount(), 3)
        self.assertEqual({widget.sig_words.topLevelItem(i).text(0)
                          for i in (0, 1, 2)}, {'livesey', 'jim', 'doctor'})

        # test fdr filter
        widget.filter_by_p = True
        widget.filter_p_value = 1e-1
        widget.filter_by_fdr = True
        widget.filter_fdr_value = 0.9

        widget.filter_and_display()
        self.assertEqual(widget.sig_words.topLevelItemCount(), 1)
        self.assertEqual(widget.sig_words.topLevelItem(0).text(0), "doctor")

        # test if different when fdr false
        widget.filter_by_p = True
        widget.filter_p_value = 1e-1
        widget.filter_by_fdr = False
        widget.filter_fdr_value = 1e-4

        widget.filter_and_display()
        self.assertEqual(widget.sig_words.topLevelItemCount(), 108)

        # test no results
        widget.filter_by_p = True
        widget.filter_p_value = 1e-11
        widget.filter_by_fdr = False
        widget.filter_fdr_value = 1e-5

        widget.filter_and_display()
        self.assertEqual(widget.sig_words.topLevelItemCount(), 0)

        self.send_signal(widget.Inputs.data, None)
        widget.filter_and_display()
        self.assertEqual(widget.sig_words.topLevelItemCount(), 0)

    def test_empty_selection(self):
        w = self.widget

        # empty selection
        self.send_signal(w.Inputs.data, self.corpus_vect)
        self.send_signal(w.Inputs.selected_data, self.subset_corpus[:0])
        self.assertTrue(self.widget.Error.empty_selection.is_shown())

        # when commands changed on non-valid data
        w.controls.filter_by_p.click()

        # selection not empty
        self.send_signal(w.Inputs.selected_data, self.subset_corpus)
        self.assertFalse(self.widget.Error.empty_selection.is_shown())

    def test_no_bow_features(self):
        w = self.widget

        iris = Table("iris")
        self.send_signal(w.Inputs.data, iris)
        self.send_signal(w.Inputs.selected_data, iris[:10])
        self.assertTrue(self.widget.Error.no_bow_features.is_shown())

        # when commands changed on non-valid data
        w.controls.filter_by_p.click()

        self.send_signal(w.Inputs.data, None)
        self.send_signal(w.Inputs.selected_data, None)
        self.assertFalse(self.widget.Error.no_bow_features.is_shown())

    def test_all_selected(self):
        w = self.widget

        self.send_signal(w.Inputs.data, self.corpus_vect)
        self.send_signal(w.Inputs.selected_data, self.corpus_vect)
        self.assertTrue(self.widget.Error.all_selected.is_shown())

        # when commands changed on non-valid data
        w.controls.filter_by_p.click()

        self.send_signal(w.Inputs.data, None)
        self.send_signal(w.Inputs.selected_data, None)
        self.assertFalse(self.widget.Error.all_selected.is_shown())

    def test_no_overlapping(self):
        w = self.widget

        # with one column bow it is easier
        corpus_vect = Corpus.from_table(Domain(
            self.corpus_vect.domain.attributes[:1],
            self.corpus_vect.domain.class_var,
            self.corpus_vect.domain.metas
        ), self.corpus_vect)

        self.send_signal(w.Inputs.data, corpus_vect[10:15])
        self.send_signal(w.Inputs.selected_data, corpus_vect[4:5])
        self.assertTrue(self.widget.Error.no_words_overlap.is_shown())

        # when commands changed on non-valid data
        w.controls.filter_by_p.click()

        self.send_signal(w.Inputs.selected_data, self.subset_corpus)
        self.send_signal(w.Inputs.data, self.corpus_vect)
        self.assertFalse(self.widget.Error.no_words_overlap.is_shown())

    @unittest.skipIf(
        Orange.__version__ < "3.24.0", "wait_until_finished not supported")
    def test_input_info(self):
        w = self.widget
        input_sum = w.info.set_input_summary = Mock()

        self.send_signal(w.Inputs.selected_data, self.subset_corpus)
        self.send_signal(w.Inputs.data, self.corpus_vect)

        input_sum.assert_called_with(
            "5923|1204", "Total words: 5923\nWords in subset: 1204")

        self.wait_until_stop_blocking()
        self.send_signal(w.Inputs.selected_data, None)
        self.send_signal(w.Inputs.data, None)
        input_sum.assert_called_with(w.info.NoInput)

    @unittest.skipIf(
        Orange.__version__ < "3.24.0", "wait_until_finished not supported")
    def test_output_info(self):
        w = self.widget
        w.filter_p_value = 1e-3
        w.filter_by_p = True
        w.filter_by_fdr = False

        self.send_signal(w.Inputs.selected_data, self.subset_corpus)
        self.send_signal(w.Inputs.data, self.corpus_vect)
        self.wait_until_finished(timeout=10000)

        self.assertEqual(w.info_fil.text(), "Words displayed: 3")

        # test fdr filter
        w.filter_by_p = True
        w.filter_p_value = 1e-4
        w.filter_by_fdr = True
        w.filter_fdr_value = 1e-4
        w.filter_and_display()
        self.assertEqual(w.info_fil.text(), "Words displayed: 0")

        self.send_signal(w.Inputs.selected_data, None)
        self.assertEqual(w.info_fil.text(), "Words displayed: 0")
        self.send_signal(w.Inputs.data, None)
        self.assertEqual(w.info_fil.text(), "Words displayed: 0")

    @unittest.skipIf(
        Orange.__version__ < "3.24.0", "wait_until_finished not supported")
    def test_filter_changed(self):
        """
        This case tests whether function are correctly triggered when
        values in filter field changes
        """
        w = self.widget

        self.send_signal(w.Inputs.data, self.corpus_vect)
        self.send_signal(w.Inputs.selected_data, self.subset_corpus)
        self.wait_until_finished(timeout=10000)

        # test p-value filter
        w.controls.filter_by_p.click()  # set to true
        w.controls.filter_p_value.valueChanged.emit(1e-3)
        w.controls.filter_by_fdr.click()  # set to false
        w.controls.filter_fdr_value.valueChanged.emit(0.1)

        self.assertEqual(w.sig_words.topLevelItemCount(), 3)
        self.assertEqual({w.sig_words.topLevelItem(i).text(0)
                          for i in (0, 1, 2)}, {'livesey', 'jim', 'doctor'})

        # # test fdr filter
        w.controls.filter_p_value.valueChanged.emit(1e-1)
        w.controls.filter_by_fdr.click()  # set to True
        w.controls.filter_fdr_value.valueChanged.emit(0.9)

        self.assertEqual(w.sig_words.topLevelItemCount(), 1)
        self.assertEqual(w.sig_words.topLevelItem(0).text(0), "doctor")

        # test if different when fdr false
        w.controls.filter_by_fdr.click()  # set to False

        self.assertEqual(w.sig_words.topLevelItemCount(), 108)

        # # # test no results
        w.controls.filter_p_value.valueChanged.emit(1e-11)

        self.assertEqual(w.sig_words.topLevelItemCount(), 0)

    @unittest.skipIf(
        Orange.__version__ < "3.24.0", "wait_until_finished not supported")
    def test_report(self):
        """
        Just test if report works.
        """
        w = self.widget

        w.send_report()

        self.send_signal(w.Inputs.data, self.corpus_vect)
        self.send_signal(w.Inputs.selected_data, self.subset_corpus)
        self.wait_until_finished(timeout=10000)

        w.send_report()


if __name__ == "__main__":
    unittest.main()
