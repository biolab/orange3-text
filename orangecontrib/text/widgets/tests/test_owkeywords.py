# pylint: disable=missing-docstring
import unittest
from unittest.mock import Mock, patch

import numpy as np
from AnyQt.QtCore import QItemSelectionModel
from AnyQt.QtWidgets import QCheckBox

from Orange.data import Table
from Orange.widgets.tests.base import WidgetTest, simulate

from orangecontrib.text import Corpus
from orangecontrib.text.keywords import tfidf_keywords, yake_keywords, \
    rake_keywords
from orangecontrib.text.preprocess import *
from orangecontrib.text.widgets.owkeywords import (
    OWKeywords,
    run,
    AggregationMethods,
    ScoringMethods,
    SelectionMethods,
    CONNECTION_WARNING,
)
from orangecontrib.text.widgets.utils.words import create_words_table


class TestRunner(unittest.TestCase):
    def setUp(self):
        self.corpus = Corpus.from_file("deerwester")
        self.state = Mock()
        self.state.is_interruption_requested = Mock(return_value=False)

    def test_run_default(self):
        results = run(self.corpus, None, {}, {ScoringMethods.TF_IDF}, {},
                      AggregationMethods.MEAN, self.state)
        self.assertEqual(results.scores[0][0], "system")
        self.assertAlmostEqual(results.scores[0][1], 0.114, 2)
        self.assertEqual(results.labels, ["TF-IDF"])

    def test_run_multiple_methods(self):
        results = run(self.corpus, None, {},
                      {ScoringMethods.TF_IDF, ScoringMethods.YAKE}, {},
                      AggregationMethods.MEAN, self.state)
        self.assertEqual(results.scores[0][0], "system")
        self.assertAlmostEqual(results.scores[0][1], 0.114, 2)
        self.assertTrue(np.isnan(np.nan))
        self.assertEqual(results.labels, ["TF-IDF", "YAKE!"])

    def test_run_no_data(self):
        results = run(None, None, {}, {ScoringMethods.TF_IDF}, {},
                      AggregationMethods.MEAN, Mock())
        self.assertEqual(results.scores, [])
        self.assertEqual(results.labels, [])
        self.assertEqual(results.all_keywords, {})

    def test_run_no_methods(self):
        cached_keywords = Mock()
        results = run(self.corpus, None, cached_keywords, set(), {},
                      AggregationMethods.MEAN, Mock())
        self.assertEqual(results.scores, [])
        self.assertEqual(results.labels, [])
        self.assertIs(results.all_keywords, cached_keywords)

    def test_run_with_words(self):
        words = ["human", "graph", "minors", "trees"]
        results = run(self.corpus, words, {}, {ScoringMethods.TF_IDF}, {},
                      AggregationMethods.MEAN, self.state)
        self.assertEqual(len(results.scores), 4)

        words = ["foo", "bar"]
        results = run(self.corpus, words, {}, {ScoringMethods.TF_IDF}, {},
                      AggregationMethods.MEAN, self.state)
        self.assertEqual(len(results.scores), 0)

        words = []
        results = run(self.corpus, words, {}, {ScoringMethods.TF_IDF}, {},
                      AggregationMethods.MEAN, self.state)
        self.assertEqual(len(results.scores), 42)

        words = None
        results = run(self.corpus, words, {}, {ScoringMethods.TF_IDF}, {},
                      AggregationMethods.MEAN, self.state)
        self.assertEqual(len(results.scores), 42)

    def test_run_normalize_words(self):
        normalizer = LemmagenLemmatizer()
        corpus = normalizer(self.corpus)

        words = ["minor", "tree"]
        results = run(corpus, words, {}, {ScoringMethods.TF_IDF}, {},
                      AggregationMethods.MEAN, self.state)
        self.assertEqual(len(results.scores), 2)

        words = ["minors", "trees"]
        results = run(corpus, words, {}, {ScoringMethods.TF_IDF}, {},
                      AggregationMethods.MEAN, self.state)
        self.assertEqual(len(results.scores), 2)

    def test_run_with_cached_results(self):
        results1 = run(self.corpus, None, {},
                       {ScoringMethods.TF_IDF, ScoringMethods.YAKE}, {},
                       AggregationMethods.MEAN, self.state)

        with patch("orangecontrib.text.keywords.tfidf_keywords") as mock:
            results2 = run(self.corpus, None, results1.all_keywords,
                           {ScoringMethods.TF_IDF, ScoringMethods.YAKE}, {},
                           AggregationMethods.MEAN, self.state)
            mock.assert_not_called()
            self.assertNanEqual(results1.scores, results2.scores)
            self.assertNanEqual(results1.labels, results2.labels)
            self.assertNanEqual(results1.all_keywords, results2.all_keywords)

    def test_run_interrupt(self):
        state = Mock()
        state.is_interruption_requested = Mock(return_value=True)
        self.assertRaises(Exception, run, self.corpus, None, {},
                          {ScoringMethods.TF_IDF}, {},
                          AggregationMethods.MEAN, state)

    def test_run_mbert_fail(self):
        """Test mbert partially or completely fails due to connection issues"""
        agg, sc = AggregationMethods.MEAN, {ScoringMethods.MBERT}
        res = [[("keyword1", 10), ("keyword2", 2)], None, [("keyword1", 5)]]
        with patch.object(ScoringMethods, "ITEMS", [("mBERT", Mock(return_value=res))]):
            results = run(self.corpus[:3], None, {}, sc, {}, agg, self.state)
            self.assertListEqual([["keyword1", 7.5], ["keyword2", 1]], results.scores)
            self.assertListEqual(["mBERT"], results.labels)
            # not stored to all_keywords since not all extracted exactly
            self.assertDictEqual({}, results.all_keywords)
            self.assertListEqual([CONNECTION_WARNING], results.warnings)

        res = [None] * 3
        with patch.object(ScoringMethods, "ITEMS", [("mBERT", Mock(return_value=res))]):
            results = run(self.corpus[:3], None, {}, sc, {}, agg, self.state)
            self.assertListEqual([], results.scores)
            self.assertListEqual(["mBERT"], results.labels)
            # not stored to all_keywords since not all extracted exactly
            self.assertDictEqual({}, results.all_keywords)
            self.assertListEqual([CONNECTION_WARNING], results.warnings)

    def assertNanEqual(self, table1, table2):
        for list1, list2 in zip(table1, table2):
            for x1, x2 in zip(list1, list2):
                if isinstance(x1, float) and np.isnan(x1):
                    self.assertTrue(np.isnan(x2))
                else:
                    self.assertEqual(x1, x2)


class TestOWKeywords(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWKeywords)
        self.corpus = Corpus.from_file("deerwester")

    def test_default(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.wait_until_finished()
        output = self.get_output(self.widget.Outputs.words)
        self.assertEqual(len(output), 3)
        self.assertTrue(output.domain.metas[0].name, "Words")
        self.assertDictEqual(output.domain.metas[0].attributes,
                             {"type": "words"})
        self.assertListEqual(list(output.metas[:, 0]),
                             ['system', 'a', 'survey'])

    def test_input_words(self):
        words = create_words_table(["foo", "graph", "minors", "trees"])
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.send_signal(self.widget.Inputs.words, words)
        self.wait_until_finished()
        output = self.get_output(self.widget.Outputs.words)
        self.assertListEqual(list(output.metas[:, 0]),
                             ['graph', 'minors', 'trees'])

    def test_input_words_no_type(self):
        words = Table("zoo")
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.send_signal(self.widget.Inputs.words, words)
        self.assertTrue(self.widget.Warning.no_words_column.is_shown())
        self.send_signal(self.widget.Inputs.words, None)
        self.assertFalse(self.widget.Warning.no_words_column.is_shown())

    def test_sort_nans_desc(self):
        settings = {"selected_scoring_methods": {"TF-IDF", "YAKE!"},
                    "sort_column_order": (2, 1)}
        widget = self.create_widget(OWKeywords, stored_settings=settings)
        self.send_signal(widget.Inputs.corpus, self.corpus, widget=widget)
        self.wait_until_finished(widget=widget)
        output = self.get_output(widget.Outputs.words, widget=widget)
        self.assertListEqual(list(output.metas[:, 0]),
                             ['user', 'minors', 'trees'])

    def test_sort_nans_asc(self):
        settings = {"selected_scoring_methods": {"TF-IDF", "YAKE!"},
                    "sort_column_order": (2, 0)}
        widget = self.create_widget(OWKeywords, stored_settings=settings)
        self.send_signal(widget.Inputs.corpus, self.corpus, widget=widget)
        self.wait_until_finished(widget=widget)
        output = self.get_output(widget.Outputs.words, widget=widget)
        self.assertListEqual(list(output.metas[:, 0]),
                             ["System", "Widths", "opinion"])

    def test_scoring_methods(self):
        # speed-up the test execution
        def dummy_mbert(tokens, progress_callback=None):
            return [[("kw1", 0.2), ("kw2", 0.3)] * len(tokens)]

        methods = [
            ("TF-IDF", Mock(wraps=tfidf_keywords)),
            ("YAKE!", Mock(wraps=yake_keywords)),
            ("Rake", Mock(wraps=rake_keywords)),
            ("MBERT", Mock(side_effect=dummy_mbert)),
        ]
        with patch.object(ScoringMethods, "ITEMS", methods) as m:
            scores = {"TF-IDF", "YAKE!", "Rake", "MBERT"}
            settings = {"selected_scoring_methods": scores}
            widget = self.create_widget(OWKeywords, stored_settings=settings)
            self.send_signal(widget.Inputs.corpus, self.corpus, widget=widget)
            self.wait_until_finished(widget=widget, timeout=10000)

            for i in range(4):
                m[i][1].assert_called_once()
                m[i][1].reset_mock()

            cb = widget.controls.yake_language
            simulate.combobox_activate_item(cb, "Arabic")
            self.wait_until_finished(widget=widget, timeout=10000)
            cb = widget.controls.rake_language
            simulate.combobox_activate_item(cb, "Finnish")
            self.wait_until_finished(widget=widget, timeout=10000)

            out = self.get_output(widget.Outputs.words, widget=widget)
            self.assertEqual(scores, {a.name for a in out.domain.attributes})

            m[1][1].assert_called_once()
            m[2][1].assert_called_once()
            self.assertEqual(m[1][1].call_args[1]["language"], "ar")
            self.assertEqual(m[2][1].call_args[1]["language"], "fi")

    def test_method_change(self):
        """Test method change by clicking"""
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        out = self.get_output(self.widget.Outputs.words)
        self.assertEqual({"TF-IDF"}, {a.name for a in out.domain.attributes})

        self.widget.controlArea.findChildren(QCheckBox)[1].click()  # yake cb
        out = self.get_output(self.widget.Outputs.words)
        self.assertEqual({"TF-IDF", "YAKE!"}, {a.name for a in out.domain.attributes})

        self.widget.controlArea.findChildren(QCheckBox)[1].click()
        out = self.get_output(self.widget.Outputs.words)
        self.assertEqual({"TF-IDF"}, {a.name for a in out.domain.attributes})

    def test_send_report(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.wait_until_finished()
        self.widget.send_report()
        words = create_words_table(["human", "graph", "minors", "trees"])
        self.send_signal(self.widget.Inputs.words, words)
        self.wait_until_finished()
        self.widget.send_report()
        self.send_signal(self.widget.Inputs.corpus, None)
        self.widget.send_report()

    def test_selection_none(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        radio_buttons = self.widget._OWKeywords__sel_method_buttons
        radio_buttons.button(SelectionMethods.NONE).click()

        output = self.get_output(self.widget.Outputs.words)
        self.assertIsNone(output)

    def tests_selection_all(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        radio_buttons = self.widget._OWKeywords__sel_method_buttons
        radio_buttons.button(SelectionMethods.ALL).click()

        output = self.get_output(self.widget.Outputs.words)
        self.assertEqual(42, len(output))

    def test_selection_manual(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.wait_until_finished()
        radio_buttons = self.widget._OWKeywords__sel_method_buttons
        radio_buttons.button(SelectionMethods.MANUAL).click()

        mode = QItemSelectionModel.Rows | QItemSelectionModel.Select
        self.widget.view.clearSelection()
        model = self.widget.view.model()
        self.widget.view.selectionModel().select(model.index(2, 0), mode)
        self.widget.view.selectionModel().select(model.index(3, 0), mode)

        output = self.get_output(self.widget.Outputs.words)
        self.assertEqual(2, len(output))

    def test_selection_n_best(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        radio_buttons = self.widget._OWKeywords__sel_method_buttons
        radio_buttons.button(SelectionMethods.N_BEST).click()

        output = self.get_output(self.widget.Outputs.words)
        self.assertEqual(3, len(output))

        self.widget.controls.n_selected.setValue(5)
        output = self.get_output(self.widget.Outputs.words)
        self.assertEqual(5, len(output))

    def test_connection_error(self):
        self.widget.controlArea.findChildren(QCheckBox)[0].click()  # unselect tfidf
        self.widget.controlArea.findChildren(QCheckBox)[3].click()  # unselect mbert
        res = [[("keyword1", 10), ("keyword2", 2)], None, [("keyword1", 5)]]
        with patch.object(ScoringMethods, "ITEMS", [("mBERT", Mock(return_value=res))]):
            self.send_signal(self.widget.Inputs.corpus, self.corpus)
            output = self.get_output(self.widget.Outputs.words)
            self.assertEqual(len(output), 2)
            np.testing.assert_array_equal(output.metas, [["keyword1"], ["keyword2"]])
            np.testing.assert_array_equal(output.X, [[7.5], [1]])
            self.assertTrue(self.widget.Warning.extraction_warnings.is_shown())
            self.assertEqual(
                CONNECTION_WARNING, str(self.widget.Warning.extraction_warnings)
            )

        res = [None] * 3  # all failed
        with patch.object(ScoringMethods, "ITEMS", [("mBERT", Mock(return_value=res))]):
            self.send_signal(self.widget.Inputs.corpus, self.corpus)
            self.assertIsNone(self.get_output(self.widget.Outputs.words))
            self.assertTrue(self.widget.Warning.extraction_warnings.is_shown())
            self.assertEqual(
                CONNECTION_WARNING, str(self.widget.Warning.extraction_warnings)
            )

        res = [[("keyword1", 10), ("keyword2", 2)], [("keyword1", 5)]]
        with patch.object(ScoringMethods, "ITEMS", [("mBERT", Mock(return_value=res))]):
            self.send_signal(self.widget.Inputs.corpus, self.corpus)
            output = self.get_output(self.widget.Outputs.words)
            np.testing.assert_array_equal(output.metas, [["keyword1"], ["keyword2"]])
            np.testing.assert_array_equal(output.X, [[7.5], [1]])
            self.assertFalse(self.widget.Warning.extraction_warnings.is_shown())

    def test_language_from_corpus(self):
        self.corpus.attributes["language"] = "it"
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.assertEqual("it", self.widget.yake_language)
        self.assertEqual("it", self.widget.rake_language)

        simulate.combobox_activate_item(self.widget.controls.yake_language, "Finnish")
        simulate.combobox_activate_item(self.widget.controls.rake_language, "Finnish")
        self.assertEqual("fi", self.widget.yake_language)
        self.assertEqual("fi", self.widget.rake_language)

        # language none of them support - language should not change
        self.corpus.attributes["language"] = "mr"
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.assertEqual("fi", self.widget.yake_language)
        self.assertEqual("fi", self.widget.rake_language)

        # language that is supported by RAKE - language sets for RAKE
        self.corpus.attributes["language"] = "hi_eng"
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.assertEqual("fi", self.widget.yake_language)
        self.assertEqual("hi_eng", self.widget.rake_language)

        # language that is supported by YAKE! - language sets for YAKE
        self.corpus.attributes["language"] = "uk"
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.assertEqual("uk", self.widget.yake_language)
        self.assertEqual("hi_eng", self.widget.rake_language)

        # language that both support - widget sets both langagues
        self.corpus.attributes["language"] = "it"
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.assertEqual("it", self.widget.yake_language)
        self.assertEqual("it", self.widget.rake_language)

        # langauge is None - nothing changes
        self.corpus.attributes["language"] = None
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.assertEqual("it", self.widget.yake_language)
        self.assertEqual("it", self.widget.rake_language)

        # corpus None - nothing changes
        self.send_signal(self.widget.Inputs.corpus, None)
        self.assertEqual("it", self.widget.yake_language)
        self.assertEqual("it", self.widget.rake_language)

    def test_language_from_settings(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        simulate.combobox_activate_item(self.widget.controls.yake_language, "Slovenian")
        simulate.combobox_activate_item(self.widget.controls.rake_language, "Nepali")

        self.assertEqual("sl", self.widget.yake_language)
        self.assertEqual("ne", self.widget.rake_language)
        settings = self.widget.settingsHandler.pack_data(self.widget)

        widget = self.create_widget(OWKeywords, stored_settings=settings)
        self.assertEqual("en", self.corpus.language)
        self.send_signal(widget.Inputs.corpus, self.corpus, widget=widget)
        self.assertEqual("sl", widget.yake_language)
        self.assertEqual("ne", widget.rake_language)

    def test_language_migration(self):
        settings = {"__version__": 1, "yake_lang_index": 0, "rake_lang_index": 0}
        widget = self.create_widget(OWKeywords, stored_settings=settings)
        self.assertEqual("ar", widget.yake_language)
        self.assertEqual("ar", widget.rake_language)

        settings = {"__version__": 1, "yake_lang_index": 4, "rake_lang_index": 4}
        widget = self.create_widget(OWKeywords, stored_settings=settings)
        self.assertEqual("zh", widget.yake_language)
        self.assertEqual("ca", widget.rake_language)

        settings = {"__version__": 1, "yake_lang_index": 20, "rake_lang_index": 20}
        widget = self.create_widget(OWKeywords, stored_settings=settings)
        self.assertEqual("lv", widget.yake_language)
        self.assertEqual("no", widget.rake_language)

        settings = {"__version__": 1, "yake_lang_index": 33, "rake_lang_index": 28}
        widget = self.create_widget(OWKeywords, stored_settings=settings)
        self.assertEqual("uk", widget.yake_language)
        self.assertEqual("tr", widget.rake_language)


if __name__ == "__main__":
    unittest.main()
