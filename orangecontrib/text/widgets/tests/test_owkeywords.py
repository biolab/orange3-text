# pylint: disable=missing-docstring
import unittest
from unittest.mock import Mock, patch

import numpy as np
from AnyQt.QtWidgets import QCheckBox

from Orange.data import Table
from Orange.widgets.tests.base import WidgetTest, simulate

from orangecontrib.text import Corpus
from orangecontrib.text.keywords import tfidf_keywords, yake_keywords, \
    rake_keywords
from orangecontrib.text.preprocess import *
from orangecontrib.text.widgets.owkeywords import OWKeywords, run, \
    AggregationMethods, ScoringMethods
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

            for language in ("ar", "fi", "de"):
                self.corpus.attributes["language"] = language
                self.send_signal(widget.Inputs.corpus, self.corpus, widget=widget)
                self.wait_until_finished(widget=widget, timeout=10000)
                out = self.get_output(widget.Outputs.words, widget=widget)
                self.assertEqual(scores, {a.name for a in out.domain.attributes})

                m[0][1].assert_called_once()
                m[1][1].assert_called_once()
                m[2][1].assert_called_once()
                m[3][1].assert_called_once()
                for mo in m:
                    mo[1].reset_mock()

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


if __name__ == "__main__":
    unittest.main()
