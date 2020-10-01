import unittest
from unittest.mock import Mock

import numpy as np
import pkg_resources
from AnyQt.QtWidgets import QPushButton

from Orange.data import Domain, StringVariable
from Orange.widgets.tests.base import WidgetTest
from orangecontrib.text import Corpus
from orangecontrib.text.tag import AveragedPerceptronTagger
from orangecontrib.text.widgets.owstatistics import (
    STATISTICS_NAMES,
    OWStatistics,
)


class TestStatisticsWidget(WidgetTest):
    def setUp(self) -> None:
        self.widget = self.create_widget(OWStatistics)
        self.book_data = Corpus.from_file("book-excerpts")
        self._create_simple_data()

    def _create_simple_data(self) -> None:
        """
        Creat a simple dataset with 4 documents. Save it to `self.corpus`.
        """
        metas = np.array(
            [
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                "Duis viverra elit eu mi blandit, {et} sollicitudin nisi ",
                " a porta\tleo. Duis vitae ultrices massa. Mauris ut pulvinar a",
                "tortor. Class (aptent) taciti\nsociosqu ad lit1ora torquent per",
            ]
        ).reshape(-1, 1)
        text_var = StringVariable("text")
        domain = Domain([], metas=[text_var])
        self.corpus = Corpus(
            domain,
            X=np.empty((len(metas), 0)),
            metas=metas,
            text_features=[text_var],
        )

    def _set_feature(self, feature_name: str, value: str = ""):
        """
        Set statistic which need to be computed by widget. It sets only one
        statistics.

        Parameters
        ----------
        feature_name
            The name of statistic
        value
            If statistic need a value (e.g. prefix) it is passed here.
        """
        feature_index = STATISTICS_NAMES.index(feature_name)
        self.widget.active_rules = [(feature_index, value)]
        self.widget.adjust_n_rule_rows()

    def _compute_features(self, feature_name: str, value: str = "") -> Corpus:
        """
        Send `self.corpus` to widget, set statistic which need bo be computed,
        run the computation, and return widget output.

        Parameters
        ----------
        feature_name
            The name of the statistic, only one statistic is set
        value
            The value if statistic need it.

        Returns
        -------
        Resulting corpus.
        """
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.wait_until_finished()
        self._set_feature(feature_name, value)
        self.widget.apply()
        self.wait_until_finished()
        res = self.get_output(self.widget.Outputs.corpus)
        self.assertTupleEqual((len(self.corpus), 1), res.X.shape)
        return res

    def test_send_data(self):
        """ Test with basic data, and empty data """
        self.send_signal(self.widget.Inputs.corpus, self.book_data)
        self.assertEqual(len(self.book_data), len(self.widget.corpus))

        self.send_signal(self.widget.Inputs.corpus, None)
        self.assertIsNone(self.widget.corpus)
        self.widget.apply()
        self.assertIsNone(self.get_output(self.widget.Outputs.corpus))

    def test_words_count(self):
        """ Test words count statistic """
        data = self._compute_features("Word count")
        np.testing.assert_array_equal(data.X.flatten(), [8, 9, 11, 9])

        self.send_signal(self.widget.Inputs.corpus, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.corpus))

    def test_characters_count(self):
        """ Test characters count statistic """
        data = self._compute_features("Character count")
        np.testing.assert_array_equal(data.X.flatten(), [47, 44, 48, 51])

        self.send_signal(self.widget.Inputs.corpus, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.corpus))

    def test_n_gram_count(self):
        """ Test n-grams count statistic """
        data = self._compute_features("N-gram count")
        np.testing.assert_array_equal(data.X.flatten(), [10, 12, 13, 12])

        self.send_signal(self.widget.Inputs.corpus, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.corpus))

    def test_average_word_len(self):
        """ Test word density statistic """
        data = self._compute_features("Average word length")
        np.testing.assert_array_almost_equal(
            data.X.flatten(), [5.875, 4.888889, 4.363636, 5.666667]
        )

        self.send_signal(self.widget.Inputs.corpus, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.corpus))

    def test_punctuations_cont(self):
        """ Test punctuations count statistic """
        data = self._compute_features("Punctuation count")
        np.testing.assert_array_equal(data.X.flatten(), [2, 3, 2, 3])

        self.send_signal(self.widget.Inputs.corpus, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.corpus))

    def test_capitals_count(self):
        """ Test capitals count statistic """
        data = self._compute_features("Capital letter count")
        np.testing.assert_array_equal(data.X.flatten(), [1, 1, 2, 1])

        self.send_signal(self.widget.Inputs.corpus, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.corpus))

    def test_vowels_count(self):
        """ Test vowels count statistic """
        data = self._compute_features("Vowel count", "a,e,i,o,u")
        np.testing.assert_array_equal(data.X.flatten(), [19, 20, 23, 20])

        self.send_signal(self.widget.Inputs.corpus, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.corpus))

    def test_consonants_count(self):
        """ Test consonants count statistic """
        data = self._compute_features(
            "Consonant count", "b,c,d,f,g,h,j,k,l,m,n,p,q,r,s,t,v,w,x,y,z"
        )
        np.testing.assert_array_equal(data.X.flatten(), [28, 24, 25, 30])

        self.send_signal(self.widget.Inputs.corpus, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.corpus))

    def test_per_cent_unique_words(self):
        """ Test per-cent unique words statistic """
        data = self._compute_features("Per cent unique words")
        np.testing.assert_array_almost_equal(
            data.X.flatten(), [1, 1, 0.909091, 1]
        )

        self.corpus[1][-1] = ""
        data = self._compute_features("Per cent unique words")
        np.testing.assert_array_almost_equal(
            data.X.flatten(), [1, np.nan, 0.909091, 1]
        )
        
        self.send_signal(self.widget.Inputs.corpus, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.corpus))

    def test_starts_with(self):
        """ Test starts with count statistic """
        data = self._compute_features("Starts with", "a")
        np.testing.assert_array_almost_equal(data.X.flatten(), [2, 0, 2, 2])

        data = self._compute_features("Starts with", "ap")
        np.testing.assert_array_almost_equal(data.X.flatten(), [0, 0, 0, 1])

        self.send_signal(self.widget.Inputs.corpus, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.corpus))

    def test_ends_with(self):
        """ Test ends with count statistic """
        data = self._compute_features("Ends with", "t")
        np.testing.assert_array_almost_equal(data.X.flatten(), [3, 3, 1, 2])

        data = self._compute_features("Ends with", "et")
        np.testing.assert_array_almost_equal(data.X.flatten(), [1, 1, 0, 0])

        self.send_signal(self.widget.Inputs.corpus, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.corpus))

    def test_contains(self):
        """ Test contains count statistic """
        data = self._compute_features("Contains", "t")
        np.testing.assert_array_almost_equal(data.X.flatten(), [5, 4, 4, 9])

        data = self._compute_features("Contains", "et")
        np.testing.assert_array_almost_equal(data.X.flatten(), [2, 1, 0, 0])

        data = self._compute_features("Contains", "is")
        np.testing.assert_array_almost_equal(data.X.flatten(), [1, 2, 2, 0])

        self.send_signal(self.widget.Inputs.corpus, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.corpus))

    def test_regex(self):
        """ Test regex statistic """
        # words that contains digit
        data = self._compute_features("Regex", "\w*\d\w*")
        np.testing.assert_array_almost_equal(data.X.flatten(), [0, 0, 0, 1])

        # words that contains digit
        data = self._compute_features("Regex", "\w*is\w*")
        np.testing.assert_array_almost_equal(data.X.flatten(), [1, 2, 2, 0])

        self.send_signal(self.widget.Inputs.corpus, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.corpus))

    def test_pos(self):
        """
        Test post tags count
        - test with corpus that has no pos tags - warning raised
        - test with corpus that has pos tags
        """
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self._set_feature("POS tag", "NN")
        self.widget.apply()
        self.wait_until_finished()
        res = self.get_output(self.widget.Outputs.corpus)
        self.assertEqual(0, res.X.shape[1])
        self.assertTrue(self.widget.Warning.not_computed.is_shown())

        tagger = AveragedPerceptronTagger()
        result = tagger(self.corpus)

        self.send_signal(self.widget.Inputs.corpus, result)
        self._set_feature("POS tag", "NN")
        self.widget.apply()
        self.wait_until_finished()
        res = self.get_output(self.widget.Outputs.corpus)
        self.assertTupleEqual((len(self.corpus), 1), res.X.shape)
        np.testing.assert_array_almost_equal(res.X.flatten(), [6, 5, 4, 5])
        self.assertFalse(self.widget.Warning.not_computed.is_shown())

    def test_statistics_combination(self):
        """
        Testing three statistics at same time and see if column concatenated
        correctly.
        """
        self.send_signal(self.widget.Inputs.corpus, self.corpus)

        wc_index = STATISTICS_NAMES.index("Word count")
        starts_with_index = STATISTICS_NAMES.index("Starts with")
        capital_counts_index = STATISTICS_NAMES.index("Capital letter count")
        self.widget.active_rules = [
            (wc_index, ""),
            (starts_with_index, "a"),
            (capital_counts_index, ""),
        ]
        self.widget.adjust_n_rule_rows()

        self.widget.apply()
        self.wait_until_finished()
        res = self.get_output(self.widget.Outputs.corpus)

        self.assertTupleEqual((len(self.corpus), 3), res.X.shape)
        np.testing.assert_array_almost_equal(
            res.X[:, 0].flatten(), [8, 9, 11, 9]
        )
        np.testing.assert_array_almost_equal(
            res.X[:, 1].flatten(), [2, 0, 2, 2]
        )
        np.testing.assert_array_almost_equal(
            res.X[:, 2].flatten(), [1, 1, 2, 1]
        )

    def test_dictionary_statistics(self):
        """
        Test remove statistic from the dictionary when they are not required
        """
        self.send_signal(self.widget.Inputs.corpus, self.corpus)

        self.widget.active_rules = [
            (1, ""),
        ]
        self.widget.adjust_n_rule_rows()
        self.widget.apply()
        self.wait_until_finished()

        self.assertListEqual([(1, "")], list(self.widget.result_dict.keys()))

        self.widget.active_rules = [(1, ""), (2, "")]
        self.widget.adjust_n_rule_rows()
        self.widget.apply()
        self.wait_until_finished()

        self.assertListEqual(
            [(1, ""), (2, "")], list(self.widget.result_dict.keys())
        )

        self.widget.active_rules = [(2, "")]
        self.widget.adjust_n_rule_rows()
        self.widget.apply()
        self.wait_until_finished()

        self.assertListEqual([(2, "")], list(self.widget.result_dict.keys()))

        # dict should empty on new data
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.assertListEqual([], list(self.widget.result_dict.keys()))

    def test_context(self):
        """ Test whether context correctly restore rules """
        rules = [(0, ""), (1, ""), (2, "")]
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.widget.active_rules = rules[:]

        self.send_signal(self.widget.Inputs.corpus, self.book_data)
        self.assertListEqual([(0, ""), (1, "")], self.widget.active_rules)

        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.assertListEqual(rules, self.widget.active_rules)

    def test_compute_values(self):
        """ Test compute values on new data """
        data = self._compute_features("Word count")

        computed = Corpus.from_table(data.domain, self.book_data)
        self.assertEqual(data.domain, computed.domain)
        self.assertTupleEqual((len(self.book_data), 1), computed.X.shape)

    def test_append_to_existing_X(self):
        """ Test if new features are correctly attached to X matrix """
        data = Corpus.from_file("election-tweets-2016")
        self.send_signal(self.widget.Inputs.corpus, data)
        self.wait_until_finished()
        statistics = self.get_output(self.widget.Outputs.corpus)

        self.assertTupleEqual(
            (data.X.shape[0], data.X.shape[1] + 2), statistics.X.shape
        )

    def test_add_row(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.wait_until_finished()
        self.widget.active_rules = []
        self.widget.adjust_n_rule_rows()
        add_button = [
            x
            for x in self.widget.controlArea.findChildren(QPushButton)
            if x.text() == "+"
        ][0]
        add_button.click()
        self.assertListEqual([(0, "")], self.widget.active_rules)

    def test_remove_row(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.widget.active_rules = [(0, "")]
        self.widget.adjust_n_rule_rows()
        self.assertListEqual([(0, "")], self.widget.active_rules)

        remove_button = [
            x
            for x in self.widget.controlArea.findChildren(QPushButton)
            if x.text() == "×"
        ][0]
        remove_button.click()
        self.assertListEqual([], self.widget.active_rules)

    def test_input_summary(self):
        """ Test correctness of the input summary """
        self.widget.info.set_input_summary = in_sum = Mock()

        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        in_sum.assert_called_with(
            len(self.corpus),
            "4 instances, 1 variable\nFeatures: — (No missing values)"
            "\nTarget: —\nMetas: string",
        )
        in_sum.reset_mock()

        self.send_signal(self.widget.Inputs.corpus, self.book_data)
        in_sum.assert_called_with(
            len(self.book_data),
            "140 instances, 2 variables\nFeatures: — (No missing values)"
            "\nTarget: categorical\nMetas: string",
        )
        in_sum.reset_mock()

        self.send_signal(self.widget.Inputs.corpus, None)
        in_sum.assert_called_with(self.widget.info.NoInput)

    def test_output_summary(self):
        """ Test correctness of the output summary"""
        self.widget.info.set_output_summary = out_sum = Mock()

        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.wait_until_finished()
        out_sum.assert_called_with(
            len(self.corpus),
            "4 instances, 3 variables\nFeatures: 2 numeric (No missing values)"
            "\nTarget: —\nMetas: string",
        )
        out_sum.reset_mock()

        self.send_signal(self.widget.Inputs.corpus, self.book_data)
        self.wait_until_finished()
        out_sum.assert_called_with(
            len(self.book_data),
            "140 instances, 4 variables\nFeatures: 2 numeric (No missing values)"
            "\nTarget: categorical\nMetas: string",
        )
        out_sum.reset_mock()

        self.send_signal(self.widget.Inputs.corpus, None)
        self.wait_until_finished()
        out_sum.assert_called_with(self.widget.info.NoOutput)


if __name__ == "__main__":
    unittest.main()
