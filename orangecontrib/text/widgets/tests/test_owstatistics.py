import unittest

import numpy as np
from AnyQt.QtWidgets import QPushButton

from Orange.data import Domain, StringVariable
from Orange.preprocess import SklImpute
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate
from orangecontrib.text import Corpus
from orangecontrib.text.preprocess import (
    PreprocessorList,
    LowercaseTransformer,
    RegexpTokenizer,
    StopwordsFilter,
)
from orangecontrib.text.tag import AveragedPerceptronTagger
from orangecontrib.text.widgets.owstatistics import (
    STATISTICS_NAMES,
    OWStatistics, Sources,
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
                ["Lorem ipsum dolor sit amet, consectetur adipiscing elit."],
                ["Duis viverra elit eu mi blandit, {et} sollicitudin nisi "],
                [" a porta\tleo. Duis vitae ultrices massa. Mauris ut pulvinar a"],
                ["tortor. Class (aptent) taciti\nsociosqu ad lit1ora torquent per"],
            ]
        )
        text_var = StringVariable("text")
        domain = Domain([], metas=[text_var])
        self.corpus = Corpus.from_numpy(
            domain,
            X=np.empty((len(metas), 0)),
            metas=metas,
            text_features=[text_var],
        )

    def _set_feature(
            self, feature_name: str, value: str = "", source: str = Sources.DOCUMENTS
    ):
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
        simulate.combobox_activate_item(self.widget.statistics_combos[0], feature_name)
        self.widget.line_edits[0].setText(value)
        simulate.combobox_activate_item(self.widget.source_combos[0], source)
        for button in self.widget.remove_buttons[1:]:
            button.click()

    def _compute_features(
        self, feature_name: str, value: str = "", source: str = Sources.DOCUMENTS
    ) -> Corpus:
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
        self._set_feature(feature_name, value, source)
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
        data = self._compute_features("Character count", source=Sources.DOCUMENTS)
        np.testing.assert_array_equal(data.X.flatten(), [47, 44, 48, 51])

        data = self._compute_features("Character count", source=Sources.TOKENS)
        np.testing.assert_array_equal(data.X.flatten(), [47, 44, 48, 51])

        self.send_signal(self.widget.Inputs.corpus, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.corpus))

    def test_n_gram_count(self):
        """ Test n-grams count statistic """
        data = self._compute_features("N-gram count", source=Sources.TOKENS)
        np.testing.assert_array_equal(data.X.flatten(), [10, 12, 13, 12])

        self.send_signal(self.widget.Inputs.corpus, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.corpus))

    def test_average_word_len(self):
        """ Test word density statistic """
        data = self._compute_features("Average term length", source=Sources.DOCUMENTS)
        np.testing.assert_array_almost_equal(
            data.X.flatten(), [5.875, 4.888889, 4.363636, 5.666667]
        )

        data = self._compute_features("Average term length", source=Sources.TOKENS)
        np.testing.assert_array_almost_equal(
            data.X.flatten(), [4.7, 3.666667, 3.692308, 4.25]
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
        data = self._compute_features("Per cent unique terms", source=Sources.TOKENS)
        np.testing.assert_array_almost_equal(
            data.X.flatten(), [1, 1, 0.84615, 1], decimal=5
        )

        with self.corpus.unlocked():
            self.corpus[1][-1] = " "
        data = self._compute_features("Per cent unique terms", source=Sources.TOKENS)
        np.testing.assert_array_almost_equal(
            data.X.flatten(), [1, np.nan, 0.84615, 1], decimal=5
        )
        
        self.send_signal(self.widget.Inputs.corpus, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.corpus))

    def test_starts_with(self):
        """ Test starts with count statistic """
        data = self._compute_features("Starts with", "a", Sources.TOKENS)
        np.testing.assert_array_almost_equal(data.X.flatten(), [2, 0, 2, 2])

        data = self._compute_features("Starts with", "ap", Sources.TOKENS)
        np.testing.assert_array_almost_equal(data.X.flatten(), [0, 0, 0, 1])

        self.send_signal(self.widget.Inputs.corpus, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.corpus))

    def test_ends_with(self):
        """ Test ends with count statistic """
        data = self._compute_features("Ends with", "t", Sources.TOKENS)
        np.testing.assert_array_almost_equal(data.X.flatten(), [3, 3, 1, 2])

        data = self._compute_features("Ends with", "et", Sources.TOKENS)
        np.testing.assert_array_almost_equal(data.X.flatten(), [1, 1, 0, 0])

        self.send_signal(self.widget.Inputs.corpus, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.corpus))

    def test_contains(self):
        """ Test contains count statistic """
        data = self._compute_features("Contains", "t", Sources.DOCUMENTS)
        np.testing.assert_array_almost_equal(data.X.flatten(), [5, 4, 4, 9])

        data = self._compute_features("Contains", "et", Sources.DOCUMENTS)
        np.testing.assert_array_almost_equal(data.X.flatten(), [2, 1, 0, 0])

        data = self._compute_features("Contains", "is", Sources.DOCUMENTS)
        np.testing.assert_array_almost_equal(data.X.flatten(), [1, 2, 2, 0])

        data = self._compute_features("Contains", "t", Sources.TOKENS)
        np.testing.assert_array_almost_equal(data.X.flatten(), [5, 4, 4, 9])

        data = self._compute_features("Contains", " ", Sources.TOKENS)
        np.testing.assert_array_almost_equal(data.X.flatten(), [0, 0, 0, 0])

        self.send_signal(self.widget.Inputs.corpus, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.corpus))

    def test_regex(self):
        """ Test regex statistic """
        # words that contain digit
        data = self._compute_features("Regex", r"\w*\d\w*", Sources.DOCUMENTS)
        np.testing.assert_array_almost_equal(data.X.flatten(), [0, 0, 0, 1])

        # words that contain is
        data = self._compute_features("Regex", r"\w*is\w*", Sources.DOCUMENTS)
        np.testing.assert_array_almost_equal(data.X.flatten(), [1, 2, 2, 0])

        # count specific n-gram
        data = self._compute_features("Regex", r"ipsum\ dolor", Sources.DOCUMENTS)
        np.testing.assert_array_almost_equal(data.X.flatten(), [1, 0, 0, 0])

        # words that contain digit
        data = self._compute_features("Regex", r"\w*\d\w*", Sources.TOKENS)
        np.testing.assert_array_almost_equal(data.X.flatten(), [0, 0, 0, 1])

        # words that contain is
        data = self._compute_features("Regex", r"\w*is\w*", Sources.TOKENS)
        np.testing.assert_array_almost_equal(data.X.flatten(), [1, 2, 2, 0])

        # count specific n-gram
        data = self._compute_features("Regex", r"ipsum\ dolor", Sources.TOKENS)
        np.testing.assert_array_almost_equal(data.X.flatten(), [0, 0, 0, 0])

        self.send_signal(self.widget.Inputs.corpus, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.corpus))

    def test_pos(self):
        """
        Test post tags count
        - test with corpus that has no pos tags - warning raised
        - test with corpus that has pos tags
        """
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self._set_feature("POS tag", "NN", Sources.TOKENS)
        self.widget.apply()
        self.wait_until_finished()
        res = self.get_output(self.widget.Outputs.corpus)
        self.assertEqual(0, res.X.shape[1])
        self.assertTrue(self.widget.Warning.not_computed.is_shown())

        tagger = AveragedPerceptronTagger()
        result = tagger(self.corpus)

        self.send_signal(self.widget.Inputs.corpus, result)
        self._set_feature("POS tag", "NN", Sources.TOKENS)
        self.widget.apply()
        self.wait_until_finished()
        res = self.get_output(self.widget.Outputs.corpus)
        self.assertTupleEqual((len(self.corpus), 1), res.X.shape)
        np.testing.assert_array_almost_equal(res.X.flatten(), [6, 5, 4, 5])
        self.assertFalse(self.widget.Warning.not_computed.is_shown())

    def test_yule(self):
        """
        Test Yule's I - complexity index.
        - test with corpus that has no pos tags - warning raised
        - test with corpus that has pos tags
        """
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self._set_feature("Yule's I", source=Sources.TOKENS)
        self.widget.apply()
        self.wait_until_finished()
        res = self.get_output(self.widget.Outputs.corpus)
        self.assertEqual(0, res.X.shape[1])
        self.assertTrue(self.widget.Warning.not_computed.is_shown())

        with self.corpus.unlocked():
            self.corpus[1][-1] = "simple"
        tagger = AveragedPerceptronTagger()
        result = tagger(self.corpus)

        self.send_signal(self.widget.Inputs.corpus, result)
        self._set_feature("Yule's I", source=Sources.TOKENS)
        self.widget.apply()
        self.wait_until_finished()
        res = self.get_output(self.widget.Outputs.corpus)
        self.assertTupleEqual((len(self.corpus), 1), res.X.shape)
        # the second document will have lower complexity than the first one
        self.assertLess(res[1][0], res[0][0])
        self.assertFalse(self.widget.Warning.not_computed.is_shown())

    def test_lix(self):
        """
        Test LIX readability score.
        """
        with self.corpus.unlocked():
            self.corpus[1][-1] = "simple. simple."
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self._set_feature("LIX index", source=Sources.TOKENS)
        self.widget.apply()
        self.wait_until_finished()
        res = self.get_output(self.widget.Outputs.corpus)
        self.assertTupleEqual((len(self.corpus), 1), res.X.shape)
        # the second document will have lower complexity than the first one
        self.assertLess(res[1][0], res[0][0])

    def test_stats_different_preprocessing(self):
        pp = [LowercaseTransformer(), RegexpTokenizer(), StopwordsFilter(language="en")]
        pp = PreprocessorList(pp)
        self.corpus = pp(self.corpus)

        data = self._compute_features("Character count", "", Sources.TOKENS)
        np.testing.assert_array_almost_equal(data.X.flatten(), [47, 44, 46, 51])

        data = self._compute_features("N-gram count", "", Sources.TOKENS)
        np.testing.assert_array_almost_equal(data.X.flatten(), [8, 9, 9, 9])

        data = self._compute_features("Per cent unique terms", "", Sources.TOKENS)
        np.testing.assert_array_almost_equal(data.X.flatten(), [1, 1, 1, 1])

        # none start with the capital because of Lowercase preprocessor
        data = self._compute_features("Starts with", "L", Sources.TOKENS)
        np.testing.assert_array_almost_equal(data.X.flatten(), [0, 0, 0, 0])

        data = self._compute_features("Starts with", "a", Sources.TOKENS)
        np.testing.assert_array_almost_equal(data.X.flatten(), [2, 0, 0, 2])

        data = self._compute_features("Ends with", "a", Sources.TOKENS)
        np.testing.assert_array_almost_equal(data.X.flatten(), [0, 1, 2, 1])

        # non contain comma since we use RegexP preprocessor
        data = self._compute_features("Contains", ",", Sources.TOKENS)
        np.testing.assert_array_almost_equal(data.X.flatten(), [0, 0, 0, 0])

        data = self._compute_features("Contains", "a", Sources.TOKENS)
        np.testing.assert_array_almost_equal(data.X.flatten(), [2, 2, 6, 5])

        data = self._compute_features("Regex", "{e", Sources.TOKENS)
        np.testing.assert_array_almost_equal(data.X.flatten(), [0, 0, 0, 0])

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
            (wc_index, "", Sources.DOCUMENTS),
            (starts_with_index, "a", Sources.TOKENS),
            (capital_counts_index, "", Sources.DOCUMENTS),
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

        self.widget.active_rules = [(1, "", Sources.DOCUMENTS)]
        self.widget.adjust_n_rule_rows()
        self.widget.apply()
        self.wait_until_finished()

        expected = [(1, "", Sources.DOCUMENTS)]
        self.assertListEqual(expected, list(self.widget.result_dict.keys()))

        self.widget.active_rules = [(1, "", Sources.DOCUMENTS), (2, "", Sources.TOKENS)]
        self.widget.adjust_n_rule_rows()
        self.widget.apply()
        self.wait_until_finished()

        expected = [(1, "", Sources.DOCUMENTS), (2, "", Sources.TOKENS)]
        self.assertListEqual(expected, list(self.widget.result_dict.keys()))

        self.widget.active_rules = [(2, "", Sources.TOKENS)]
        self.widget.adjust_n_rule_rows()
        self.widget.apply()
        self.wait_until_finished()

        expected = [(2, "", Sources.TOKENS)]
        self.assertListEqual(expected, list(self.widget.result_dict.keys()))

        # dict should empty on new data
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.assertListEqual([], list(self.widget.result_dict.keys()))

    def test_settings(self):
        """Test whether context correctly restore rules"""
        doc, tk = Sources.DOCUMENTS, Sources.TOKENS
        rules = [(0, "", doc), (1, "", doc), (2, "", tk)]
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.widget.active_rules = rules[:]

        self.send_signal(self.widget.Inputs.corpus, self.book_data)
        expected = [(0, "", doc), (1, "", doc), (2, "", tk)]
        self.assertListEqual(expected, self.widget.active_rules)

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
        self.assertListEqual([(0, "", Sources.DOCUMENTS)], self.widget.active_rules)

    def test_remove_row(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.widget.active_rules = [(0, "", Sources.DOCUMENTS)]
        self.widget.adjust_n_rule_rows()
        self.assertListEqual([(0, "", Sources.DOCUMENTS)], self.widget.active_rules)

        remove_button = [
            x
            for x in self.widget.controlArea.findChildren(QPushButton)
            if x.text() == "×"
        ][0]
        remove_button.click()
        self.assertListEqual([], self.widget.active_rules)

    def test_migrate_settings(self):
        vals = [""] * 6 + ["a,e", "b,c", "", "a", "b", "c", r"\w*is", "NN,VV", "", ""]
        settings = {"__version__": 1, "active_rules": list(zip(range(17), vals))}
        widget = self.create_widget(OWStatistics, stored_settings=settings)
        self.send_signal(self.widget.Inputs.corpus, self.corpus, widget=widget)

        expected = [
            (0, "", Sources.DOCUMENTS),
            (1, "", Sources.DOCUMENTS),
            (2, "", Sources.TOKENS),
            (3, "", Sources.DOCUMENTS),
            (4, "", Sources.DOCUMENTS),
            (5, "", Sources.DOCUMENTS),
            (6, "a,e", Sources.DOCUMENTS),
            (7, "b,c", Sources.DOCUMENTS),
            (8, "", Sources.TOKENS),
            (9, "a", Sources.TOKENS),
            (10, "b", Sources.TOKENS),
            (11, "c", Sources.DOCUMENTS),
            (12, r"\w*is", Sources.DOCUMENTS),
            (13, "NN,VV", Sources.TOKENS),
            (14, "", Sources.TOKENS),
            (15, "", Sources.TOKENS),
        ]
        self.assertListEqual(expected, widget.active_rules)

    def test_preprocess_output(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        output = self.get_output(self.widget.Outputs.corpus)
        imputed = SklImpute()(output)
        self.assertIsNotNone(imputed)


if __name__ == "__main__":
    unittest.main()
