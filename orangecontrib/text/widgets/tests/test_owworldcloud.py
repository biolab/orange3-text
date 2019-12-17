import unittest
import numpy as np

from Orange.widgets.tests.base import WidgetTest
from scipy.sparse import csr_matrix

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

    def test_bow_features(self):
        """
        When bag of words features are at the input word cloud must be made
        based on BOW weights.
        """
        data = self.corpus[:3]
        data.extend_attributes(
            csr_matrix([[3, 2, 0], [0, 3, 6], [0, 1, 0]]),
            ["Word1", "Word2", "Word3"])
        for v in data.domain.attributes:
            v.attributes["bow-feature"] = True

        self.send_signal(self.widget.Inputs.corpus, data)
        self.assertDictEqual(
            self.widget.corpus_counter, {"Word1": 1, "Word2": 2, "Word3": 2})
        output = self.get_output(self.widget.Outputs.word_counts)
        np.testing.assert_array_equal([2, 2, 1], output.X.flatten())
        np.testing.assert_array_equal(
            ["Word2", "Word3", "Word1"], output.metas.flatten())
        self.assertListEqual(
            [(2.0, 'Word2'), (2.0, 'Word3'), (1.0, 'Word1')],
            self.widget.tablemodel[:])

        # try with one word not bow-feature
        data = self.corpus[:3]
        data.extend_attributes(
            csr_matrix([[3, 2, 0], [0, 3, 6], [0, 1, 0]]),
            ["Word1", "Word2", "Word3"])
        for v in data.domain.attributes[:2]:
            v.attributes["bow-feature"] = True

        self.send_signal(self.widget.Inputs.corpus, data)
        self.assertDictEqual(
            self.widget.corpus_counter, {"Word1": 1, "Word2": 2})
        output = self.get_output(self.widget.Outputs.word_counts)
        np.testing.assert_array_equal([2, 1], output.X.flatten())
        np.testing.assert_array_equal(
            ["Word2", "Word1"], output.metas.flatten())
        self.assertListEqual(
            [(2.0, 'Word2'), (1.0, 'Word1')],
            self.widget.tablemodel[:])

    def test_bow_info(self):
        """
        Widget shows info when bow-features used. This test tests this info.
        """
        data = self.corpus[:3]

        # no data no info
        self.assertFalse(self.widget.Info.bow_weights.is_shown())
        self.send_signal(self.widget.Inputs.corpus, data)
        self.assertFalse(self.widget.Info.bow_weights.is_shown())
        self.send_signal(self.widget.Inputs.corpus, None)
        self.assertFalse(self.widget.Info.bow_weights.is_shown())

        # send bow data
        data.extend_attributes(
            csr_matrix([[3, 2, 0], [0, 3, 6], [0, 1, 0]]),
            ["Word1", "Word2", "Word3"])
        for v in data.domain.attributes:
            v.attributes["bow-feature"] = True
        self.send_signal(self.widget.Inputs.corpus, data)
        self.assertTrue(self.widget.Info.bow_weights.is_shown())
        self.send_signal(self.widget.Inputs.corpus, None)
        self.assertFalse(self.widget.Info.bow_weights.is_shown())


if __name__ == "__main__":
    unittest.main()
