import unittest
from unittest.mock import Mock

import numpy as np
import pkg_resources

from Orange.widgets.tests.base import WidgetTest
from Orange.data import StringVariable, Domain
from scipy.sparse import csr_matrix

from orangecontrib.text.corpus import Corpus
from orangecontrib.text.topics import Topic
from orangecontrib.text.widgets.owwordcloud import OWWordCloud


@unittest.skipIf(
    pkg_resources.get_distribution("orange3").version < "3.24.0",
    "Wait until finished not implemented in lower version"
)
class TestWordCloudWidget(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWWordCloud)
        self.corpus = Corpus.from_file('deerwester')

        self.topic = self.create_topic()

    def create_topic(self):
        words = [[f"a{i}"] for i in range(10)]
        weights = list(range(10))
        t = Topic.from_numpy(
            Domain([], metas=[
                StringVariable("Topic 1")
            ]),
            X=np.empty((10, 0)),
            metas=np.array(words),
            W=weights  #np.array(weights).reshape(-1, 1)
        )
        t.attributes["topic-method-name"] = "LsiModel"
        return t

    def test_data(self):
        """
        Just basic test.
        GH-244
        """
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.send_signal(self.widget.Inputs.corpus, None)
        self.wait_until_finished()

    def test_empty_data(self):
        """
        Widget crashes when receives zero length data.
        GH-244
        """
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.send_signal(self.widget.Inputs.corpus, self.corpus[:0])
        self.wait_until_finished()

    def test_bow_features(self):
        """
        When bag of words features are at the input word cloud must be made
        based on BOW weights.
        """
        data = self.corpus[:3]
        data = data.extend_attributes(
            csr_matrix([[3, 2, 0], [0, 3, 6], [0, 1, 0]]),
            ["Word1", "Word2", "Word3"])
        for v in data.domain.attributes:
            v.attributes["bow-feature"] = True

        self.send_signal(self.widget.Inputs.corpus, data)
        self.wait_until_finished()
        weights = list(zip(*sorted(self.widget.corpus_counter.items())))[1]
        # due to computation error in computing mean use array_almost_equal
        np.testing.assert_array_almost_equal(weights, [1, 2, 2])

        output = self.get_output(self.widget.Outputs.word_counts)
        np.testing.assert_array_almost_equal([2, 2, 1], output.X.flatten())
        np.testing.assert_array_equal(
            ["Word3", "Word2", "Word1"], output.metas.flatten())
        self.assertTupleEqual(
            ("Word3", "Word2", "Word1"),
            list(zip(*self.widget.tablemodel[:]))[1])
        np.testing.assert_array_almost_equal(
            [2, 2, 1],
            list(zip(*self.widget.tablemodel[:]))[0])

        # try with one word not bow-feature
        data = self.corpus[:3]
        data = data.extend_attributes(
            csr_matrix([[3, 2, 0], [0, 3, 6], [0, 1, 0]]),
            ["Word1", "Word2", "Word3"])
        for v in data.domain.attributes[:2]:
            v.attributes["bow-feature"] = True

        self.send_signal(self.widget.Inputs.corpus, data)
        self.wait_until_finished()
        weights = list(zip(*sorted(self.widget.corpus_counter.items())))[1]
        np.testing.assert_array_almost_equal(weights, [1, 2])

        output = self.get_output(self.widget.Outputs.word_counts)
        np.testing.assert_array_almost_equal([2, 1], output.X.flatten())
        np.testing.assert_array_equal(
            ["Word2", "Word1"], output.metas.flatten())
        self.assertTupleEqual(
            ("Word2", "Word1"),
            list(zip(*self.widget.tablemodel[:]))[1])
        np.testing.assert_array_almost_equal(
            [2, 1],
            list(zip(*self.widget.tablemodel[:]))[0])

    def test_bow_info(self):
        """
        Widget shows info when bow-features used. This test tests this info.
        """
        data = self.corpus[:3]

        # no data no info
        self.assertFalse(self.widget.Info.bow_weights.is_shown())
        self.send_signal(self.widget.Inputs.corpus, data)
        self.wait_until_finished()
        self.assertFalse(self.widget.Info.bow_weights.is_shown())
        self.send_signal(self.widget.Inputs.corpus, None)
        self.wait_until_finished()
        self.assertFalse(self.widget.Info.bow_weights.is_shown())

        # send bow data
        data = data.extend_attributes(
            csr_matrix([[3, 2, 0], [0, 3, 6], [0, 1, 0]]),
            ["Word1", "Word2", "Word3"])
        for v in data.domain.attributes:
            v.attributes["bow-feature"] = True
        self.send_signal(self.widget.Inputs.corpus, data)
        self.wait_until_finished()
        self.assertTrue(self.widget.Info.bow_weights.is_shown())
        self.send_signal(self.widget.Inputs.corpus, None)
        self.wait_until_finished()
        self.assertFalse(self.widget.Info.bow_weights.is_shown())

    def test_topic(self):
        self.send_signal(self.widget.Inputs.topic, self.topic)

        self.assertIsNotNone(self.widget.topic)
        self.assertEqual("a0", self.widget.wordlist[0][0])
        self.assertEqual(10, self.widget.wordlist[0][1])
        self.assertEqual("a9", self.widget.wordlist[9][0])
        self.assertEqual(40, self.widget.wordlist[9][1])

        self.assertListEqual(
            self.topic.metas[:, 0].tolist(), self.widget.shown_words.tolist())
        np.testing.assert_array_almost_equal(self.topic.W, self.widget.shown_weights)

    def test_input_summary(self):
        insum = self.widget.info.set_input_summary = Mock()

        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.wait_until_finished()
        insum.assert_called_with("42", "9 documents with 42 words\n")

        self.send_signal(self.widget.Inputs.topic, self.topic)
        self.wait_until_finished()
        insum.assert_called_with(
            "42 | 10", "9 documents with 42 words\n10 words in a topic.")

        self.send_signal(self.widget.Inputs.corpus, None)
        self.wait_until_finished()
        insum.assert_called_with(f"10", "10 words in a topic.")

        self.send_signal(self.widget.Inputs.topic, None)
        self.wait_until_finished()
        insum.assert_called_with(self.widget.info.NoInput)

        self.send_signal(self.widget.Inputs.topic, self.topic)
        self.wait_until_finished()
        insum.assert_called_with(f"10", "10 words in a topic.")

    def test_output_summary(self):
        outsum = self.widget.info.set_output_summary = Mock()

        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.wait_until_finished()
        outsum.assert_called_with(
            "0 | 0 | 42", "0 documents\n0 selected words\n42 words with counts"
        )

        self.send_signal(self.widget.Inputs.topic, self.topic)
        self.wait_until_finished()
        outsum.assert_called_with(
            "0 | 0 | 42", "0 documents\n0 selected words\n42 words with counts"
        )

        self.send_signal(self.widget.Inputs.corpus, None)
        self.wait_until_finished()
        outsum.assert_called_with(self.widget.info.NoOutput)

        self.send_signal(self.widget.Inputs.topic, None)
        self.wait_until_finished()
        outsum.assert_called_with(self.widget.info.NoOutput)

    def test_send_report(self):
        self.widget.send_report()
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.wait_until_finished()
        self.widget.send_report()

    def test_no_tokens(self):
        """
        In some very rare cases (when all text strings empty) word cloud all
        token lists empty. Widget must work in those cases.
        """
        self.corpus.metas = np.array([[" "]] * len(self.corpus))
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.wait_until_finished()


if __name__ == "__main__":
    unittest.main()
