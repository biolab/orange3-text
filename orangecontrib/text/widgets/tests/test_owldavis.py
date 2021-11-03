import os
import unittest

import pandas as pd

from orangecontrib.text.topics import Topics
from orangecontrib.text.widgets.owldavis import OWLDAvis
from orangewidget.tests.base import WidgetTest


class TestStatisticsWidget(WidgetTest):
    def setUp(self) -> None:
        self.widget = self.create_widget(OWLDAvis)
        self.topics = Topics.from_file(
            os.path.join(os.path.dirname(__file__), "data/LDAvis/LDAtopics.tab")
        )
        self._load_comparable_data()

    def _load_comparable_data(self) -> None:
        """
        Load results for Topic1 from pyLDAvis. Save it to `self.topic1`
        """
        self.topic1 = pd.read_csv(
            os.path.join(os.path.dirname(__file__),  "data/LDAvis/topic1.csv"),
            index_col="Term"
        )

    def test_send_data(self):
        """Test with basic data, and empty data"""
        self.send_signal(self.widget.Inputs.topics, self.topics)
        # self.widget.data is transposed, compare accordingly
        self.assertEqual(len(self.topics), len(self.widget.data.domain.variables))
        self.assertEqual(len(self.topics.domain.variables), len(self.widget.data))

        self.send_signal(self.widget.Inputs.topics, None)
        self.assertIsNone(self.widget.data)

    def _get_graph_labels(self):
        """Get y axis ticks which correspond to words"""
        return [v for _, v in self.widget.graph.getAxis("left")._tickLevels[0][::-1]]

    def test_frequencies(self):
        """
        Test if compute frequencies are correct - in self.topic1 are results
        that correspond to values from original PyLDAvis implementation
        """
        self.send_signal(self.widget.Inputs.topics, self.topics)
        self.wait_until_finished()

        topic1 = pd.DataFrame(
            {
                "Term": self._get_graph_labels(),
                "Freq": self.widget.graph.term_topic_freq_item.opts["width"][::-1],
                "Total": self.widget.graph.marg_prob_item.opts["width"][::-1],
            }
        )
        topic1 = topic1.set_index("Term")
        pd.testing.assert_frame_equal(topic1, self.topic1)

    def test_topic_changed(self):
        """Test if topic is changed correctly"""
        self.send_signal(self.widget.Inputs.topics, self.topics)
        self.wait_until_finished()
        self.assertEqual(self.widget.selected_topic, 0)

        self.widget.topic_box.setCurrentRow(1)
        self.assertEqual(self.widget.selected_topic, 1)
        self.assertListEqual(
            ["trees", "widths", "quasi", "iv", "random"], self._get_graph_labels()[:5]
        )

    def test_relevance_changed(self):
        """Test different relevances"""
        self.send_signal(self.widget.Inputs.topics, self.topics)
        self.wait_until_finished()

        self.assertEqual(self.widget.relevance, 0.5)
        self.assertListEqual(
            ["system", "eps", "human", "interface", "engineering", "abc"],
            self._get_graph_labels()[:6],
        )

        self.widget.rel_slider.setValue(0)
        self.widget.on_params_change()
        self.assertEqual(self.widget.relevance, 0)
        self.assertListEqual(
            ["eps", "human", "interface", "system", "engineering", "abc"],
            self._get_graph_labels()[:6],
        )

        self.widget.rel_slider.setValue(1)
        self.widget.on_params_change()
        self.assertEqual(self.widget.relevance, 1)
        self.assertListEqual(
            ["system", "eps", "human", "interface", "of", "computer"],
            self._get_graph_labels()[:6],
        )

    def test_report(self):
        self.send_signal(self.widget.Inputs.topics, self.topics)
        self.wait_until_finished()

        self.widget.send_report()


if __name__ == "__main__":
    unittest.main()
