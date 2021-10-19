import pandas as pd
import numpy as np

from orangecontrib.text.topics import Topics
from orangecontrib.text.widgets.owldavis import OWLDAvis
from orangewidget.tests.base import WidgetTest


class TestStatisticsWidget(WidgetTest):
    def setUp(self) -> None:
        self.widget = self.create_widget(OWLDAvis)
        self.topics = Topics.from_file("data/LDAvis/LDAtopics.tab")
        self._load_comparable_data()

    def _load_comparable_data(self) -> None:
        """
        Load results for Topic1 from pyLDAvis. Save it to `self.topic1`
        """
        self.topic1 = pd.read_csv("data/LDAvis/topic1.csv")

    def test_send_data(self):
        """ Test with basic data, and empty data """
        self.send_signal(self.widget.Inputs.topics, self.topics)
        # self.widget.data is transposed, compare accordingly
        self.assertEqual(len(self.topics),
                         len(self.widget.data.domain.variables))
        self.assertEqual(len(self.topics.domain.variables),
                         len(self.widget.data))

        self.send_signal(self.widget.Inputs.topics, None)
        self.assertIsNone(self.widget.data)

    def test_relevance(self):
        # results for first 10 terms from pyLDAvis for Topic1 with lambda 0.5
        df = pd.DataFrame({"Term": ["minors", "ordering", "quasi", "well",
                                    "widths", "iv", "survey", "human",
                                    "interface", "computer"],
                           "relevance": [-0.19905, -0.74485, -0.74485, -0.74485,
                                         -0.74485, -0.74485, -1.36715, -2.03665,
                                         -2.18435, -2.35890]})

        self.send_signal(self.widget.Inputs.topics, self.topics)
        topic1 = pd.DataFrame({"Term": self.widget.shown_words[:10],
                               "relevance": self.widget.shown_weights[:10]})
        topic1.sort_values(["relevance", "Term"], ascending=[False, True],
                           inplace=True)
        np.testing.assert_array_almost_equal(df["relevance"],
                                             topic1["relevance"], decimal=4)

    def test_term_distribution(self):
        """ Test that distribution of bars matches the one in pyLDAvis """
        self.send_signal(self.widget.Inputs.topics, self.topics)
        ratio = self.widget.shown_term_topic_freq / self.widget.shown_marg_prob
        np.testing.assert_array_almost_equal(self.topic1["Freq"]/self.topic1[
            "Total"], ratio, decimal=4)
