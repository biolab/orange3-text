import unittest

import numpy as np
from AnyQt.QtCore import QItemSelectionModel

from Orange.widgets.tests.base import WidgetTest
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.widgets.owtopicmodeling import OWTopicModeling


class TestTopicModeling(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.corpus = Corpus.from_file('deerwester')

    def setUp(self):
        self.widget = self.create_widget(OWTopicModeling)

    def test_data(self):
        def until():
            return bool(self.get_output(self.widget.Outputs.selected_topic))

        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.process_events(until)

        self.send_signal(self.widget.Inputs.corpus, None)
        output = self.get_output(self.widget.Outputs.selected_topic)
        self.assertIsNone(output)

    def test_saved_selection(self):
        def until(widget=self.widget):
            return bool(self.get_output(widget.Outputs.selected_topic,
                                        widget=widget))

        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.process_events(until)
        idx = self.widget.topic_desc.model().index(2, 0)
        self.widget.topic_desc.selectionModel().select(
            idx, QItemSelectionModel.Rows | QItemSelectionModel.ClearAndSelect)
        output1 = self.get_output(self.widget.Outputs.selected_topic)
        state = self.widget.settingsHandler.pack_data(self.widget)

        w = self.create_widget(OWTopicModeling, stored_settings=state)
        self.send_signal(w.Inputs.corpus, self.corpus, widget=w)
        self.process_events(lambda: until(w))
        output2 = self.get_output(w.Outputs.selected_topic, widget=w)
        # gensim uses quicksort, so sorting is unstable
        m1 = output1.metas[output1.metas[:, 0].argsort()]
        m2 = output2.metas[output2.metas[:, 0].argsort()]
        # test words and weights separately, weights are not exactly equal
        self.assertTrue((m1[:, 0] == m2[:, 0]).all())
        np.testing.assert_allclose(m1[:, 1].astype(float),
                                   m2[:, 1].astype(float))

    def test_all_topics_output(self):
        # LSI produces 9 topics for deerwester, output should be 9
        def until(widget=self.widget):
            return bool(self.get_output(widget.Outputs.selected_topic,
                                        widget=widget))

        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.process_events(until)
        output = self.get_output(self.widget.Outputs.all_topics)
        self.assertEqual(len(output), self.widget.model.actual_topics)
        self.assertEqual(output.metas.shape[1],
                         self.widget.corpus.metas.shape[1] + 1)


if __name__ == "__main__":
    unittest.main()
