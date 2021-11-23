import unittest
from unittest import skipIf

import numpy as np
import gensim
from AnyQt.QtCore import QItemSelectionModel

from Orange.widgets.tests.base import WidgetTest
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.topics import Topics
from orangecontrib.text.widgets.owtopicmodeling import OWTopicModeling


class TestTopicModeling(WidgetTest):
    def setUp(self):
        self.corpus = Corpus.from_file("deerwester")
        self.widget = self.create_widget(OWTopicModeling)

    def test_data(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.wait_until_finished()

        self.send_signal(self.widget.Inputs.corpus, None)
        output = self.get_output(self.widget.Outputs.selected_topic)
        self.assertIsNone(output)

    @skipIf(gensim.__version__ <= "4.1.2", "lsi model does not have random_seed")
    def test_saved_selection(self):
        self.assertTrue(False)
        # LSI does not have random_seed in gensim 4.1.2 but will have in next version
        # when this test start to fail
        # - remove line above
        # - set random_seed to LSI model and set gensim requirement
        # - remove skip on this test
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.wait_until_finished()

        idx = self.widget.topic_desc.model().index(2, 0)
        self.widget.topic_desc.selectionModel().select(
            idx, QItemSelectionModel.Rows | QItemSelectionModel.ClearAndSelect)
        output1 = self.get_output(self.widget.Outputs.selected_topic)
        state = self.widget.settingsHandler.pack_data(self.widget)

        w = self.create_widget(OWTopicModeling, stored_settings=state)
        self.send_signal(w.Inputs.corpus, self.corpus, widget=w)

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

        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        output = self.get_output(self.widget.Outputs.all_topics)

        self.assertEqual(len(output), self.widget.model.actual_topics)
        self.assertEqual(output.metas.shape[1],
                         self.widget.corpus.metas.shape[1] + 1)

    def test_topic_evaluation(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.wait_until_finished()

        # test LSI
        self.assertEqual(self.widget.perplexity, "n/a")
        self.assertNotEqual(self.widget.coherence, "n/a")

        # test LDA, which is the only one with log perplexity
        self.widget.method_index = 1
        self.widget.commit()
        self.wait_until_finished()

        self.assertNotEqual(self.widget.perplexity, "n/a")
        self.assertTrue(self.widget.coherence)


if __name__ == "__main__":
    unittest.main()
