import unittest

import numpy as np
from AnyQt.QtCore import QItemSelectionModel

from Orange.widgets.tests.base import WidgetTest
from orangecontrib.text.corpus import Corpus
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

    def test_saved_selection(self):
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

        # test LDA, which is the only one with log perplexity
        self.assertNotEqual(self.widget.perplexity, "n/a")
        self.assertTrue(self.widget.coherence)

        # test LSI
        self.widget.method_index = 1
        self.widget.commit.now()
        self.wait_until_finished()
        self.assertEqual(self.widget.perplexity, "n/a")
        self.assertNotEqual(self.widget.coherence, "n/a")

    def test_migrate_settings_transform(self):
        # 0 used to be LSI in version <2 - it is on index 1 now
        settings = {"__version__": 1, "method_index": 0}
        widget = self.create_widget(OWTopicModeling, stored_settings=settings)
        self.assertEqual(1, widget.method_index)
        self.assertEqual("Latent Semantic Indexing", widget.model.name)

        # 1 used to be LDA in version <2 - it is on index 0 now
        settings = {"__version__": 1, "method_index": 1}
        widget = self.create_widget(OWTopicModeling, stored_settings=settings)
        self.assertEqual(0, widget.method_index)
        self.assertEqual("Latent Dirichlet Allocation", widget.model.name)

        # 2 is unchanged - still HDP
        settings = {"__version__": 1, "method_index": 2}
        widget = self.create_widget(OWTopicModeling, stored_settings=settings)
        self.assertEqual(2, widget.method_index)
        self.assertEqual("Hierarchical Dirichlet Process", widget.model.name)

        # 2 is unchanged - still NMF
        settings = {"__version__": 1, "method_index": 3}
        widget = self.create_widget(OWTopicModeling, stored_settings=settings)
        self.assertEqual(3, widget.method_index)
        self.assertEqual("Negative Matrix Factorization", widget.model.name)


if __name__ == "__main__":
    unittest.main()
