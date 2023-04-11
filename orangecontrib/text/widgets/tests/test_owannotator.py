# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring,arguments-differ
import unittest
from itertools import chain
from unittest.mock import Mock, patch

import numpy as np
from AnyQt.QtCore import QRectF, QPointF

from Orange.data import Domain, Table
from Orange.projection import PCA
from Orange.widgets.tests.base import WidgetTest, simulate
from Orange.widgets.unsupervised.owtsne import OWtSNE
from Orange.widgets.unsupervised.tests.test_owtsne import DummyTSNE, \
    DummyTSNEModel
from orangecontrib.text import Corpus
from orangecontrib.text.preprocess import LowercaseTransformer, \
        RegexpTokenizer, StopwordsFilter, FrequencyFilter
from orangecontrib.text.vectorization import BowVectorizer
from orangecontrib.text.widgets.owannotator import OWAnnotator


def preprocess(corpus: Corpus) -> Corpus:
    for pp in (LowercaseTransformer(), RegexpTokenizer(r"\w+"),
               StopwordsFilter(), FrequencyFilter(0.25, 0.5)):
        corpus = pp(corpus)

    transformed_corpus = BowVectorizer().transform(corpus)

    pca = PCA(n_components=2)
    pca_model = pca(transformed_corpus)
    projection = pca_model(transformed_corpus)

    domain = Domain(
        transformed_corpus.domain.attributes,
        transformed_corpus.domain.class_vars,
        chain(transformed_corpus.domain.metas,
              projection.domain.attributes)
    )
    return corpus.transform(domain)


class TestOWAnnotator(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.corpus = preprocess(Corpus.from_file("book-excerpts"))

    def setUp(self):
        self.widget = self.create_widget(OWAnnotator)

    def _select_data(self):
        rect = QRectF(QPointF(-20, -20), QPointF(20, 20))
        self.widget.graph.select_by_rectangle(rect)
        return self.widget.graph.get_selection()

    def test_output_data_type(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus[::20])
        self.wait_until_finished()
        self._select_data()
        annotated = self.get_output(self.widget.Outputs.annotated_data)
        selected = self.get_output(self.widget.Outputs.selected_data)
        self.assertIsInstance(annotated, Corpus)
        self.assertIsInstance(selected, Corpus)

    def test_output_data_domain(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus[::10])
        self.wait_until_finished()

        annotated = self.get_output(self.widget.Outputs.annotated_data)
        self.assertIn("Cluster", [a.name for a in annotated.domain.metas])

        self.widget.controls.clustering_type.buttons[2].click()
        annotated = self.get_output(self.widget.Outputs.annotated_data)
        self.assertNotIn("Cluster", [a.name for a in annotated.domain.metas])

    def test_output_scores_type(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus[::20])
        self.wait_until_finished()
        scores = self.get_output(self.widget.Outputs.scores)
        self.assertIsInstance(scores, Table)

        self.send_signal(self.widget.Inputs.corpus, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.scores))

    def test_output_scores_domain(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus[::20])
        self.wait_until_finished()

        annotated = self.get_output(self.widget.Outputs.annotated_data)
        cluster_labels = annotated.domain["Cluster"].values
        scores = self.get_output(self.widget.Outputs.scores)
        self.assertIn("Words", [a.name for a in scores.domain.metas])

        attrs = scores.domain.attributes
        for label in cluster_labels:
            self.assertIn(f"Score({label})", [a.name for a in attrs])
            self.assertIn(f"p_value({label})", [a.name for a in attrs])

    def test_axis_controls(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus[::30])
        simulate.combobox_activate_index(self.widget.controls.attr_y, 0)
        self.assertTrue(self.widget.Warning.same_axis_features.is_shown())
        self.send_signal(self.widget.Inputs.corpus, None)
        self.assertFalse(self.widget.Warning.same_axis_features.is_shown())

    def test_epsilon_control(self):
        self.widget._run = Mock()
        self.send_signal(self.widget.Inputs.corpus, self.corpus[::30])
        self.widget._run.assert_called_once()
        self.assertFalse(self.widget.controls.epsilon.isEnabled())
        self.widget.controls.use_epsilon.click()
        self.assertTrue(self.widget.controls.epsilon.isEnabled())
        self.widget._run.assert_called_once()

    def test_n_components_control(self):
        self.widget._run = Mock()
        self.send_signal(self.widget.Inputs.corpus, self.corpus[::30])
        self.widget._run.assert_called_once()

        self.widget._run.reset_mock()
        self.widget.controls.clustering_type.buttons[1].click()
        self.assertFalse(self.widget.controls.n_components.isEnabled())
        self.widget._run.assert_called_once()

        self.widget.controls.use_n_components.click()
        self.assertTrue(self.widget.controls.n_components.isEnabled())
        self.widget._run.assert_called_once()

    def test_cluster_var_control(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus[::10])
        self.wait_until_finished()
        cluster_var_button = self.widget.controls.clustering_type.buttons[2]
        self.assertTrue(cluster_var_button.isEnabled())
        self.assertFalse(self.widget.controls.cluster_var.isEnabled())
        cluster_var_button.click()
        self.assertTrue(self.widget.controls.cluster_var.isEnabled())

        domain = self.corpus.domain
        domain = Domain(domain.attributes, metas=domain.metas)
        corpus = self.corpus[::10].transform(domain)
        self.send_signal(self.widget.Inputs.corpus, corpus)
        self.assertFalse(cluster_var_button.isEnabled())

    def test_cluster_var_control_subset(self):
        corpus = self.corpus[::30]
        mask = corpus.Y == 1
        corpus = corpus[mask]
        self.send_signal(self.widget.Inputs.corpus, corpus)
        cluster_var_button = self.widget.controls.clustering_type.buttons[2]
        cluster_var_button.click()
        self.wait_until_finished()
        self.assertFalse(self.widget.Error.proj_error.is_shown())

    def test_color_by_cluster_control(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus[::30])
        self.wait_until_finished()
        self.assertTrue(self.widget.controls.attr_color.isEnabled())
        self.widget.controls.color_by_cluster.click()
        self.assertFalse(self.widget.controls.attr_color.isEnabled())

    def test_missing_embedding(self):
        corpus = self.corpus[::30].copy()
        with corpus.unlocked():
            corpus[:, "PC1"] = np.nan
            corpus[:, "PC2"] = np.nan
        self.send_signal(self.widget.Inputs.corpus, corpus)
        self.wait_until_finished()
        self.assertTrue(self.widget.Error.proj_error.is_shown())
        self.send_signal(self.widget.Inputs.corpus, None)
        self.assertFalse(self.widget.Error.proj_error.is_shown())

    def test_plot_once(self):
        side_effect = self.widget.setup_plot
        self.widget.setup_plot = Mock(side_effect=side_effect)
        self.send_signal(self.widget.Inputs.corpus, self.corpus[::30])
        self.widget.setup_plot.assert_called_once()
        self.wait_until_finished()
        self.widget.setup_plot.assert_called_once()

    def test_send_data_once(self):
        self.widget.send_data = Mock()
        self.send_signal(self.widget.Inputs.corpus, self.corpus[::20])
        self.widget.send_data.assert_called_once()
        self.widget.send_data.reset_mock()
        self.wait_until_finished()
        self.widget.send_data.assert_called_once()

    def test_saved_selection(self):
        c = self.corpus[::10]
        self.send_signal(self.widget.Inputs.corpus, c)
        self.wait_until_finished()

        indices = list(range(0, len(c), 5))
        self.widget.graph.select_by_indices(indices)
        settings = self.widget.settingsHandler.pack_data(self.widget)
        widget = self.create_widget(self.widget.__class__,
                                    stored_settings=settings)

        self.send_signal(widget.Inputs.corpus, c, widget=widget)
        self.wait_until_finished(widget=widget)

        self.assertEqual(np.sum(widget.graph.selection), 3)
        np.testing.assert_equal(self.widget.graph.selection,
                                widget.graph.selection)

    def test_attr_label_metas(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus[::20])
        self.wait_until_finished()
        simulate.combobox_activate_item(self.widget.controls.attr_label,
                                        self.corpus.domain[-1].name)

    def test_attr_models(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus[::30])
        self.wait_until_finished()
        controls = self.widget.controls
        self.assertEqual(len(controls.attr_x.model()), 2)
        self.assertEqual(len(controls.attr_y.model()), 2)
        self.assertEqual(len(controls.cluster_var.model()), 1)
        self.assertEqual(len(controls.attr_color.model()), 6)
        self.assertEqual(len(controls.attr_shape.model()), 3)
        self.assertEqual(len(controls.attr_size.model()), 4)
        self.assertEqual(len(controls.attr_label.model()), 7)

    @patch("Orange.projection.manifold.TSNE", DummyTSNE)
    @patch("Orange.projection.manifold.TSNEModel", DummyTSNEModel)
    def test_tsne_output(self):
        owtsne = self.create_widget(OWtSNE)
        self.send_signal(owtsne.Inputs.data, self.corpus, widget=owtsne)
        self.wait_until_finished(widget=owtsne)
        tsne_output = self.get_output(owtsne.Outputs.annotated_data, owtsne)

        self.send_signal(self.widget.Inputs.corpus, tsne_output)
        self.wait_until_finished()
        output = self.get_output(self.widget.Outputs.annotated_data)
        self.assertIsNotNone(output)

    def test_send_report(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus[::50])
        self.widget.report_button.click()
        self.wait_until_finished()
        self.widget.report_button.click()
        self.send_signal(self.widget.Inputs.corpus, None)
        self.widget.report_button.click()

    def test_no_disc_var_context(self):
        domain = Domain(self.corpus.domain.attributes,
                        metas=self.corpus.domain.metas)
        corpus = self.corpus[::10].transform(domain)
        self.send_signal(self.widget.Inputs.corpus, corpus)
        self.wait_until_finished()
        self.assertIsNone(self.widget.cluster_var)

        self.send_signal(self.widget.Inputs.corpus, self.corpus[::10])
        self.wait_until_finished()
        self.widget.controls.clustering_type.buttons[2].click()
        self.assertIsNotNone(self.widget.cluster_var)

    def test_invalidate(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus[::4])

        self.wait_until_finished()
        self.assertEqual(len(self.widget.clusters.groups), 1)

        self.widget.controls.clustering_type.buttons[1].click()
        self.wait_until_finished()
        self.assertEqual(len(self.widget.clusters.groups), 8)

        self.widget.controls.use_n_components.setChecked(True)
        self.widget.controls.n_components.setValue(4)
        self.wait_until_finished()
        self.assertEqual(len(self.widget.clusters.groups), 4)

        self.widget.controls.clustering_type.buttons[2].click()
        self.wait_until_finished()
        self.assertEqual(len(self.widget.clusters.groups), 2)


if __name__ == "__main__":
    unittest.main()
