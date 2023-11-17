import unittest

import numpy as np
from Orange.data import Domain, ContinuousVariable
from orangecontrib.text import Corpus
from orangecontrib.text.annotate_documents import _get_characteristic_terms, \
    _hypergeom_clusters, annotate_documents, ClusterDocuments


def add_embedding(corpus: Corpus, num_clusters: int) -> Corpus:
    """
    Generate random points around cluster centers to have reproducible clusters
    this solution is resistant to changes in random_seeds by scikit
    """
    new_domain = Domain(
        corpus.domain.attributes,
        corpus.domain.class_vars,
        corpus.domain.metas + (ContinuousVariable("PC1"), ContinuousVariable("PC2"))
    )
    corpus = corpus.transform(new_domain)

    cluster_centers = np.array([(x * 10, x * 10) for x in range(num_clusters)])
    points = np.random.randn(len(corpus), 2)
    cluster_assignments = np.tile(
        np.arange(len(cluster_centers)), len(corpus) // num_clusters + 1
    )[:len(corpus)]
    points += cluster_centers[cluster_assignments]  # move points to its cluster center
    with corpus.unlocked(corpus.metas):
        corpus[:, ["PC1", "PC2"]] = points
    return corpus


class TestClusterDocuments(unittest.TestCase):
    def setUp(self):
        self.corpus = add_embedding(Corpus.from_file("deerwester"), 2)

    def test_gmm(self):
        labels = ClusterDocuments.gmm(self.corpus.metas[:, -2:], 2, 0.6)
        self.assertIn(
            list(labels), ([0, 1, 0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1, 0, 1])
        )

    def test_gmm_n_comp(self):
        emb = self.corpus.metas[:, -2:]
        n = ClusterDocuments.gmm_compute_n_components(emb)
        self.assertEqual(n, 2)

    def test_dbscan(self):
        labels = ClusterDocuments.dbscan(self.corpus.metas[:, -2:], 5)
        self.assertEqual([0, -1, 0, -1, 0, -1, 0, -1, 0], list(labels))


class TestAnnotateDocuments(unittest.TestCase):
    def setUp(self):
        self.corpus = add_embedding(Corpus.from_file("deerwester"), 3)

    def test_get_characteristic_terms(self):
        keywords = _get_characteristic_terms(self.corpus, 4)
        keywords = [[w for w, _ in doc_keywords] for doc_keywords in keywords]
        self.assertEqual(["abc", "applications", "for", "lab"], keywords[0])

    def test_hypergeom_clusters(self):
        labels = ClusterDocuments.gmm(self.corpus.metas[:, -2:], 3, 0.6)
        keywords = _get_characteristic_terms(self.corpus, 4)
        selected_clusters_keywords, all_keywords, scores, p_values = \
            _hypergeom_clusters(labels, keywords, 0.05, 3)
        self.assertEqual(len(selected_clusters_keywords), len(set(labels)))
        self.assertEqual(scores.shape, p_values.shape)
        self.assertEqual(scores.shape, (len(set(labels)), len(all_keywords)))

    def test_annotate_documents(self):
        embedding = self.corpus.metas[:, -2:]
        n_components = 3
        n_words_in_cluster = 5

        labels, clusters, n_comp, eps, _ = annotate_documents(
            self.corpus, embedding, ClusterDocuments.GAUSSIAN_MIXTURE,
            n_components=n_components,
            fdr_threshold=1.1,
            n_words_in_cluster=n_words_in_cluster
        )
        self.assertEqual(len(labels), len(self.corpus))
        self.assertEqual(len(clusters), n_components)
        self.assertEqual(n_components, n_comp)
        self.assertIsNone(eps)
        self.assertEqual(len(clusters[0]), 3)
        self.assertEqual(len(clusters[0][0]), n_words_in_cluster)


if __name__ == "__main__":
    unittest.main()
