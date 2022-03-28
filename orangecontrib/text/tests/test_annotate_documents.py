import unittest
from itertools import chain

from Orange.data import Domain
from Orange.projection import PCA
from orangecontrib.text import Corpus
from orangecontrib.text.annotate_documents import _get_characteristic_terms, \
    _hypergeom_clusters, annotate_documents, ClusterDocuments
from orangecontrib.text.vectorization import BowVectorizer


def add_embedding(corpus: Corpus) -> Corpus:
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


class TestClusterDocuments(unittest.TestCase):
    def setUp(self):
        self.corpus = add_embedding(Corpus.from_file("deerwester"))

    def test_gmm(self):
        labels = ClusterDocuments.gmm(self.corpus.metas[:, -2:], 3, 0.6)
        self.assertEqual([2, 1, 2, 2, 1, 0, 0, 0, 0], list(labels))

    def test_gmm_n_comp(self):
        emb = self.corpus.metas[:, -2:]
        n = ClusterDocuments.gmm_compute_n_components(emb)
        self.assertEqual(n, 3)

    def test_dbscan(self):
        labels = ClusterDocuments.dbscan(self.corpus.metas[:, -2:], 2)
        self.assertEqual([0, -1, 0, 0, -1, 0, 0, 0, 0], list(labels))


class TestAnnotateDocuments(unittest.TestCase):
    def setUp(self):
        self.corpus = add_embedding(Corpus.from_file("deerwester"))

    def test_get_characteristic_terms(self):
        keywords = _get_characteristic_terms(self.corpus, 4)
        keywords = [[w for w, _ in doc_keywords] for doc_keywords in keywords]
        self.assertEqual(["applications", "abc", "lab", "for"], keywords[0])

    def test_hypergeom_clusters(self):
        labels = ClusterDocuments.gmm(self.corpus.metas[:, -2:], 3, 0.6)
        keywords = _get_characteristic_terms(self.corpus, 4)
        keywords = _hypergeom_clusters(labels, keywords, 0.05, 3)
        self.assertEqual(len(keywords), len(set(labels)))

    def test_annotate_documents(self):
        embedding = self.corpus.metas[:, -2:]
        n_components = 3
        n_words_in_cluster = 5

        labels, clusters, n_comp, eps = annotate_documents(
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
