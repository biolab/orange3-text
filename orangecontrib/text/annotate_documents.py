"""
Module for documents annotation.
"""
from collections import Counter
from itertools import chain
from typing import List, Tuple, Dict, Callable, Optional

import numpy as np
from scipy.spatial import distance
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from Orange.data import dummy_callback
from Orange.util import wrap_callback
from orangecontrib.text import Corpus
from orangecontrib.text.hull import compute_hulls
from orangecontrib.text.keywords import tfidf_keywords
from orangecontrib.text.stats import hypergeom_p_values


KeywordsType = List[Tuple[str, float]]
CentroidType = Tuple[float, float]
ClusterType = Tuple[KeywordsType, CentroidType, np.ndarray]
ScoresType = Tuple[List[str], np.ndarray, np.ndarray]


def annotate_documents(
        corpus: Corpus,
        embedding: np.ndarray,
        clustering_method: int,
        n_components: Optional[int] = None,
        epsilon: Optional[float] = None,
        cluster_labels: Optional[np.ndarray] = None,
        fdr_threshold: float = 0.05,
        n_words_in_cluster: int = 10,
        progress_callback: Optional[Callable] = None
) -> Tuple[np.ndarray, Dict[int, ClusterType], int, float, ScoresType]:
    """
    Annotate documents in corpus, by performing clustering on the corpus and
    assigning characteristic terms to each cluster using Hypergeometric
    distribution.

    Return annotated clusters - for each cluster return a list of keywords
    with scores, cluster center coordinates and concave_hulls coordinates.
    Also return optimal values for n_components/epsilon if calculated and
    scores data (p-values and counts for all keywords).

    Parameters
    ----------
    corpus : Corpus
        Corpus to be annotated.
    embedding : np.ndarray of size len(corpus) × 2
        Usually tSNE projection of BoW of corpus.
    clustering_method : int
        0 for DBSCAN
        1 for Gaussian mixture models
        2 for custom clustering where cluster_labels are used
    n_components: int, optional, default = None
        Number of clusters for Gaussian mixture models. If None, set to the
        number of clusters with maximal silhouette.
    epsilon : float, optional, default = None
        epsilon for DBSCAN. If None, optimal value is computed.
    cluster_labels : np.ndarray, optional
        Custom cluster labels. Usually included in corpus.
    fdr_threshold : float, optional, default = 0.05
        hypergeom_p_values threshold
    n_words_in_cluster : int, optional, default = 10
        Number of characteristic terms in each cluster.
    progress_callback : callable, optional
        Progress callback.

    Returns
    -------
    cluster_labels : np.ndarray of size len(corpus)
        An array of floats (i.e. 0, 1, np.nan) that represent cluster labels
        for all documents in the corpus.
    clusters : dict
        Dictionary of keywords with scores, centroids and concave hulls
        for each cluster.
    n_components : int
        Optimal number of clusters for Gaussian mixture models, if the
        n_components is None, and clustering_method is
        ClusterDocuments.GAUSSIAN_MIXTURE. n_components otherwise.
    epsilon : float
        Optimal value for epsilon for DBSCAN, if the epsilon is None, and
        clustering_method is ClusterDocuments.DBSCAN. epsilon otherwise.
    scores : tuple
        Tuple of all keywords with p-values and counts.

    Raises
    ------
    ValueError when there are no clusters in the embedding.

    """
    if progress_callback is None:
        progress_callback = dummy_callback

    if clustering_method == ClusterDocuments.GAUSSIAN_MIXTURE:
        if n_components is None:
            n_components = ClusterDocuments.gmm_compute_n_components(
                embedding,
                wrap_callback(progress_callback, end=0.3)
            )
        n_components = min([n_components, len(embedding)])
        cluster_labels = ClusterDocuments.gmm(
            embedding,
            n_components=n_components,
            threshold=0.6
        )

    elif clustering_method == ClusterDocuments.DBSCAN:
        if epsilon is None:
            epsilon = ClusterDocuments.dbscan_compute_epsilon(embedding)
        cluster_labels = ClusterDocuments.dbscan(
            embedding,
            eps=epsilon
        )

    else:
        assert cluster_labels is not None
        cluster_labels[np.isnan(cluster_labels)] = -1

    if len(set(cluster_labels) - {-1}) == 0:
        raise ValueError("There are no clusters using current settings.")

    keywords = _get_characteristic_terms(
        corpus,
        n_keywords=20,
        progress_callback=wrap_callback(progress_callback, start=0.5)
    )
    clusters_keywords, all_keywords, scores, p_values = \
        _hypergeom_clusters(cluster_labels, keywords,
                            fdr_threshold, n_words_in_cluster)

    concave_hulls = compute_hulls(embedding, cluster_labels)

    centroids = {c: tuple(np.mean(concave_hulls[c], axis=0))
                 for c in set(cluster_labels) - {-1}}

    clusters = {int(key): (
        clusters_keywords[key],
        centroids[key],
        concave_hulls[key]
    ) for key in clusters_keywords}

    cluster_labels = cluster_labels.astype(float)
    cluster_labels[cluster_labels == -1] = np.nan

    scores = (all_keywords, scores, p_values)

    return cluster_labels, clusters, n_components, epsilon, scores


class ClusterDocuments:
    DBSCAN, GAUSSIAN_MIXTURE = TYPES = range(2)
    GAUSSIAN_KWARGS = {"covariance_type": "full",
                       "max_iter": 20, "random_state": 0}

    @staticmethod
    def gmm(
            embedding: np.ndarray,
            n_components: int,
            threshold: float
    ) -> np.ndarray:
        estimator = GaussianMixture(
            n_components=n_components,
            **ClusterDocuments.GAUSSIAN_KWARGS
        )
        estimator.fit(embedding)
        probs = estimator.predict_proba(embedding)
        cluster_labels = np.argmax(probs, axis=1)
        max_probs = np.max(probs, axis=1)
        cluster_labels[max_probs < threshold] = -1
        return cluster_labels

    @staticmethod
    def gmm_compute_n_components(
            embedding: np.ndarray,
            progress_callback: Optional[Callable] = None
    ) -> int:
        if progress_callback is None:
            progress_callback = dummy_callback

        scores = []
        max_n_clusters = min(len(embedding), 11)
        for n_clusters in range(2, max_n_clusters):
            progress_callback(n_clusters / max_n_clusters)
            estimator = GaussianMixture(
                n_components=n_clusters,
                **ClusterDocuments.GAUSSIAN_KWARGS
            )
            cluster_labels = estimator.fit_predict(embedding)
            silhouette_avg = silhouette_score(embedding, cluster_labels)
            scores.append(silhouette_avg)
        return np.argmax(scores) + 2

    @staticmethod
    def dbscan(
            embedding: np.ndarray,
            eps: float
    ) -> np.ndarray:
        clustering = DBSCAN(eps=eps)
        clustering.fit(embedding)
        return clustering.labels_

    @staticmethod
    def dbscan_compute_epsilon(
            embedding: np.ndarray,
            k: int = 10,
            skip: float = 0.1
    ) -> float:
        if len(embedding) > 1000:  # subsampling is required
            i = len(embedding) // 1000
            embedding = embedding[::i]

        d = distance.squareform(distance.pdist(embedding))
        k = min(k + 1, len(embedding) - 1)
        kth_point = np.argpartition(d, k, axis=1)[:, k]
        # k+1 since first one is item itself
        kth_dist = np.sort(d[np.arange(0, len(kth_point)), kth_point])

        # currently mark proportion equal to skip as a noise
        return kth_dist[-int(np.round(len(kth_dist) * skip))]


def _get_characteristic_terms(
        corpus: Corpus,
        n_keywords: int = 20,
        progress_callback: Callable = None
) -> List[List[Tuple[str, float]]]:
    keywords = tfidf_keywords(corpus, progress_callback)
    return [sorted(k, key=lambda x: x[1], reverse=True)[:n_keywords]
            for k in keywords]


def _hypergeom_clusters(
        cluster_labels: np.ndarray,
        keywords: List[List[str]],
        fdr_threshold: float,
        n_words: int
) -> Tuple[Dict[int, List[str]], np.ndarray, np.ndarray, np.ndarray]:
    keywords = [[w for w, _ in doc_keywords] for doc_keywords in keywords]

    clusters_keywords = {}
    for label in sorted(set(cluster_labels) - {-1}):
        indices = set(np.flatnonzero(cluster_labels == label))
        kwds = [k for i, k in enumerate(keywords) if i in indices]
        clusters_keywords[label] = kwds

    cv = CountVectorizer(tokenizer=lambda w: w, preprocessor=lambda w: w)
    X = cv.fit_transform(list(chain.from_iterable(clusters_keywords.values())))
    all_keywords = np.array(cv.get_feature_names_out())

    index = 0
    selected_clusters_keywords = {}
    all_scores, all_p_values = [], []
    for label, cls_kwds in clusters_keywords.items():
        # find words that should be specific for a group with hypergeom test
        n_docs = len(cls_kwds)
        p_values = hypergeom_p_values(X, X[index:index + n_docs])
        words = set(all_keywords[np.array(p_values) < fdr_threshold])

        # select only words with p-values less than threshold
        sel_words = [w for w in chain.from_iterable(cls_kwds)]
        sel_words = [w for w in sel_words if w in words]
        sel_words = [(w, c / n_docs) for w, c
                     in Counter(sel_words).most_common(n_words)]
        selected_clusters_keywords[label] = sel_words

        all_scores.append(X[index:index + n_docs].sum(axis=0) / n_docs)
        all_p_values.append(p_values)

        index += n_docs

    all_scores = np.vstack(all_scores)
    all_p_values = np.vstack(all_p_values)
    return selected_clusters_keywords, all_keywords, all_scores, all_p_values


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from Orange.data import Domain
    from Orange.projection import PCA
    from orangecontrib.text.preprocess import LowercaseTransformer, \
        RegexpTokenizer, StopwordsFilter, FrequencyFilter
    from orangecontrib.text.vectorization import BowVectorizer

    corpus_ = Corpus.from_file("book-excerpts")
    for pp in (LowercaseTransformer(), RegexpTokenizer(r"\w+"),
               StopwordsFilter("English"), FrequencyFilter(0.1)):
        corpus_ = pp(corpus_)

    transformed_corpus = BowVectorizer().transform(corpus_)

    pca = PCA(n_components=2)
    pca_model = pca(transformed_corpus)
    projection = pca_model(transformed_corpus)

    domain = Domain(
        transformed_corpus.domain.attributes,
        transformed_corpus.domain.class_vars,
        chain(transformed_corpus.domain.metas,
              projection.domain.attributes)
    )
    corpus_ = corpus_.transform(domain)

    embedding_ = corpus_.metas[:, -2:]
    clusters_ = ClusterDocuments.gmm(embedding_, 3, 0.6)
    keywords_ = _get_characteristic_terms(corpus_, 4)
    clusters_keywords_, _, _, _ = \
        _hypergeom_clusters(clusters_, keywords_, 0.2, 5)
    concave_hulls_ = compute_hulls(embedding_, clusters_)
    centroids_ = {c: tuple(np.mean(concave_hulls_[c], axis=0))
                  for c in set(clusters_) - {-1}}

    palette = ["#377eb8", "#ff7f00", "#4daf4a", "#f781bf", "#a65628",
               "#984ea3", "#999999", "#e41a1c", "#dede00"]
    for label_ in sorted(set(clusters_)):
        mask = label_ == clusters_
        color = palette[label_] if label_ != -1 else (0.5, 0.5, 0.5)
        plt.scatter(embedding_[mask, 0], embedding_[mask, 1], c=color)

        if label_ == -1:
            continue

        x, y = centroids_[label_]
        kwds_ = [w for w, _ in clusters_keywords_[label_]]
        text = "\n".join(kwds_) + f"\n\n{label_ + 1}"
        plt.text(x, y, text, va="center", ha="center")

        hull = concave_hulls_[label_]
        plt.plot(hull[:, 0], hull[:, 1])

    plt.show()
