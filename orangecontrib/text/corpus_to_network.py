from collections import defaultdict
from typing import Optional, Callable
import numpy as np
from scipy.sparse import csr_matrix

from Orange.data import Domain, ContinuousVariable, Table, StringVariable
from orangecontrib.text import Corpus
from orangecontrib.network import Network


class CorpusToNetwork:
    """This class can be used to obtain networks from given corpus.
    For more information on usage see documentation of __call__.

    Parameters
    ----------
    corpus: Corpus
        Corpus on which the operations are performed.
    """

    def __init__(self, corpus: Corpus) -> None:
        if not isinstance(corpus, Corpus):
            raise ValueError("Given parameter must be instance of Corpus.")
        self.corpus = corpus
        self.document_matrix = None
        self.word_matrix = None
        self.window_size = 0
        self.document_threshold = 0
        self.word_threshold = 0
        self.freq_threshold = 0
        self.word_network = None
        self.document_network = None
        self.word2ind = None
        self.word_freqs = None
        self.document_items = Table(corpus.domain,
                                    corpus.X,
                                    corpus.Y,
                                    corpus.metas)
        self.word_items = None
        self.num_ngrams = 0
        self.last_called_nodes = True
        self.param_cache = list()
        self.mask = None
        self.ngram_list = list(self.corpus.ngrams)

    def __call__(self, document_nodes: bool = True,
                 window_size: int = 1, threshold: int = 1,
                 freq_threshold: int = 1,
                 progress_callback: Optional[Callable] = None) -> Network:
        """Constructs network from a corpus. Network nodes can be either
        documents or words. Two document nodes are connected if the number
        of words (ngrams) they share is larger (or equal to) than threshold.
        Two words (ngrams) co-occur if they are located inside a window
        defined by window_size within a single document in corpus.
        Two word nodes are connected if number of their co-occurences
        is larger (or equal to) than threshold. In first case edge weight
        represents number of words that two documents share and in second case
        weight is number of co-occurences of two words (ngrams)
        in a corpus.

        Parameters
        ----------
        document_nodes : bool
            If True, the nodes of returned network will be documents, otherwise
            words.
        window_size: int
            Size of window (actual size of window will be
            2 * window_size + 1). Applies only if document_nodes=False.
        threshold: int
            Threshold that determines if edge between two nodes should exist.
        freq_threshold: int
            Ignore words with frequency smaller than freq_threshold.
            Applies only if document_nodes=False.
        progress_callback: Callable
            Callback that control progress bar of Orange widget. If calling
            from script, ignore this parameter.

        Returns
        -------
        Network
            Network obtained from corpus with given parameters.
        Raises
        ------
        ValueError
            If threshold <= 0 or window_size <= 0 or freq_threshold <= 0
            or if document_nodes is not instance of bool.
        """

        if threshold <= 0:
            raise ValueError("Threshold must be at least 1.")
        if window_size <= 0:
            raise ValueError("Window size must be at least 1.")
        if freq_threshold <= 0:
            raise ValueError("Frequency threshold must be at least 1.")
        if not isinstance(document_nodes, bool):
            raise ValueError("Document_nodes must be bool.")

        self._cache_params()  # remember old parameters
        try:
            self.last_called_nodes = document_nodes
            if (document_nodes and
                    threshold == self.document_threshold and
                    self.document_network is not None):
                # same network already exists
                if progress_callback:
                    progress_callback(100.0)
                return self.document_network
            if (not document_nodes and
                    threshold == self.word_threshold and
                    window_size == self.window_size and
                    freq_threshold == self.freq_threshold and
                    self.word_network is not None):
                # same network already exists
                if progress_callback:
                    progress_callback(100.0)
                return self.word_network
            if document_nodes:
                self.document_threshold = threshold
                if self.document_matrix is None:
                    self._compute_document_matrix(progress_callback)  # generate adjacency matrix
                self._generate_document_network(progress_callback)  # construct network
                return self.document_network
            else:
                if (window_size != self.window_size or
                        self.word_matrix is None or
                        freq_threshold != self.freq_threshold):
                    if self.word2ind is None:
                        self._compute_word2ind(progress_callback)
                    self.window_size = window_size
                    self.freq_threshold = freq_threshold
                    self._compute_word_matrix(progress_callback)  # generate adjacency matrix
                self.word_threshold = threshold
                self._generate_word_network(progress_callback)  # construct network
                return self.word_network
        except Exception:
            self._restore_params()

    def _compute_document_matrix(self, progress_callback):
        self.document_matrix = np.zeros((len(self.corpus), len(self.corpus)))
        num_ticks = (len(self.corpus) * (len(self.corpus) - 1)) // 2
        ticks = iter(np.linspace(0., 90., num_ticks))
        for i in range(len(self.corpus)):
            for j in range(i + 1, len(self.corpus)):
                w = set(self.ngram_list[i]).intersection(
                    set(self.ngram_list[j]))
                self.document_matrix[i][j] = len(w)
                if progress_callback:
                    progress_callback(next(ticks))

    def _generate_document_network(self, progress_callback):
        if progress_callback:
            progress_callback(90.0)
        edges = self.document_matrix.copy()
        edges[edges < self.document_threshold] = 0
        self.document_network = Network(nodes=np.array(self.corpus.titles),
                                        edges=csr_matrix(edges),
                                        name='Document Network')
        if progress_callback:
            progress_callback(100.0)

    def _compute_word2ind(self, progress_callback):
        self.word2ind = dict()
        self.word_freqs = defaultdict(int)
        self.num_ngrams = 0
        i = 0
        ticks = iter(np.linspace(0., 10., len(self.corpus)))
        for ngrams in self.ngram_list:
            self.num_ngrams += len(ngrams)
            for ngram in ngrams:
                self.word_freqs[ngram] += 1
                if ngram not in self.word2ind:
                    self.word2ind[ngram] = i
                    i += 1
            if progress_callback:
                progress_callback(next(ticks))
        self.word2ind = dict(sorted(self.word2ind.items(), key=lambda x: x[1]))
        if progress_callback:
            progress_callback(10.0)

    def _compute_word_matrix(self, progress_callback):
        self.mask = np.full(len(self.word2ind), True)
        ticks = iter(np.linspace(0., 90., self.num_ngrams))
        self.word_matrix = defaultdict(int)  # data for sparse matrix
        for ngrams in self.ngram_list:
            for i in range(len(ngrams)):
                # one window
                if self.word_freqs[ngrams[i]] < self.freq_threshold:
                    # ignore the ngram if frequency is not big enough
                    self.mask[self.word2ind[ngrams[i]]] = False
                    continue
                left = 0 if i - self.window_size < 0 else i - self.window_size
                right = ((len(ngrams) - 1)
                         if i + self.window_size > len(ngrams) - 1
                         else i + self.window_size)
                for j in range(left, i):
                    if ((self.word2ind[ngrams[j]], self.word2ind[ngrams[i]])
                            in self.word_matrix):
                        self.word_matrix[(self.word2ind[ngrams[j]],
                                          self.word2ind[ngrams[i]])] += 1
                    else:
                        self.word_matrix[(self.word2ind[ngrams[i]],
                                          self.word2ind[ngrams[j]])] += 1
                for j in range(i + 1, right + 1):
                    if ((self.word2ind[ngrams[j]], self.word2ind[ngrams[i]])
                            in self.word_matrix):
                        self.word_matrix[(self.word2ind[ngrams[j]],
                                          self.word2ind[ngrams[i]])] += 1
                    else:
                        self.word_matrix[(self.word2ind[ngrams[i]],
                                          self.word2ind[ngrams[j]])] += 1
                if progress_callback:
                    progress_callback(next(ticks))

    def _generate_word_network(self, progress_callback):
        if progress_callback:
            progress_callback(90.0)
        th = self.word_threshold
        data = np.array([v / 2 for v in self.word_matrix.values() if (v / 2) >= th],
                        dtype=np.float64)
        row_ind = np.array([k[0] for k, v in self.word_matrix.items() if (v / 2) >= th],
                           dtype=np.float64)
        col_ind = np.array([k[1] for k, v in self.word_matrix.items() if (v / 2) >= th],
                           dtype=np.float64)
        s = len(self.word_freqs)
        edges = csr_matrix((data, (row_ind, col_ind)), shape=(s, s))
        ind2word = {v: k for k, v in self.word2ind.items()}
        words = np.array([ind2word[ind] for ind in range(s)])
        freqs = np.array([self.word_freqs[ind2word[ind]] for ind in range(s)],
                         dtype=np.float64)
        network = Network(nodes=words,
                          edges=edges,
                          name='Word Network')
        self.word_network = network.subgraph(self.mask)
        domain = Domain([ContinuousVariable('word_frequency')],
                        None,
                        [StringVariable('word')])
        self.word_items = Table(domain,
                                freqs.reshape((-1, 1))[self.mask],
                                None,
                                words.reshape((-1, 1))[self.mask])
        if progress_callback:
            progress_callback(100.0)

    def get_current_items(self, document_nodes):
        """Returns table containing data about nodes of
        currently saved network.

        Parameters
        ----------
        document_nodes : bool
            If True, returns data for document network, otherwise for word
            network.

        Returns
        -------
        Table
            Table with additional data about nodes.
        """
        return self.document_items if document_nodes else self.word_items

    def _cache_params(self):
        self.param_cache = list()
        self.param_cache.append(self.last_called_nodes)
        self.param_cache.append(self.document_threshold if self.last_called_nodes
                                else self.word_threshold)
        self.param_cache.append(self.window_size)
        self.param_cache.append(self.freq_threshold)

    def _restore_params(self):
        self.last_called_nodes = self.param_cache[0]
        if self.last_called_nodes:
            self.document_threshold = self.param_cache[1]
        else:
            self.word_threshold = self.param_cache[1]
        self.window_size = self.param_cache[2]
        self.freq_threshold = self.param_cache[3]
