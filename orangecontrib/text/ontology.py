from typing import List, Set, Dict, Tuple, Optional, Callable
from collections import Counter
from itertools import chain
import os
import pickle

import numpy as np

from orangecontrib.text.vectorization.sbert import SBERT
from Orange.misc.environ import cache_dir
from Orange.util import dummy_callback, wrap_callback

EMB_DIM = 384


class Tree:

    def __init__(self, adj_list: List[Set[int]], labels: List[str], root: int) -> None:
        assert (
            set([neighbor for node in adj_list for neighbor in node]) ==
            set(range(len(adj_list)))
        ), "Vertices must be enumerated from 0 to n."
        assert len(adj_list) == len(labels), "adj_list and labels must be of same length"
        self.adj_list = adj_list
        self.labels = labels
        self.root = root

    @classmethod
    def from_dict(cls, tree: Dict):
        assert len(tree.keys()) == 1, "Not a tree"

        def _from_dict(tree, root, edgelist, labels):
            labels.append(root)
            idx = len(labels) - 1
            for i, neighbor in enumerate(tree[root]):
                edgelist.append((idx, len(labels)))
                _from_dict(tree[root], neighbor, edgelist, labels)

        edgelist = list()
        labels = list()
        _from_dict(tree, list(tree.keys())[0], edgelist, labels)
        adj_list = [set() for _ in range(len(labels))]
        for u, v in edgelist:
            adj_list[u].add(v)
            adj_list[v].add(u)

        return Tree(adj_list, labels, 0)

    @classmethod
    def from_prufer_sequence(cls, sequence: List[int], labels: List[str], root: int = None):
        n_nodes = len(sequence) + 2
        adj_list = [set() for _ in range(n_nodes)]
        degrees = Counter(chain(sequence, range(n_nodes)))
        idx = u = next(i for i in range(n_nodes) if degrees[i] == 1)
        nodes = set(range(n_nodes))
        for v in sequence:
            nodes.remove(u)
            adj_list[u].add(v)
            adj_list[v].add(u)
            degrees[v] -= 1
            if v < idx and degrees[v] == 1:
                u = v
            else:
                idx = u = next(i for i in range(idx + 1, n_nodes) if degrees[i] == 1)
        u, v = nodes
        adj_list[u].add(v)
        adj_list[v].add(u)

        return Tree(adj_list, labels, root)

    def to_dict(self) -> Dict:

        def _to_dict(adj_list, labels, root, prev=None):
            tree = dict()
            neighbors = [node for node in adj_list[root] if node != prev]
            for neighbor in neighbors:
                tree[labels[neighbor]] = _to_dict(adj_list, labels, neighbor, prev=root)
            return tree

        return {self.labels[self.root]: _to_dict(self.adj_list, self.labels, self.root)}


class FitnessFunction:

    def __init__(
        self,
        words,
        similarities,
        min_children: int = 2,
        max_children: int = 7,
        parent_sim_weight: float = 0.5,
        sibling_sim_weight: float = 0.3,
        n_children_weight: float = 0.1,
        length_intersection_weight: float = 0.1,
    ):
        self.words = words
        self.sims = similarities
        self.min_children = min_children
        self.max_children = max_children
        self.parent_sim_weight = parent_sim_weight
        self.sibling_sim_weight = sibling_sim_weight
        self.n_children_weight = n_children_weight
        self.length_intersection_weight = length_intersection_weight

    def fitness(self, tree: Tree, root: Optional[int] = None) -> Tuple[float, int]:
        score_list = list()
        if root is None:
            root = np.argmin([
                np.min(self.sims[i, list(tree.adj_list[i])])
                for i in range(self.sims.shape[0])
            ])
        self.score(tree.adj_list, root, score_list)
        max_level = np.max([s[1] for s in score_list])
        weights = np.array([s[1] / max_level for s in score_list])
        weights = weights / sum(weights)
        return np.sum([s[0] * w for w, s in zip(weights, score_list)]), root

    def score(
        self,
        adj_list: List[Set[int]],
        root: int,
        scores: List[Tuple[float, int]],
        prev: Optional[int] = None,
        level: int = 1
    ) -> None:
        children = [n for n in adj_list[root] if n != prev]
        if len(children) == 0:
            return

        parent_score = 0
        sibling_score = 0
        int_len_score = 0
        for i in range(len(children)):
            child1 = children[i]
            parent_score += self.sims[root, child1]
            root_words = self.words[root].split(' ')
            child1_words = self.words[child1].split(' ')
            int_len_score += (
                (len(root_words) <= len(child1_words)) and
                (len(set(root_words).intersection(set(child1_words))) > 0)
            )
            for j in range(i + 1, len(children)):
                child2 = children[j]
                sibling_score += self.sims[child1, child2]

        if len(children) > 1:
            parent_score = parent_score / len(children)
            sibling_score = sibling_score / (len(children) * (len(children) - 1) / 2)
            int_len_score = int_len_score / len(children)

        n_children_term = (
            1 if len(children) in range(self.min_children, self.max_children + 1) else -1
        )
        scores.append((
            self.parent_sim_weight * parent_score +
            self.sibling_sim_weight * sibling_score +
            self.n_children_weight * n_children_term +
            self.length_intersection_weight * int_len_score, level)
        )
        for child in children:
            self.score(adj_list, child, scores, root, level+1)


def generate_ontology(
    words: List[str],
    sims: np.array,
    num_generations: int = 500,
    population_size: int = 100,
    crossover_probability: float = 0.8,
    mutation_probability: float = 0.1,
    keep_next_gen: int = 20,
    callback: Callable = dummy_callback
) -> Tuple[np.array, int]:
    seq_len = sims.shape[0] - 2
    max_val = sims.shape[0]
    rng = np.random.default_rng(0)
    generation = rng.integers(max_val, size=(population_size, seq_len))
    fitness_function = FitnessFunction(words, sims).fitness
    ticks = iter(np.linspace(0, 1.0, num_generations))

    for _ in range(num_generations):
        callback(next(ticks))
        fitness = np.array([
            fitness_function(Tree.from_prufer_sequence(generation[i, :], words))[0]
            for i in range(population_size)
        ])
        fitness = fitness / fitness.sum()
        new_generation = np.zeros(generation.shape, dtype=int)
        parents_to_keep = np.argsort(fitness)[-keep_next_gen:]
        new_generation[:keep_next_gen, :] = generation[parents_to_keep, :]
        for i in range(keep_next_gen, population_size):
            if rng.uniform() > crossover_probability:
                choice = rng.choice(
                    np.arange(len(fitness)),
                    size=1,
                    p=fitness,
                    replace=False
                )[0]
                new_generation[i, :] = generation[choice, :]
            else:
                u, v = rng.choice(
                    np.arange(len(fitness)),
                    size=2,
                    p=fitness,
                    replace=False
                )
                point = rng.integers(seq_len // 2)
                new_generation[i, :point] = generation[u, :point]
                new_generation[i, point:point+seq_len//2] = generation[v, point:point+seq_len//2]
                new_generation[i, point+seq_len//2:] = generation[u, point+seq_len//2:]
            mutation_idx = np.where(rng.uniform(size=seq_len) < mutation_probability)[0]
            new_generation[i, mutation_idx] = rng.integers(seq_len, size=len(mutation_idx))
        generation = new_generation

    # select the best solution
    scores, roots = list(), list()
    for i in range(population_size):
        score, root = fitness_function(Tree.from_prufer_sequence(generation[i, :], words))
        scores.append(score)
        roots.append(root)
    best = np.argmax(scores)
    return generation[best, :], roots[best]


def cos_sim(x: np.array, y: np.array) -> float:
    dot = np.dot(x, y)
    return 0 if np.allclose(dot, 0) else dot / (np.linalg.norm(x) * np. linalg.norm(y))


class EmbeddingStorage:

    def __init__(self):
        self.cache_dir = os.path.join(cache_dir(), 'ontology')
        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.similarities = dict()
        try:
            with open(os.path.join(self.cache_dir, 'sims.pkl'), 'rb') as file:
                self.similarities = pickle.load(file)
        except IOError:
            self.similarities = dict()
        self.embeddings = dict()

    def save_similarities(self):
        with open(os.path.join(self.cache_dir, 'sims.pkl'), 'wb') as file:
            pickle.dump(self.similarities, file)

    def get_embedding(self, word: str) -> Optional[np.array]:
        if word in self.embeddings:
            return self.embeddings[word]
        try:
            emb = np.load(os.path.join(self.cache_dir, f'{word}.npy'))
            self.embeddings[word] = emb
            return emb
        except IOError:
            return None

    def save_embedding(self, word: str, emb: np.array) -> None:
        self.embeddings[word] = emb
        np.save(os.path.join(self.cache_dir, f'{word}.npy'), emb)

    def clear_storage(self) -> None:
        self.similarities = dict()
        self.embeddings = dict()
        if os.path.isdir(self.cache_dir):
            for file in os.listdir(self.cache_dir):
                os.remove(os.path.join(self.cache_dir, file))


class OntologyHandler:

    def __init__(self):
        self.embedder = SBERT()
        self.storage = EmbeddingStorage()

    def generate(
        self,
        words: List[str],
        callback: Callable = dummy_callback
    ) -> Dict:
        if len(words) == 0:
            return {}
        if len(words) == 1:
            return {words[0]: {}}
        if len(words) == 2:
            return {sorted(words)[0]: {sorted(words)[1]: {}}}
        sims = self._get_similarities(
            words,
            self._get_embeddings(words, wrap_callback(callback, end=0.1)),
            wrap_callback(callback, start=0.1, end=0.2)
        )
        if len(words) == 3:
            root = np.argmin(np.sum(sims, axis=1))
            rest = sorted([words[i] for i in range(3) if i != root])
            return {words[root]: {rest[0]: {}, rest[1]: {}}}
        ontology, root = generate_ontology(
            words,
            sims,
            callback=wrap_callback(callback, start=0.2)
        )
        return Tree.from_prufer_sequence(ontology, words, root).to_dict()

    def insert(
        self,
        tree: Dict,
        words: List[str],
        callback: Callable = dummy_callback
    ) -> Dict:
        tree = Tree.from_dict(tree)
        self._get_embeddings(words, wrap_callback(callback, end=0.3))
        ticks = iter(np.linspace(0.3, 0.9, len(words)))

        for word in words:
            tick = next(ticks)
            tree.adj_list.append(set())
            tree.labels.append(word)
            sims = self._get_similarities(
                tree.labels,
                self._get_embeddings(tree.labels, lambda x: callback(tick)),
                lambda x: callback(tick)
            )
            idx = len(tree.adj_list) - 1
            fitness_function = FitnessFunction(tree.labels, sims).fitness
            scores = list()
            for i in range(idx):
                tree.adj_list[i].add(idx)
                tree.adj_list[idx].add(i)
                scores.append(fitness_function(tree, tree.root)[0])
                tree.adj_list[i].remove(idx)
                tree.adj_list[idx].remove(i)
            best = np.argmax(scores)
            tree.adj_list[best].add(idx)
            tree.adj_list[idx].add(best)
            callback(tick)

        return tree.to_dict()

    def score(self, tree: Dict, callback: Callable = dummy_callback) -> float:
        tree = Tree.from_dict(tree)
        sims = self._get_similarities(
            tree.labels,
            self._get_embeddings(tree.labels, wrap_callback(callback, end=0.7)),
            wrap_callback(callback, start=0.7, end=0.8)
        )
        callback(0.9)
        fitness_function = FitnessFunction(tree.labels, sims).fitness
        return fitness_function(tree, tree.root)[0]

    def _get_embeddings(
        self,
        words: List[str],
        callback: Callable = dummy_callback
    ) -> np.array:
        embeddings = np.zeros((len(words), EMB_DIM))
        missing, missing_idx = list(), list()
        ticks = iter(np.linspace(0.0, 0.6, len(words)))
        for i, word in enumerate(words):
            callback(next(ticks))
            emb = self.storage.get_embedding(word)
            if emb is None:
                missing.append(word)
                missing_idx.append(i)
            else:
                embeddings[i, :] = emb
        if len(missing_idx) > 0:
            embs = self.embedder(missing, callback=wrap_callback(callback, start=0.6, end=0.9))
            if None in embs:
                raise RuntimeError("Couldn't obtain embeddings.")
            embeddings[missing_idx, :] = np.array(embs)
        for i in missing_idx:
            self.storage.save_embedding(words[i], embeddings[i, :])

        return embeddings

    def _get_similarities(
        self,
        words: List[str],
        embeddings: np.array,
        callback: Callable = dummy_callback
    ) -> np.array:
        sims = np.zeros((len(words), len(words)))
        ticks = iter(np.linspace(0.0, 1.0, int(len(words) * (len(words) - 1) / 2)))
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                callback(next(ticks))
                key = tuple(sorted((words[i], words[j])))
                try:
                    sim = self.storage.similarities[key]
                except KeyError:
                    sim = cos_sim(embeddings[i, :], embeddings[j, :])
                    self.storage.similarities[key] = sim
                sims[i, j] = sim
                sims[j, i] = sim
        self.storage.save_similarities()
        return sims
